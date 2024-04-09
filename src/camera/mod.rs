pub mod worker;

use anyhow::{Error, Result};
use log::{debug, error, info};
use opencv::core::Vector;
use opencv::imgcodecs::imencode;
use opencv::imgproc::{cvt_color, COLOR_BGR2GRAY};
use opencv::prelude::Mat;
use opencv::videoio::{
  self, VideoCapture, VideoCaptureAPIs, VideoCaptureTrait, VideoCaptureTraitConst,
};
use serde::{Deserialize, Serialize};
use std::fs;
use tokio::sync::mpsc::Sender;
use tokio::time::{Duration, Instant};

/// This configures encoding consumed video streams for which is
/// written to the output channel.
#[derive(Debug, Clone)]
pub struct CameraVideoEncoding {
  // Array of key value pairs from opencv's ImwriteFlags.
  pub encoding_flags: Vec<(i32, i32)>,

  // Image extension for encoding the video stream to.
  pub image_ext: String,

  // Whether to convert image to grayscale or not.
  pub grayscale: bool,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct CameraVideoProps {
  // Video properties. These are required to resize and process the
  // consumed video stream.
  pub image_width: f64,
  pub image_height: f64,
  pub video_fps: f64,
}

#[derive(Debug)]
pub struct CameraDevice {
  // OpenCV's Video Capture instance.
  video_capture: VideoCapture,

  // Encoding configurations used on consumed image Mat instances.
  video_encoding: CameraVideoEncoding,

  // Target consumption FPS, throttling consumption to meet requirement.
  target_fps: f64,
}

impl CameraDevice {
  /// Creates a new CameraDevice instance which reads from the given
  /// video device or video filepath.
  ///
  /// # Arguments
  /// * filepath - Filepath to video device or video file.
  /// * video_props - Video properties to configure video stream consumption.
  /// * video_encoding - Video encoding properties to apply on consumed image streams.
  ///
  /// # Returns
  /// * A CameraDevice instance.
  pub fn new(
    filepath: &str,
    video_props: CameraVideoProps,
    video_encoding: CameraVideoEncoding,
  ) -> Result<CameraDevice, String> {
    // Verify the camera fd exists.
    fs::metadata(filepath).map_err(|e| {
      // Log the failure and bubble up the error.
      error!("CameraDevice: Video filepath does not exist {}", e);
      e.to_string()
    })?;

    // https://docs.rs/opencv/0.88.1/opencv/videoio/enum.VideoCaptureAPIs.html
    let mut video_capture =
      VideoCapture::from_file(filepath, VideoCaptureAPIs::CAP_ANY.into()).map_err(|e| e.message)?;
    info!("CameraDevice: Instanitated video fd -> {:?}", video_capture);

    if !video_capture.is_opened().unwrap() {
      return Err(format!(
        "CameraDevice: VideoCapture failed to open {}",
        filepath
      ));
    }

    // Adjust the frame resolution.
    let _ = adjust_video_frame(&mut video_capture, &video_props).map_err(|e| e.to_string())?;

    Ok(Self {
      video_capture,
      video_encoding,
      target_fps: video_props.video_fps,
    })
  }

  /// Helper function which encodes the given Mat image instance to the Encoding
  /// configuration supplied to this instance.
  ///
  /// Args:
  /// * image_mat: Image of type Mat to apply encodings to.
  ///
  /// Returns:
  /// * Encoded image.
  fn encode_image_mat(&self, image_mat: &Mat) -> Result<Vec<u8>> {
    // Construct encoding flags into an OpenCV Vector.
    let image_enc_flags: Vector<i32> = Vector::from_iter(
      self
        .video_encoding
        .encoding_flags
        .iter()
        .flat_map(|(flag_name, flag_value)| [*flag_name, *flag_value])
        .collect::<Vec<i32>>(),
    );

    // Apply grayscale encoding if configured.
    let mut _image_mat = image_mat.clone();
    if self.video_encoding.grayscale {
      cvt_color(image_mat, &mut _image_mat, COLOR_BGR2GRAY, 0)?;
    }

    // Encode and return the image.
    let mut image_vec: Vector<u8> = Vector::new();
    return match imencode(
      &self.video_encoding.image_ext,
      &_image_mat,
      &mut image_vec,
      &image_enc_flags,
    ) {
      Ok(_) => Ok(image_vec.to_vec()),
      Err(err) => Err(err.into()),
    };
  }

  /// Starts consuming a stream of images from the current device, encodes the image,
  /// then pushes image results on the given sender channel.
  ///
  /// # Arguments:
  /// * chan_tx - Tokio Sender channel to push images to.
  pub fn start(&mut self, chan_tx: Sender<(Mat, Vec<u8>)>) -> Result<()> {
    // Matrix buffer for which to store captured images to.
    let mut image_buff = Mat::default();

    // Calculate FPS and expected elapsed seconds to yield target fps.
    // Expected elsapsed seconds maths:
    // 1F / nS = FPS
    // 1F / nS = target_fps
    // 1F / target_fps = nS <- target elapsed seconds.
    let mut start_time: Instant = Instant::now();
    let frame_delay: Duration = Duration::from_secs_f64(1.0 / self.target_fps);

    // Throughput statistics.
    let mut total_bytes_sent: usize = 0;
    let mut stat_check_time = Instant::now();

    // Loop over the video stream.
    // NOTE: If we sleep within this thread without constantly calling read,
    // io breaks internally causing the read to time out.
    while self.video_capture.read(&mut image_buff)? {
      // Check statistics every second.
      if stat_check_time.elapsed().as_millis() >= 1000 {
        let total_kb_sent = total_bytes_sent / 1024;
        info!(
          "CameraDevice: Produced ~{}KB/s @{}FPS",
          total_kb_sent, self.target_fps
        );

        // Reset.
        stat_check_time = Instant::now();
        total_bytes_sent = 0;
      }

      // Calculate the number of read frames per second.
      let elapsed_time: Duration = start_time.elapsed();
      let fps: f64 = math::round::floor(1.0 / elapsed_time.as_secs_f64(), 2);
      debug!(
        "CameraDevice: elapsed_seconds={}s | fps_now={}",
        elapsed_time.as_secs_f64(),
        fps,
      );

      // Consume images once target delay has been reached, which means that
      // we've met our target FPS.
      if elapsed_time >= frame_delay {
        // Reset timer.
        start_time = Instant::now();
        debug!(
          "CameraDevice: FPS target met -> elapsed_seconds={}s | fps_now={} | target_elapsed_seconds={}s | target_fps={}",
          elapsed_time.as_secs_f64(),
          fps,
          frame_delay.as_secs_f64(),
          self.target_fps,
        );

        // We've reached our FPS target, decode and send the image.
        match self.encode_image_mat(&image_buff) {
          Ok(encoded_image) => {
            // Track statistics.
            total_bytes_sent += encoded_image.len();

            // Send image to consumers.
            let chan_tx_res = chan_tx.blocking_send((image_buff.clone(), encoded_image));
            if let Err(err) = chan_tx_res {
              error!("CameraDevice: Failed to send image on channel -> {err}");
            }
          }
          Err(err) => {
            return Err(Error::msg(format!(
              "CameraDevice: Failed to encode image -> {err}"
            )));
          }
        }
      }
    }
    Ok(())
  }
}

/// Drop implementation for CameraDevice which acts like destructor
/// logic. This gets triggered when this instance goes out of scope.
impl Drop for CameraDevice {
  /// This gets invoked when the instance goes out of scope.
  fn drop(&mut self) {
    info!(
      "CameraDevice: Releasing VideoCapture instance -> {:?}",
      self.video_capture
    );
    self
      .video_capture
      .release()
      .expect("CameraDevice: failed to release video capture");
  }
}

/// Adjusts the VideoCapture instance to match expected set configurations.
///
/// # Arguments:
/// * video_cap - A VideoCapture reference for which to configure.
/// * video_props - Video properties to configure video stream consumption with.
fn adjust_video_frame(video_cap: &mut VideoCapture, video_props: &CameraVideoProps) -> Result<()> {
  info!(
    "CameraDevice: [Success={}] Set 'CAP_PROP_FPS' -> {}",
    video_cap.set(videoio::CAP_PROP_FPS, video_props.video_fps)?,
    video_props.video_fps,
  );
  info!(
    "CameraDevice: [Success={}] Set 'CAP_PROP_FRAME_WIDTH' -> {}",
    video_cap.set(videoio::CAP_PROP_FRAME_WIDTH, video_props.image_width)?,
    video_props.image_width,
  );
  info!(
    "CameraDevice: [Success={}] Set 'CAP_PROP_FRAME_HEIGHT' -> {}",
    video_cap.set(videoio::CAP_PROP_FRAME_HEIGHT, video_props.image_height)?,
    video_props.image_height,
  );
  info!(
    "CameraDevice: [Success={}] Set 'CAP_PROP_HW_ACCELERATION' -> VIDEO_ACCELERATION_ANY(prefer HW Acceleration)",
    video_cap.set(videoio::CAP_PROP_HW_ACCELERATION, videoio::VIDEO_ACCELERATION_ANY as f64)?,
  );
  Ok(())
}
