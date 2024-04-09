use crate::api::interfaces::{CameraBaseData, CameraStreamResponse};
use crate::api::AdjustedCameraBuffer;
use crate::camera;
use crate::config::ConfigLocalVideo;

use log::{debug, error, info};
use opencv::imgcodecs::ImwriteFlags;
use opencv::prelude::{Mat, MatTraitConst};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Starts local camera ingestion client tokio tasks.
///   [/dev/videoN -> video_adjustments -> tx_channel]
///
/// This consumes images from the given video filepath and then sends the
/// image on the given Sender channel.
///
/// Args:
/// * video_config: Parsed local video configuration instance.
/// * produced_images_tx: Sender channel for writing consumed images to.
///
/// Returns:
/// * The video consumption and adjustment task workers.
pub async fn start_local_video_worker(
  video_config: &ConfigLocalVideo,
  produced_images_tx: mpsc::Sender<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
  let video_fd = video_config.video_fd.clone();
  let video_props = video_config.video_properties.clone();
  let video_chan_buff_size = video_config.video_stream_channel_size;

  // Setup encoding flags such that we optimize for size.
  let video_encodings = camera::CameraVideoEncoding {
    encoding_flags: vec![
      (ImwriteFlags::IMWRITE_JPEG_OPTIMIZE as i32, 1),
      (ImwriteFlags::IMWRITE_JPEG_QUALITY as i32, 75),
    ],
    image_ext: ".jpeg".to_string(),

    // TODO: fix server to allow grayscale?
    grayscale: false,
  };
  let video_enc_copy = video_encodings.clone();

  let video_capture = camera::CameraDevice::new(&video_fd, video_props, video_encodings).unwrap();

  // Create a multi-producer single consumer channel for which to produce images from multiple sources
  // and then consume and classify those images in a single worker.
  let (tx, mut rx) = tokio::sync::mpsc::channel(video_chan_buff_size);

  info!("VideoDeviceIngestor: Starting video capture");
  let video_consumer_task = tokio::task::spawn_blocking(|| {
    let mut video_fd = video_capture;
    match video_fd.start(tx) {
      Err(err) => {
        error!("VideoDeviceIngestor: Failed video feed -> {err:?}");
      }
      _ => {}
    }
  });

  info!("VideoDeviceIngestor: Consuming images from channel");
  let video_fd_clone = video_fd.clone();
  let video_adjustment_task = tokio::task::spawn(async move {
    // Consume all of the buffered images in the channel.
    let mut img_buffer: Vec<(Mat, Vec<u8>)> = vec![];

    // Construct device values such that we can create a hashmap to be passed into
    // the channel.
    let dev_filename: String = std::path::Path::new(&video_fd_clone)
      .file_name()
      .and_then(|file_name| file_name.to_str())
      .unwrap_or("video0")
      .to_string();

    // Unused empty map, we're taking advantage of the adjusted instance within
    // the channel to be classified.
    let mut device_map: HashMap<String, CameraBaseData> = HashMap::new();
    device_map.insert(
      dev_filename.clone(),
      CameraBaseData {
        name: dev_filename.clone(),
        data_b64: "".to_string(),
      },
    );

    while rx.recv_many(&mut img_buffer, video_chan_buff_size).await > 0 {
      debug!("VideoDeviceIngestor: Received images[{}]", img_buffer.len());

      // Convert the image buffers into adjusted camera buffers.
      let adjusted_camera_buffers: Vec<AdjustedCameraBuffer> = img_buffer
        .iter()
        .map(|(image_mat, image_vec)| {
          let image_vec_size_kb: f64 = math::round::floor(image_vec.len() as f64 / 1024.0, 2);
          // Info status logs.
          let image_size = image_mat.size().unwrap_or_default();
          debug!(
            "VideoDeviceIngestor: Image Encoded -> ext={} | is_grayscale={} | vec={}KB | size{}x{}",
            video_enc_copy.image_ext,
            video_enc_copy.grayscale,
            image_vec_size_kb,
            image_size.width,
            image_size.height,
          );

          // Construct an adjusted instance for which is used for classification.
          AdjustedCameraBuffer {
            cam_ip: "127.0.0.1".to_string(),
            cam_name: dev_filename.clone(),
            device_name: dev_filename.clone(),
            image_buff: image_vec.clone(),
          }
        })
        .collect();

      // Send a defaulted stream response.
      let stream_response = CameraStreamResponse {
        cameras: device_map.clone(),
      };

      let chan_tx_res = produced_images_tx
        .send((stream_response, Some(adjusted_camera_buffers)))
        .await;
      if let Err(err) = chan_tx_res {
        error!("VideoDeviceIngestor: Failed to send consumed image on channel -> {err}");
        break;
      }

      img_buffer.clear()
    }
    info!("VideoDeviceIngestor: Receive channel closed");
  });

  return (video_consumer_task, video_adjustment_task);
}
