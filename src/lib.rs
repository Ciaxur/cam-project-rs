pub mod api;
pub mod camera;
pub mod config;
pub mod data_struct;

use anyhow::{Error, Result};
use api::classifier_grpc::{
  classifier::{ClassifyImageRequest, ClassifyImageResponse},
  ClassifierClient,
};
use api::interfaces::{CameraBaseData, CameraStreamResponse};
use api::storage_manager::{StorageManager, StorageManagerFile};
use api::{AdjustedCameraBuffer, CameraApi, CameraApiOptions};
use clap::Parser;
use log::{debug, error, info, warn};
use opencv::imgcodecs::ImwriteFlags;
use opencv::prelude::{Mat, MatTraitConst};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};

use crate::config::Config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
  /// File path to configuration json file.
  /// See config/mod.rs for more info.
  #[arg(short, long, required = true)]
  config: String,
}

/// Starts local camera ingestion client tokio tasks.
///
/// This consumes images from the given video filepath and then sends the
/// image on the given Sender channel.
///
/// Args:
/// * config: Parsed configuration instance.
/// * produced_images_tx: Sender channel for writing consumed images to.
///
/// Returns:
/// * The video consumption and adjustment task workers.
async fn start_local_video_ingestor(
  config: &Config,
  produced_images_tx: mpsc::Sender<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
  let video_fd = config.video_fd.clone();
  let video_props = config.camera_properties.clone();

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
  let (tx, mut rx) = tokio::sync::mpsc::channel(128);

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

    while rx.recv_many(&mut img_buffer, 128).await > 0 {
      debug!("VideoDeviceIngestor: Received images[{}]", img_buffer.len());

      // Convert the image buffers into adjusted camera buffers.
      let adjusted_camera_buffers: Vec<AdjustedCameraBuffer> = img_buffer
        .iter()
        .map(|(image_mat, image_vec)| {
          let image_vec_size_kb: f64 = math::round::floor(image_vec.len() as f64 / 1024.0, 2);
          // Info status logs.
          let image_size = image_mat.size().unwrap_or_default();
          info!(
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

/// Starts the Classifier client within a tokio task, retruning the task instance.
///
/// The function expects the server's endpoint for which to connect to for classifying
/// images retrieved from the classify request channel and then writes the response to
/// the sender channel.
///
/// Args:
/// * camera_api: Camera API instance reference.
/// * config: Parsed configuration instance.
/// * classify_response_tx: Sender channel used to write image classification responses to.
/// * classify_response_rx: Receiver channel used to consume response from.
/// * classify_request_rx: Receiver channel used to consume requests to be proxied and classified
///     by the sever.
///
/// Returns:
/// * Tokio task handler.
async fn start_classifier_client(
  client_api: &CameraApi,
  config: &Config,
  classify_response_tx: mpsc::Sender<ClassifyImageResponse>,
  mut classify_response_rx: mpsc::Receiver<ClassifyImageResponse>,
  classify_request_rx: mpsc::Receiver<ClassifyImageRequest>,
) -> Result<tokio::task::JoinHandle<()>> {
  let telegram_chat_id = config.camera_api.telegram_chat_id;
  let grpc_server_endpoint = config.grpc_server_endpoint.clone();
  let image_storage_path = config.image_storage_path.clone();
  let mut classifier_client =
    ClassifierClient::new(grpc_server_endpoint, config.camera_properties.clone());
  let classifier_client_api = client_api.clone();

  // Cooldown after classifying a hooman. Construct a map which tracks
  // each the last notification time.
  let cooldown_duration: Duration = Duration::from_secs(config.cooldown_s);
  let mut last_notify_map: HashMap<String, Instant> = HashMap::new();

  // Create a StorageManager instance to save classifier matched images to.
  info!("Created a Storage manager instance -> {image_storage_path}");
  let mut storage_manager = StorageManager::new(image_storage_path, 2048);
  let storage_manager_tx = storage_manager.start();

  // Accepted classified matches.
  let accepted_matches: Vec<u64> = config.accepted_match_labels.to_vec();

  // Start the classifier client, grabbing channels for which to consumer and
  // send images to be classified.
  classifier_client
    .start(classify_response_tx, classify_request_rx)
    .await
    .map_err(|err| Error::msg(format!("ClassifierClient: Failed to start -> {err}")))?;

  let classifier_consumer_task = tokio::task::spawn(async move {
    // Create a buffer to consume multiple classified images.
    let mut classified_images_buffer: Vec<ClassifyImageResponse> = vec![];

    // Start consuming responses from the server.
    info!("ClassifierClient: Started consuming classified images");
    while classify_response_rx
      .recv_many(&mut classified_images_buffer, 128)
      .await
      > 0
    {
      debug!(
        "ClassifierClient: Received {} classified images",
        classified_images_buffer.len()
      );

      // Iterate over classified images, pushing notifications as
      // appropriate.
      for image in classified_images_buffer.to_vec() {
        if image
          .matches
          .into_iter()
          .any(|v| accepted_matches.to_owned().contains(&(v as u64)))
        {
          // Save image to local disk.
          let status = storage_manager_tx
            .send(StorageManagerFile {
              data: image.image.to_vec(),
              device_name: image.device.clone(),
              ext: "jpeg".to_string(),
            })
            .await;
          if let Err(err) = status {
            error!("ClassifierClient: Failed to save image to disk -> {err}");
          }

          // Check cooldown map on device, skipping if cooldown not yet expired.
          let device_last_notified = last_notify_map.get(&image.device);
          if let Some(last_notified) = device_last_notified {
            if last_notified.elapsed() < cooldown_duration {
              continue;
            }
          }

          // Notify the client about the classified image.
          info!(
            "ClassifierClient: {} match found, issuing notification",
            image.device
          );

          let datetime_now = chrono::Local::now().to_rfc2822().to_string();
          let msg = format!(
            "[Device={}|Scores={:?}] Hoooman detected at {}.",
            image.device, image.match_scores, datetime_now,
          );

          let notify_state = classifier_client_api
            .to_owned()
            .push_msg_notification(telegram_chat_id, msg, image.image)
            .await;

          if let Err(err) = notify_state {
            error!("ClassifierClient: Failed to push notification -> {err}");
          }

          // Update cooldown entry in device notify map.
          last_notify_map.insert(image.device, Instant::now());
        }
      }

      // Buffer consumed, clear it.
      classified_images_buffer.clear();
    }

    warn!("ClassifierClient: Terminating image classifier consumer");
    classifier_client.stop();
    storage_manager.stop();
  });

  Ok(classifier_consumer_task)
}

/// Starts the Camera API client within two tokio tasks (consumer, producer) for which
/// consumes images from an open stream, from the given api endpoint, writing those images
/// to a channel. Those images are then consumed to apply adjustments to, like pre-processing,
/// for which to then write to a Sender channel to be classified.
///
/// Args:
/// * camera_api: Camera API instance reference.
/// * camera_api_tx: Sender channel for writing consumed images to.
/// * camera_api_rx: Receiver channel for consuming images from to apply adjustments to.
/// * classify_request_tx: Sender channel for writing adjusted images being classified.
///
/// Returns:
/// * A Tokio task for the producer.
/// * A Tokio task for the consumer.
async fn start_camera_api_client(
  client_api: &CameraApi,
  camera_api_tx: mpsc::Sender<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
  mut camera_api_rx: mpsc::Receiver<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
  classify_request_tx: mpsc::Sender<ClassifyImageRequest>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
  // CameraAPI Producer.
  let mut producer_client_api = client_api.clone();
  let producer_task = tokio::task::spawn(async move {
    info!("CameraAPI Client: Started client stream");

    loop {
      let status = producer_client_api
        .run_camera_stream(camera_api_tx.clone())
        .await
        .map_err(|err| err.to_string());

      match status {
        Ok(_) => {
          warn!("CameraAPI Client: Exited, likely due to timeout, restart client");
        }
        Err(err) => {
          error!("CameraAPI Client: {err}");
          if err.contains("error reading a body from connection") {
            info!("CameraAPI Client: Restarting camera stream");
            continue;
          }
          break;
        }
      }
    }

    warn!("CameraAPI Client: Terminating producer");
  });

  // CameraAPI Consumer.
  let consumer_task = tokio::task::spawn(async move {
    info!("CameraAPI Client: Started image stream consumer");
    let mut buffer: Vec<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)> = vec![];

    while camera_api_rx.recv_many(&mut buffer, 20).await > 0 {
      // Interate over all images, applying adjustments.
      // TODO: optimize to not do a copy.
      for (image, adjusted_image) in buffer.to_owned() {
        if let Some(a_img_buffs) = adjusted_image {
          for a_img in a_img_buffs {
            info!(
              "CameraAPI Client: Consumed adjusted camera buffer {}[{}]",
              a_img.cam_name, a_img.cam_ip,
            );

            let send_status = classify_request_tx.send(ClassifyImageRequest {
              image: a_img.image_buff,
              device: a_img.device_name,
            });

            if let Err(err) = send_status.await {
              error!("CameraAPI Client: Failed to write adjusted image to classification channel -> {err}");
              break;
            }
          }
        } else {
          for (cam_ip, cam_elt) in image.cameras {
            warn!(
              "CameraAPI Client: Consumed un-adjusted camera buffer {}[{}]",
              cam_elt.name, cam_ip
            );
          }
        }
      }

      buffer.clear();
    }

    info!("CameraAPI Client: Terminating consumer");
  });

  return (producer_task, consumer_task);
}

pub async fn run() -> Result<(), String> {
  // Grab passed in cli args, forwarding them to the main execution logic.
  let args = Args::parse();
  info!("Using configuration file from -> {}", args.config);
  let config = Config::parse(args.config.clone())
    .map_err(|err| format!("Configuration parse of '{}' failed: {}", args.config, err))?;
  debug!("Configuration loaded -> {config:?}");

  // Construct a shared camera client api.
  let client_api = CameraApi::new(CameraApiOptions {
    endpoint: config.camera_api.endpoint.clone(),
    target_fps: config.camera_api.target_fps,
    credentials: api::ClientCredentials {
      trusted_ca_filepath: config.camera_api.trusted_ca_filepath.clone(),
      identity_cert_filepath: config.camera_api.identity_cert_filepath.clone(),
      identity_key_filepath: config.camera_api.identity_key_filepath.clone(),
    },
  })
  .expect("Client instantiation failed");

  // Create two channels for which to allow other workers to issue requests
  // to and receive responses from.
  let (classify_request_tx, classify_request_rx) = mpsc::channel::<ClassifyImageRequest>(128);
  let (classify_response_tx, classify_response_rx) = mpsc::channel::<ClassifyImageResponse>(128);

  // Multi-producer single-consumer channel.
  // These channels are used to produce and consume images from the client api and
  // the local video device ingestor.
  let (consumed_images_tx, consumed_images_rx) =
    tokio::sync::mpsc::channel::<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>(128);

  // Local /dev/videoN ingestor: Streams images from the local video device
  // into a shared channel for classification.
  // let video_ingestor_chan_tx = consumed_images_tx.clone();
  let (video0_ingestor_task, video0_image_adjustment_task) =
    start_local_video_ingestor(&config, consumed_images_tx.clone()).await;

  // Camera API client: Consumes an image stream from the given endpoint, applies image adjustments
  // and writes consumed images to a channel for which to be used for classification.
  let (camera_api_producer_task, camera_api_consumer_task) = start_camera_api_client(
    &client_api,
    consumed_images_tx.clone(),
    consumed_images_rx,
    classify_request_tx,
  )
  .await;

  // Image classification: Consumes images from API client, sending them to grpc classifier server,
  // and consumes responses while taking action on the classification.
  let classifier_client_task = start_classifier_client(
    &client_api,
    &config,
    classify_response_tx,
    classify_response_rx,
    classify_request_rx,
  )
  .await
  .unwrap();

  let _ = tokio::join!(
    camera_api_producer_task,
    camera_api_consumer_task,
    classifier_client_task,
    video0_ingestor_task,
    video0_image_adjustment_task
  );
  Ok(())
}
