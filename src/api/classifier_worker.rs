use crate::api::classifier_grpc::{
  classifier::{ClassifyImageRequest, ClassifyImageResponse},
  ClassifierClient,
};
use crate::api::storage_manager::{StorageManager, StorageManagerFile};
use crate::api::CameraApi;

use crate::config::{Config, ConfigCameraAPI};
use anyhow::{Error, Result};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};

/// Starts the Classifier client within a tokio task, retruning the task instance.
///
/// The function expects the server's endpoint for which to connect to for classifying
/// images retrieved from the classify request channel and then writes the response to
/// the sender channel.
///
/// Args:
/// * camera_api: Camera API instance reference.
/// * config: Parsed configuration instance.
/// * camera_api_config: Parsed CameraAPI configuration instance.
/// * classify_response_tx: Sender channel used to write image classification responses to.
/// * classify_response_rx: Receiver channel used to consume response from.
/// * classify_request_rx: Receiver channel used to consume requests to be proxied and classified
///     by the sever.
///
/// Returns:
/// * Tokio task handler.
pub async fn start_classifier_client(
  client_api: &CameraApi,
  config: &Config,
  camera_api_config: &ConfigCameraAPI,
  classify_response_tx: mpsc::Sender<ClassifyImageResponse>,
  mut classify_response_rx: mpsc::Receiver<ClassifyImageResponse>,
  classify_request_rx: mpsc::Receiver<ClassifyImageRequest>,
) -> Result<tokio::task::JoinHandle<()>> {
  let telegram_chat_id = camera_api_config.telegram_chat_id;
  let grpc_server_endpoint = config.grpc_server_endpoint.clone();

  // Construct gRPC channel buffer size.
  let mut chan_buff_size: usize = 1024;
  if let Some(video_config) = &config.local_video_config {
    let video_props = &video_config.video_properties;
    chan_buff_size = (video_props.image_width * video_props.image_height * 20.0) as usize;
  }

  let mut classifier_client = ClassifierClient::new(grpc_server_endpoint, chan_buff_size);
  let classifier_client_api = client_api.clone();

  // Cooldown after classifying a hooman. Construct a map which tracks
  // each the last notification time.
  let cooldown_duration: Duration = Duration::from_secs(config.cooldown_s);
  let mut last_notify_map: HashMap<String, Instant> = HashMap::new();

  // Optionally create a StorageManager instance to save classifier matched images to.
  let mut storage_manager: Option<StorageManager> = None;
  let mut storage_manager_tx: Option<_> = None;
  if let Some(image_storage) = &config.image_storage {
    info!(
      "Created a Storage manager instance -> {}",
      image_storage.image_storage_path
    );
    let mut storage_manager_inst = StorageManager::new(
      image_storage.image_storage_path.clone(),
      image_storage.channel_buffer_size,
    );
    storage_manager_tx = Some(storage_manager_inst.start());
    storage_manager = Some(storage_manager_inst);
  }

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
          // Optionally save image to local disk.
          if let Some(storage_manager_tx) = &storage_manager_tx {
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
    if let Some(mut storage_manager) = storage_manager {
      storage_manager.stop();
    }
  });

  Ok(classifier_consumer_task)
}
