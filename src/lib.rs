pub mod api;
pub mod camera;
pub mod classifier_grpc_client;
pub mod config;
pub mod pb;
pub mod utils;

use anyhow::Result;
use api::camera_api_client_worker::start_camera_api_client;
use api::classifier_worker::start_classifier_client;
use api::interfaces::CameraStreamResponse;
use api::{AdjustedCameraBuffer, CameraApi, CameraApiOptions};
use camera::worker::start_local_video_worker;
use clap::Parser;
use log::{debug, info};
use pb::classifier::{ClassifyImageRequest, ClassifyImageResponse};
use tokio::sync::mpsc;

use crate::config::Config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
  /// File path to configuration json file.
  /// See config/mod.rs for more info.
  #[arg(short, long, required = true)]
  config: String,
}

pub async fn run() -> Result<(), String> {
  // Grab passed in cli args, forwarding them to the main execution logic.
  let args = Args::parse();
  info!("Using configuration file from -> {}", args.config);
  let config = Config::parse(args.config.clone())
    .map_err(|err| format!("Configuration parse of '{}' failed: {}", args.config, err))?;
  debug!("Configuration loaded -> {config:?}");

  // Keep track of running concurrent tokio tasks.
  let mut tokio_tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();

  // Construct a shared camera client api.
  let camera_api_config = config
    .camera_api
    .as_ref()
    .expect("camera api configuration is required");

  let client_api = CameraApi::new(
    CameraApiOptions {
      endpoint: camera_api_config.endpoint.clone(),
      target_fps: camera_api_config.target_fps,
      credentials: api::ClientCredentials {
        trusted_ca_filepath: camera_api_config.trusted_ca_filepath.clone(),
        identity_cert_filepath: camera_api_config.identity_cert_filepath.clone(),
        identity_key_filepath: camera_api_config.identity_key_filepath.clone(),
      },
    },
    &camera_api_config,
  )
  .expect("Client instantiation failed");

  // Create two channels for which to allow other workers to issue requests
  // to and receive responses from.
  let (classify_request_tx, classify_request_rx) =
    mpsc::channel::<ClassifyImageRequest>(config.channel_buffer_size);
  let (classify_response_tx, classify_response_rx) =
    mpsc::channel::<ClassifyImageResponse>(config.channel_buffer_size);

  // Multi-producer single-consumer channel.
  // These channels are used to produce and consume images from the client api and
  // the local video device ingestor.
  let (consumed_images_tx, consumed_images_rx) = tokio::sync::mpsc::channel::<(
    CameraStreamResponse,
    Option<Vec<AdjustedCameraBuffer>>,
  )>(config.channel_buffer_size);

  // Local /dev/videoN ingestor: Streams images from the local video device
  // into a shared channel for classification.
  if let Some(local_video_config) = &config.local_video_config {
    let (ingestor_task, image_adjustment_task) =
      start_local_video_worker(&local_video_config, consumed_images_tx.clone()).await;
    tokio_tasks.push(ingestor_task);
    tokio_tasks.push(image_adjustment_task);
  }

  // Camera API client: Consumes an image stream from the given endpoint, applies image adjustments
  // and writes consumed images to a channel for which to be used for classification.
  {
    let (camera_api_producer_task, camera_api_consumer_task) = start_camera_api_client(
      &client_api,
      consumed_images_tx.clone(),
      consumed_images_rx,
      classify_request_tx,
    )
    .await;
    tokio_tasks.push(camera_api_producer_task);
    tokio_tasks.push(camera_api_consumer_task);
  }

  // Image classification: Consumes images from API client, sending them to grpc classifier server,
  // and consumes responses while taking action on the classification.
  {
    let classifier_client_task = start_classifier_client(
      &client_api,
      &config,
      &camera_api_config,
      classify_response_tx,
      classify_response_rx,
      classify_request_rx,
    )
    .await
    .unwrap();
    tokio_tasks.push(classifier_client_task);
  }

  // Wait for all tasks to conclude.
  for task in tokio_tasks {
    let _ = tokio::try_join!(task);
  }
  Ok(())
}
