use crate::camera::CameraVideoProps;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigCameraAPI {
  /// Endpoint, without protocol, to the backend API server for which
  /// is used to query image streams, camera configs, and invoke telegram
  /// notification.
  pub endpoint: String,

  /// Telegram API path used to push a POST request to when the image classification
  /// yields a match.
  /// ie: "telegram/message"
  pub post_telegram_endpoint_path: String,

  /// Camera list API path used to issue a GET request which contain a list of available
  /// cameras.
  /// ie: "camera/list"
  pub get_camera_list_endpoint_path: String,

  /// Camera snap API path used to issue a GET request which contain a snapshot of all available
  /// cameras buffers.
  /// ie: "camera/snap"
  pub get_camera_snap_endpoint_path: String,

  /// Camera stream API path used to issue a GET request which opens a stream that dispatches
  /// camera buffers as they become available.
  /// ie: "camera/subscribe"
  pub get_camera_stream_endpoint_path: String,

  // Target frames per second to contrain the congestion of image streams.
  pub target_fps: f64,

  /// Telegram chat id sent to the API to invoke a notification of the
  /// classified image to.
  pub telegram_chat_id: u64,

  // File path to the trusted CA bundle.
  pub trusted_ca_filepath: String,

  // File path to the identity certificate used by the gRPC client.
  pub identity_cert_filepath: String,

  // File path to the identity key used by the gRPC client.
  pub identity_key_filepath: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigLocalVideo {
  /// Video descriptor filepath for which to consume video stream from.
  pub video_fd: String,

  /// Local video camera stream properties.
  pub video_properties: CameraVideoProps,

  /// Channel buffering size. This is used to set the shared channel size
  /// for consuming and adjusting video streams within the worker. Essentially
  /// the number of images to buffer.
  pub video_stream_channel_size: usize,
}

impl Default for ConfigLocalVideo {
  fn default() -> Self {
    Self {
      video_fd: String::default(),
      video_properties: CameraVideoProps::default(),
      video_stream_channel_size: 128,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigImageStorage {
  /// Image storage filepath which is used by a StorageManager instance to save
  /// image classifier matches to.
  pub image_storage_path: String,

  /// Channel buffering size. This is used to set the size of the images to buffer
  /// within the storage channel.
  pub channel_buffer_size: usize,
}

impl Default for ConfigImageStorage {
  fn default() -> Self {
    Self {
      image_storage_path: String::default(),
      channel_buffer_size: 2048,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClientConfig {
  /// Cooldown period in seconds between notifying the client of an
  /// image classifier match.
  pub cooldown_s: u64,

  /// Channel buffering size. This is used to set the size of the channels
  /// shared across concurrent threads. Essentially the number of buffered images.
  pub channel_buffer_size: usize,

  /// Optional local video (/dev/videoN) configuration.
  pub local_video_config: Option<ConfigLocalVideo>,

  /// NOTE: Tightly coupled with https://github.com/Ciaxur/4bit.api.
  // Required configuration for the camera api.
  pub camera_api: Option<ConfigCameraAPI>,

  /// gRPC classifier server endpoint to stream images to be classified.
  /// This should include the protocol.
  pub grpc_server_endpoint: String,

  /// Optional local image storage configuration.
  pub image_storage: Option<ConfigImageStorage>,

  // Array of accepted classified labels for which to invoke a notification on.
  pub accepted_match_labels: Vec<u64>,
}

impl Default for ClientConfig {
  fn default() -> Self {
    Self {
      cooldown_s: 20,
      channel_buffer_size: 128,
      local_video_config: Option::default(),
      camera_api: Option::default(),
      accepted_match_labels: vec![0],
      grpc_server_endpoint: String::default(),
      image_storage: Option::default(),
    }
  }
}

impl ClientConfig {
  /// Parses and populates configuration from a given json file.
  ///
  /// Args:
  /// * filepath: File path to a valid json file.
  pub fn parse(filepath: String) -> Result<Self> {
    // Deserialize configuration from given json filepath.
    let body_vec = std::fs::read(filepath)?;
    let deseralized_config: ClientConfig = serde_json::from_slice(&body_vec)?;
    Ok(deseralized_config)
  }
}
