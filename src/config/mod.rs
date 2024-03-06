use crate::camera::CameraVideoProps;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigCameraAPI {
  /// Endpoint, without protocol, to the backend API server for which
  /// is used to query image streams, camera configs, and invoke telegram
  /// notification.
  pub endpoint: String,

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
pub struct CameraServerConfig {
  pub port: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
  /// Cooldown period in seconds between notifying the client of an
  /// image classifier match.
  pub cooldown_s: u64,

  /// Video descriptor filepath for which to consume video stream from.
  pub video_fd: String,

  /// Local video server.
  pub camera_server: CameraServerConfig,

  /// Video camera stream properties.
  pub camera_properties: CameraVideoProps,

  /// NOTE: Tightly coupled with https://github.com/Ciaxur/4bit.api.
  // Configuration for the camera api.
  pub camera_api: ConfigCameraAPI,

  /// gRPC classifier server endpoint to stream images to be classified.
  /// This should include the protocol.
  pub grpc_server_endpoint: String,

  /// Image storage filepath which is used by a StorageManager instance to save
  /// image classifier matches to.
  pub image_storage_path: String,

  // Array of accepted classified labels for which to invoke a notification on.
  pub accepted_match_labels: Vec<u64>,
}

impl Config {
  /// Parses and populates configuration from a given json file.
  ///
  /// Args:
  /// * filepath: File path to a valid json file.
  pub fn parse(filepath: String) -> Result<Self> {
    // Deserialize configuration from given json filepath.
    let body_vec = std::fs::read(filepath)?;
    let deseralized_config: Config = serde_json::from_slice(&body_vec)?;
    Ok(deseralized_config)
  }
}
