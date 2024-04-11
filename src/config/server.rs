use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
  // Confidence threshold for which to filter detected objects.
  pub confidence: f64,

  // Path to the ONNX YOLO model.
  pub model_filepath: String,

  // Path to storage directory for which to store detected images in.
  pub image_store_dir: String,

  // Array of accepted classified class ids for which to store on match.
  pub accepted_match_classes: Vec<u64>,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      model_filepath: "".to_string(),
      image_store_dir: "".to_string(),
      confidence: 0.75,
      accepted_match_classes: vec![0],
    }
  }
}

impl ServerConfig {
  /// Parses and populates configuration from a given json file.
  ///
  /// Args:
  /// * filepath: File path to a valid json file.
  pub fn parse(filepath: String) -> Result<Self> {
    // Deserialize configuration from given json filepath.
    let body_vec = std::fs::read(filepath)?;
    let deseralized_config: ServerConfig = serde_json::from_slice(&body_vec)?;
    Ok(deseralized_config)
  }
}
