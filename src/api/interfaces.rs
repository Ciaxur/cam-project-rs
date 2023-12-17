/// Structures for API request & responses with 4bit API.
/// This is used to serialize <-> deserialize payload bodies.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraListResponse {
  #[serde(rename = "Cameras")]
  pub cameras: Vec<CameraEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraSnapResponse {
  pub cameras: HashMap<String, CameraBaseData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraStreamResponse {
  pub cameras: HashMap<String, CameraBaseData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraBaseData {
  pub name: String,
  #[serde(rename = "data")]
  pub data_b64: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraEntry {
  #[serde(rename = "Id")]
  pub id: u64,
  #[serde(rename = "Name")]
  pub name: String,
  #[serde(rename = "CreatedAt")]
  pub created_at: String,
  #[serde(rename = "ModifiedAt")]
  pub modified_at: String,
  #[serde(rename = "IP")]
  pub ip: String,
  #[serde(rename = "Adjustment")]
  pub adjustment: CameraAdjustment,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraAdjustment {
  #[serde(rename = "Id")]
  pub id: u64,
  #[serde(rename = "Timestamp")]
  pub timestamp: String,
  #[serde(rename = "CropFrameHeight")]
  pub crop_frame_height: f64,
  #[serde(rename = "CropFrameWidth")]
  pub crop_frame_width: f64,
  #[serde(rename = "CropFrameX")]
  pub crop_frame_x: u64,
  #[serde(rename = "CropFrameY")]
  pub crop_frame_y: u64,
}

/// Telegram Push Message Notification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TelegramMessageRequest {
  #[serde(rename = "chatId")]
  pub chat_id: u64,
  // Message to include within the pushed message.
  pub message: String,
  // Base64-encoded image.
  #[serde(rename = "image")]
  pub image_b64: String,
}
