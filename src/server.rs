pub mod ort_backend;
use anyhow::{Error, Result};
use env_logger::Env;
use image::{imageops::FilterType, GenericImageView};
use log::{error, info};
use ndarray::{Array, Axis};
use ort::{inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session};
use ort_backend::YoloOrtModel;
use std::{borrow::Borrow, process::exit, time::Instant};

fn run_server() -> Result<(), Error> {
  // TODO: move into arg or something.
  let onnx_model_path = "./models/yolov8/yolov8n.onnx";

  let yolo_model = YoloOrtModel::new(onnx_model_path.to_string(), 0.75)?;
  info!("Version -> {}", yolo_model.version()?);

  // Load in the image and resize it to match expected input shape.
  let img = image::open("video0.jpg")?;
  info!("Image loaded -> {:?}", img.dimensions());
  yolo_model.run(img)
}

fn main() {
  // Initialize global logger. Logger value can be set via the 'RUST_LOG' environment variable.
  env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

  match run_server() {
    Ok(_) => exit(0),
    Err(err) => {
      error!("Server failed to start: {err}");
      exit(1)
    }
  }
}
