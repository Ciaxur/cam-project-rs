pub mod ort_backend;
use anyhow::{Error, Result};
use env_logger::Env;
use image::{imageops::FilterType, GenericImageView};
use log::{error, info};
use ndarray::{Array, Axis};
use opencv::core::{MatTrait, MatTraitConst, Vector};
use ort::{inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session};
use ort_backend::YoloOrtModel;
use std::{borrow::Borrow, process::exit, time::Instant};

fn run_server() -> Result<(), Error> {
  // TODO: move into arg or something.
  let onnx_model_path = "./models/yolov8/yolov8n.onnx";

  let yolo_model = YoloOrtModel::new(onnx_model_path.to_string(), 0.75)?;
  info!("Version -> {}", yolo_model.version()?);

  // Load in the image and resize it to match expected input shape.
  // NOTE: this matches what the client sends.
  let img = opencv::imgcodecs::imread("video0.jpg", opencv::imgcodecs::IMREAD_COLOR)?;
  let mut _img_mat = img.clone();

  // TODO: ensure color is preserved on response but use grayscale for inference.
  opencv::imgproc::cvt_color(&img, &mut _img_mat, opencv::imgproc::COLOR_BGR2GRAY, 0)?;

  let mut image_vec: Vector<u8> = Vector::new();
  let encode_flags: Vector<i32> = Vector::new();
  opencv::imgcodecs::imencode(".jpg", &_img_mat, &mut image_vec, &encode_flags)?;

  let client_req_img: Vec<u8> = image_vec.to_vec();

  // client_req_img.
  info!("Image loaded -> ({}, {})", img.cols(), img.rows());
  let _ = yolo_model.run(client_req_img)?;
  Ok(())
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
