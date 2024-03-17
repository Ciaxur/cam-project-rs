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

  let yolo_model = YoloOrtModel::new(onnx_model_path.to_string())?;
  let session = &yolo_model.session;

  // Grab expected input dimensions.
  let expected_input_shape = session.inputs[0].input_type.tensor_dimensions().unwrap();
  if let [num_imgs_dim, pixel_dim, height_dim, width_dim] = expected_input_shape[..4] {
    info!("Expected input dimensions -> {:?}", expected_input_shape);

    // Load in the image and resize it to match expected input shape.
    let og_img = image::open("keanu.jpg")?;
    info!("Image loaded -> {:?}", og_img.dimensions());

    let img = og_img.resize_exact(width_dim as u32, height_dim as u32, FilterType::CatmullRom);
    info!("Image resized -> {:?}", img.dimensions());

    // Let's do some inference.
    let mut input = Array::zeros((
      num_imgs_dim as usize,
      pixel_dim as usize,
      height_dim as usize,
      width_dim as usize,
    ));
    for pixel in img.pixels() {
      let x = pixel.0 as _;
      let y = pixel.1 as _;
      let [r, g, b, _] = pixel.2 .0;
      input[[0, 0, y, x]] = (r as f32) / 255.;
      input[[0, 1, y, x]] = (g as f32) / 255.;
      input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    let t = Instant::now();
    let outputs = session.run(inputs!["images" => input.view()]?)?;
    info!("Inference took {:?}", t.elapsed());

    // Extract results.
    let output_name = session.outputs[0].name.clone();
    let output = outputs[output_name].extract_tensor::<f32>()?;
    info!("Output -> {:?}", output);

    // Draw boxes around match.
    // let mut boxes = Vec::new();
    for row in output
      .view()
      .slice(ndarray::s![.., .., 0])
      .axis_iter(Axis(0))
    {
      let row: Vec<_> = row.iter().copied().collect();
      let (class_id, prob) = row
        .iter()
        // skip bounding box coordinates
        .skip(4)
        .enumerate()
        .map(|(index, value)| (index, *value))
        .reduce(|acc, row| if row.1 > acc.1 { row } else { acc })
        .unwrap();

      /**
       * WIP: continue here:
       * https://github.com/pykeio/ort/blob/main/examples/yolov8/examples/yolov8.rs#L83
       * https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-Rust/src/ort_backend.rs
       */
      let label = yolo_model.get_label(&class_id.to_string()).unwrap();
      info!("class_id={class_id} | label={label} | prob={prob}");
    }

    Ok(())
  } else {
    Err(Error::msg(format!(
      "Expected model inputs to contain 4 dimensions but got -> {:?}",
      expected_input_shape
    )))
  }
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
