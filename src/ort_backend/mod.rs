use std::{collections::HashMap, time::Instant};

use anyhow::{Error, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use log::{error, info};
use ndarray::{Array, Axis, CowArray, IxDyn};
use ort::{inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session, Value};
use regex::Regex;

pub struct YoloOrtModel {
  session: Session,
  labels: HashMap<String, String>,
}

impl YoloOrtModel {
  pub fn new(onnx_model_path: String) -> Result<Self, Error> {
    // Configure builder with performance configurations.
    let builder = Session::builder()?
      .with_parallel_execution(true)?
      .with_optimization_level(ort::GraphOptimizationLevel::Level2)?;

    // Check if ROCm is available, otherwise, use the CPU.
    let rocm_provider = ROCmExecutionProvider::default();
    info!("ROCm available? {:?}", rocm_provider.is_available()?);

    let cpu_provider = CPUExecutionProvider::default();
    info!("CPU Provider available? {}", cpu_provider.is_available()?);

    if rocm_provider.is_available()? {
      match rocm_provider.register(&builder) {
        Ok(_) => info!("Successfuly registered ROCm provider"),
        Err(err) => {
          return Err(Error::msg(format!(
            "Failed to register rocm provider -> {err:?}"
          )));
        }
      }
    } else {
      match cpu_provider.register(&builder) {
        Ok(_) => info!("Successfuly registered CPU provider"),
        Err(err) => {
          return Err(Error::msg(format!(
            "Failed to register cpu provider -> {err:?}"
          )));
        }
      }
    };

    let mut model = Self {
      // Commit the builder by loading the model.
      session: builder.with_model_from_file(onnx_model_path)?,
      labels: HashMap::new(),
    };
    model.extract_labels_from_metadata()?;

    Ok(model)
  }

  pub fn get_label(&self, key: &String) -> Option<&String> {
    self.labels.get(key)
  }

  pub fn version(&self) -> Result<String, Error> {
    let version: String = self
      .session
      .metadata()?
      .custom("version")?
      .unwrap_or("0.0.0".to_string());

    let author: String = self
      .session
      .metadata()?
      .custom("author")?
      .unwrap_or("unknown".to_string());

    Ok(format!("{author} YOLO-{version}"))
  }

  fn preprocess(&self, input: &DynamicImage) -> Result<Array<f32, IxDyn>, Error> {
    let expected_input_shape = self.session.inputs[0]
      .input_type
      .tensor_dimensions()
      .unwrap();
    info!("Expected input dimensions -> {:?}", expected_input_shape);

    if let [num_imgs_dim, pixel_dim, height_dim, width_dim] = expected_input_shape[..4] {
      // Resize the inputs to match expected by the model.
      let xs = input.resize_exact(width_dim as u32, height_dim as u32, FilterType::CatmullRom);
      info!("Input Image resized -> {:?}", xs.dimensions());

      let mut input = Array::zeros((
        num_imgs_dim as usize,
        pixel_dim as usize,
        height_dim as usize,
        width_dim as usize,
      ))
      .into_dyn();
      for pixel in xs.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
      }
      Ok(input)
    } else {
      Err(Error::msg(format!(
        "Expected model inputs to contain 4 dimensions but got -> {:?}",
        expected_input_shape
      )))
    }
  }

  fn run_inference(&self, xs: &Array<f32, IxDyn>) -> Result<(), Error> {
    let ys = self.session.run(inputs!["images" => xs.view()]?)?;

    // TODO: understand how tf this is layed out.

    // Extract results.
    let output_name = self.session.outputs[0].name.clone();
    let output = ys[output_name].extract_tensor::<f32>()?;
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
      let label = self.get_label(&class_id.to_string()).unwrap();
      info!("class_id={class_id} | label={label} | prob={prob}");
    }

    // TODO: make a result struct.
    Ok(())
  }

  fn postprocess(&self) {}

  pub fn run(&self, input: DynamicImage) -> Result<(), Error> {
    // Pre-process.
    let pre_process_dt = Instant::now();
    let xs = self.preprocess(&input)?;
    info!("Pre-process took {:?}", pre_process_dt.elapsed());

    // Inference.
    let inference_dt = Instant::now();
    self.run_inference(&xs)?;
    info!("Inference took {:?}", inference_dt.elapsed());

    // Post-process.
    // let post_process_dt = Instant::now();
    // self.postprocess();
    // info!("Post-process took {:?}", post_process_dt.elapsed());

    // Results.
    Ok(())
  }

  fn extract_labels_from_metadata(&mut self) -> Result<(), Error> {
    match self.session.metadata()?.custom("names")? {
      Some(labels_raw) => {
        for cap in Regex::new(r"(\d+):\s+\'(\w*\s*\w*)\'")?.captures_iter(&labels_raw) {
          let class_id = cap.get(1).unwrap().as_str();
          let label = cap.get(2).unwrap().as_str();

          self.labels.insert(class_id.to_string(), label.to_string());
        }
        Ok(())
      }
      None => Err(Error::msg("Failed to extract labels from model metadata")),
    }
  }
}
