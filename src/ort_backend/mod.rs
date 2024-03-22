use std::{collections::HashMap, time::Instant};

use anyhow::{Error, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer};
use log::{debug, error, info};
use ndarray::{Array, Axis, CowArray, IxDyn};
use ort::{
  inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session, SessionOutputs,
  Tensor, Value,
};
use rand::{thread_rng, Rng};
use regex::Regex;

pub struct YoloOrtModel {
  session: Session,
  labels: HashMap<u32, String>,
  input_shape: Vec<i64>,
  color_palette: Vec<(i64, i64, i64)>,
  prob_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct Bbox {
  pub x: f32,
  pub y: f32,
  pub w: f32,
  pub h: f32,
}

#[derive(Debug, Clone)]
struct YoloPrediction {
  pub class_id: u32,
  pub label: String,
  pub confidence: f64,
  pub color: image::Rgb<i64>,
}

#[derive(Debug, Clone)]
struct YoloDetection {
  pub bbox: Bbox,
  pub predictions: Vec<YoloPrediction>,

  // Resized to match the original input.
  pub resized_bbox: Bbox,
}

#[derive(Debug, Clone)]
struct YoloOutput {
  pub output_img: DynamicImage,
  pub detections: Vec<YoloDetection>,
}

// TODO: docstrings
impl YoloOrtModel {
  pub fn new(onnx_model_path: String, prob_threshold: f64) -> Result<Self, Error> {
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

    let session = builder.with_model_from_file(onnx_model_path)?;
    let input_shape = session.inputs[0]
      .input_type
      .tensor_dimensions()
      .unwrap()
      .clone();

    let mut model = Self {
      // Commit the builder by loading the model.
      session,
      labels: HashMap::new(),
      input_shape,
      color_palette: Vec::new(),
      prob_threshold,
    };
    model.extract_labels_from_metadata()?;

    // Create a color palette.
    let mut rng = thread_rng();
    model.color_palette = model
      .labels
      .iter()
      .map(|_| {
        (
          rng.gen_range(0..=255),
          rng.gen_range(0..=255),
          rng.gen_range(0..=255),
        )
      })
      .collect();

    Ok(model)
  }

  pub fn get_label(&self, key: &u32) -> Option<&String> {
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

  // TODO: optimize
  fn preprocess(&self, input: &DynamicImage) -> Result<Array<f32, IxDyn>, Error> {
    let expected_input_shape = self.session.inputs[0]
      .input_type
      .tensor_dimensions()
      .unwrap();
    debug!("Expected input dimensions -> {:?}", expected_input_shape);

    if let [num_imgs_dim, pixel_dim, height_dim, width_dim] = expected_input_shape[..4] {
      // Resize the inputs to match expected by the model.
      let xs = input.resize_exact(width_dim as u32, height_dim as u32, FilterType::CatmullRom);
      debug!("Input Image resized -> {:?}", xs.dimensions());

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

  // TODO: optimize
  fn postprocess(&self, xs0: &DynamicImage, ys: SessionOutputs) -> Result<YoloOutput, Error> {
    // Extract results.
    let output_name = self.session.outputs[0].name.clone();
    let output = ys[output_name].extract_tensor::<f32>()?;
    let in_width = self.input_shape[3];
    let in_height = self.input_shape[2];

    // Original input resolution.
    let og_width = xs0.width();
    let og_height = xs0.height();
    let ratio: f32 = (in_width as f32 / og_width as f32).min(in_height as f32 / og_height as f32);

    // YOLOv8 Model output shape [1, 84, 8400] means the following:
    // - 1st Dimension -> Batch size
    // - 2nd Dimension ->
    //   - [0..4] -> bounding box coordinates (x, y, width and height) of the detected object.
    //   - [4..]  -> probability for each class.
    // - 3rd Dimension -> Max number of detected objects that the model can detect.
    //
    let total_classes = self.labels.len();
    let cxywh_offset: usize = 4;

    // Iterate over the batches, which in our case is capable of only one.
    let mut detections: Vec<YoloDetection> = Vec::new();
    let mut output_img = xs0.to_rgb8();

    for anchor in output.view().axis_iter(Axis(0)) {
      for prediction in anchor.axis_iter(Axis(1)) {
        // First 4 elements are the bounding box coordinates (x, y, width and height).
        let bbox = prediction.slice(ndarray::s![0..cxywh_offset]);
        let bbox = Bbox {
          x: bbox[0],
          y: bbox[1],
          w: bbox[2],
          h: bbox[3],
        };

        // Adjust bounding box to match the original input.
        let resized_bbox = Bbox {
          x: bbox.x / ratio,
          y: bbox.y / ratio,
          w: bbox.w / ratio,
          h: bbox.h / ratio,
        };

        let class_predictions: Vec<YoloPrediction> = prediction
          .slice(ndarray::s![cxywh_offset..(cxywh_offset + total_classes)])
          .into_iter()
          .enumerate()
          // Construct and filter predictions that meet confidence threshold.
          .filter_map(|(i, v)| {
            let id: u32 = i as u32;
            let prob: f64 = *v as f64;

            if prob >= self.prob_threshold {
              let color = image::Rgb(self.color_palette[i].into());

              // Add detections to input image.
              // WIP:

              Some(YoloPrediction {
                class_id: id as u32,
                label: self.labels.get(&id).unwrap().to_string(),
                confidence: prob,
                color,
              })
            } else {
              None
            }
          })
          .collect();

        // Return early if no detections satisfied confidence threshold.
        if class_predictions.is_empty() {
          continue;
        }

        let detection = YoloDetection {
          bbox,
          resized_bbox,

          // Remaining elements are probabilities for each class [4..84].
          predictions: class_predictions,
        };

        debug!("Detection -> {:?}", detection);
        detections.push(detection);
      }
    }

    Ok(YoloOutput {
      detections,
      output_img: DynamicImage::ImageRgb8(output_img),
    })
  }

  pub fn run(&self, input: DynamicImage) -> Result<(), Error> {
    // Pre-process.
    let pre_process_dt = Instant::now();
    let xs = self.preprocess(&input)?;
    info!("Pre-process took {:?}", pre_process_dt.elapsed());

    // Inference.
    let inference_dt = Instant::now();
    let ys = self.session.run(inputs!["images" => xs.view()]?)?;
    info!("Inference took {:?}", inference_dt.elapsed());

    // Post-process.
    let post_process_dt = Instant::now();
    let _ = self.postprocess(&input, ys)?;
    info!("Post-process took {:?}", post_process_dt.elapsed());

    // Results.
    Ok(())
  }

  fn extract_labels_from_metadata(&mut self) -> Result<(), Error> {
    match self.session.metadata()?.custom("names")? {
      Some(labels_raw) => {
        for cap in Regex::new(r"(\d+):\s+\'(\w*\s*\w*)\'")?.captures_iter(&labels_raw) {
          let class_id: u32 = cap.get(1).unwrap().as_str().parse::<u32>()?;
          let label = cap.get(2).unwrap().as_str();

          self.labels.insert(class_id, label.to_string());
        }
        Ok(())
      }
      None => Err(Error::msg("Failed to extract labels from model metadata")),
    }
  }
}
