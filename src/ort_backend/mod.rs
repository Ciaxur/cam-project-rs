use std::{collections::HashMap, time::Instant};

use anyhow::{Error, Result};
use log::{debug, info};
use ndarray::{Array, Axis, IxDyn};
use opencv::core::{Mat, MatSize, MatTrait, MatTraitConst, Point, Rect, Scalar, Size, Vector};
use opencv::{imgcodecs, imgproc};
use ort::{
  inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session, SessionOutputs,
};
use rand::{thread_rng, Rng};
use regex::Regex;

pub struct YoloOrtModel {
  session: Session,
  labels: HashMap<u32, String>,
  input_shape: Vec<i64>,
  color_palette: Vec<(f64, f64, f64)>,
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
pub struct YoloPrediction {
  pub class_id: u32,
  pub label: String,
  pub confidence: f64,
  pub color: Scalar,
}

#[derive(Debug, Clone)]
pub struct YoloDetection {
  pub bbox: Bbox,
  pub predictions: Vec<YoloPrediction>,

  // Resized to match the original input.
  pub resized_bbox: Bbox,
}

#[derive(Debug, Clone)]
pub struct YoloOutput {
  pub output_img: Vector<u8>,

  // TODO: have both resized and original matricies of that have bboxes.
  pub output_img_mat: Mat,
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
          rng.gen_range(0.0..=255.0),
          rng.gen_range(0.0..=255.0),
          rng.gen_range(0.0..=255.0),
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
  // returns pre-processed image and encoded original image.
  fn preprocess(&self, input: &Vec<u8>) -> Result<(Array<f32, IxDyn>, Mat), Error> {
    let expected_input_shape = self.session.inputs[0]
      .input_type
      .tensor_dimensions()
      .unwrap();
    debug!("Expected input dimensions -> {:?}", expected_input_shape);

    if let [num_imgs_dim, pixel_dim, height_dim, width_dim] = expected_input_shape[..4] {
      // Resize the inputs to match expected by the model.
      let input_mat = imgcodecs::imdecode(
        &Vector::<u8>::from(input.to_vec()),
        imgcodecs::IMREAD_ANYCOLOR,
      )?;
      let mut xs_mat = Mat::default();
      imgproc::resize(
        &input_mat,
        &mut xs_mat,
        Size {
          height: height_dim as i32,
          width: width_dim as i32,
        },
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
      )?;
      debug!(
        "Input Image resized -> ({}, {})",
        xs_mat.cols(),
        xs_mat.rows()
      );

      // Convert matrix to a byte array.
      let mut xs_vec: Vector<u8> = Vector::new();
      let encode_flags: Vector<i32> = Vector::new();
      imgcodecs::imencode(".jpg", &xs_mat, &mut xs_vec, &encode_flags)?;

      // Load the image and then convert to a 3D Array.
      let img = image::load_from_memory(&xs_vec.to_vec())?.to_rgb8();
      let (width, height) = img.dimensions();

      let mut input = Array::zeros((
        num_imgs_dim as usize,
        pixel_dim as usize,
        height_dim as usize,
        width_dim as usize,
      ))
      .into_dyn();
      for y in 0..height {
        for x in 0..width {
          let pixel: &image::Rgb<u8> = img.get_pixel(x, y);
          let [r, g, b] = pixel.0;

          // Set all rgb values for each pixel.
          input[[0, 0, y as usize, x as usize]] = (r as f32) / 255.;
          input[[0, 1, y as usize, x as usize]] = (g as f32) / 255.;
          input[[0, 2, y as usize, x as usize]] = (b as f32) / 255.;
        }
      }

      Ok((input, input_mat))
    } else {
      Err(Error::msg(format!(
        "Expected model inputs to contain 4 dimensions but got -> {:?}",
        expected_input_shape
      )))
    }
  }

  // TODO: optimize
  // xs0 -> original encoded image.
  fn postprocess(&self, xs0: Mat, ys: SessionOutputs) -> Result<YoloOutput, Error> {
    // Extract results.
    let output_name = self.session.outputs[0].name.clone();
    let output = ys[output_name].extract_tensor::<f32>()?;
    let in_width = self.input_shape[3];
    let in_height = self.input_shape[2];

    // Original input resolution.
    let og_width = xs0.cols();
    let og_height = xs0.rows();
    let w_ratio: f32 = og_width as f32 / in_width as f32;
    let h_ratio: f32 = og_height as f32 / in_height as f32;

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
    let mut img_mat = xs0.clone();

    for anchor in output.view().axis_iter(Axis(0)) {
      for prediction in anchor.axis_iter(Axis(1)) {
        // First 4 elements are the bounding box coordinates (x, y, width and height).
        let bbox = prediction.slice(ndarray::s![0..cxywh_offset]);
        // TODO: do we need to center x/y here like the resized one?
        let bbox = Bbox {
          x: bbox[0],
          y: bbox[1],
          w: bbox[2],
          h: bbox[3],
        };

        // Adjust bounding box to match the original input.
        // We can do so by adjusting the ratio.
        // Then centering x/y since they point at the detected image center.
        // TODO: BUG: fix this. this is wrong!
        let rw = bbox.w * w_ratio;
        let rh = bbox.y * h_ratio;
        let resized_bbox = Bbox {
          x: (bbox.x * w_ratio) - (rw / 2.),
          y: (bbox.y * h_ratio) - (rh / 4.),
          w: rw,
          h: rh / 2.,
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
              let _color = self.color_palette[i];
              let color: Scalar = Scalar::new(_color.0, _color.1, _color.2, 0.);
              let label = self.labels.get(&id).unwrap().to_string();

              // Add detections to input image.
              imgproc::rectangle(
                &mut img_mat,
                Rect::new(
                  resized_bbox.x as i32,
                  resized_bbox.y as i32,
                  resized_bbox.w as i32,
                  resized_bbox.h as i32,
                ),
                color,
                1,
                imgproc::LINE_8,
                0,
              )
              .expect("draw detection bbox");

              // Add label to the image.
              let text_origin_x = resized_bbox.x;
              let text_origin_y = resized_bbox.y;

              // TEST:
              // TODO: use the OG color image. use white if image is grayscale otherwise.
              let white_color = Scalar::new(255., 255., 255., 255.);
              imgproc::put_text(
                &mut img_mat,
                label.as_str(),
                Point::new(text_origin_x as i32, text_origin_y as i32),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                white_color,
                1,
                imgproc::LINE_AA,
                false,
              )
              .expect("draw text label");

              Some(YoloPrediction {
                class_id: id as u32,
                label,
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

    let mut _img_buf = Vector::new();
    let imwrite_flags: Vector<i32> = Vector::new();
    opencv::imgcodecs::imencode(".jpeg", &img_mat, &mut _img_buf, &imwrite_flags).expect("shit");

    Ok(YoloOutput {
      detections,
      output_img: _img_buf,
      output_img_mat: img_mat,
    })
  }

  pub fn run(&self, input: Vec<u8>) -> Result<YoloOutput, Error> {
    // Pre-process.
    let pre_process_dt = Instant::now();
    let (xs, xs_mat) = self.preprocess(&input)?;
    info!("Pre-process took {:?}", pre_process_dt.elapsed());

    // Inference.
    let inference_dt = Instant::now();
    let ys = self.session.run(inputs!["images" => xs.view()]?)?;
    info!("Inference took {:?}", inference_dt.elapsed());

    // Post-process.
    let post_process_dt = Instant::now();
    let yolo_output = self.postprocess(xs_mat, ys)?;
    info!("Post-process took {:?}", post_process_dt.elapsed());

    // Results.
    Ok(yolo_output)
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
