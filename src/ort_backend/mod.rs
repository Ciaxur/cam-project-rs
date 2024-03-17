use std::collections::HashMap;

use anyhow::{Error, Result};
use log::{error, info};
use ort::{inputs, CPUExecutionProvider, ExecutionProvider, ROCmExecutionProvider, Session};
use regex::Regex;

pub struct YoloOrtModel {
  pub session: Session,
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

    // if rocm_provider.is_available()? {
    if false {
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
