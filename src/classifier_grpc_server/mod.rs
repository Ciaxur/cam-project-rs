pub mod classifier {
  tonic::include_proto!("classifier");
}

use crate::ort_backend::YoloOrtModel;

use anyhow::{Error, Result};
use classifier::image_classifier_server::ImageClassifier;
use classifier::{ClassifyImageRequest, ClassifyImageResponse};
use log::info;
use std::pin::Pin;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

type ResponseStream = Pin<Box<dyn Stream<Item = Result<ClassifyImageResponse, Status>> + Send>>;

#[derive(Debug)]
pub struct ClassifierServer {
  model: YoloOrtModel,
}

impl ClassifierServer {
  pub fn new(onnx_model_path: String, prob_threshold: f64) -> Result<Self, Error> {
    Ok(Self {
      model: YoloOrtModel::new(onnx_model_path, prob_threshold)?,
    })
  }
}

#[tonic::async_trait]
impl ImageClassifier for ClassifierServer {
  type ClassifyImageStream = ResponseStream;

  async fn classify_image(
    &self,
    request: Request<Streaming<ClassifyImageRequest>>,
  ) -> Result<Response<Self::ClassifyImageStream>, Status> {
    // https://github.com/hyperium/tonic/blob/master/examples/routeguide-tutorial.md#bidirectional-streaming-rpc
    // - https://github.com/hyperium/tonic/blob/master/examples/src/streaming/server.rs
    let mut req_stream = request.into_inner();

    // TODO: how tf am i supposed to do this?
    // - https://stackoverflow.com/questions/72403669/tokio-tonic-how-to-fix-this-error-self-has-lifetime-life0-but-it-needs
    let output_stream = async_stream::try_stream! {
      while let Some(req) = req_stream.next().await {
        let req = req?;
        info!("Yoinked request -> {} | img = {}B", req.device, req.image.len());
        // self.model.run(Vec::new()).expect("shit");

        // match self.model.run(req.image.clone()) {
        //   Ok(_) => info!("ok"),
        //   Err(_) => info!("err")
        // }

        yield ClassifyImageResponse{
          device: format!("server response to -> {}", req.device),
          // image: req.image.clone(),
          ..Default::default()
        };
      }
    };
    Ok(Response::new(
      Box::pin(output_stream) as Self::ClassifyImageStream
    ))
  }
}
