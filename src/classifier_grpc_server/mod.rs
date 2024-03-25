pub mod classifier {
  tonic::include_proto!("classifier");
}

use crate::ort_backend::YoloOrtModel;

use anyhow::{Error, Result};
use classifier::image_classifier_server::ImageClassifier;
use classifier::{ClassifyImageRequest, ClassifyImageResponse};
use log::{debug, info};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

type ResponseStream = Pin<Box<dyn Stream<Item = Result<ClassifyImageResponse, Status>> + Send>>;

#[derive(Debug)]
pub struct ClassifierServer {
  model: Arc<YoloOrtModel>,
}

impl ClassifierServer {
  pub fn new(onnx_model_path: String, prob_threshold: f64) -> Result<Self, Error> {
    Ok(Self {
      model: Arc::new(YoloOrtModel::new(onnx_model_path, prob_threshold)?),
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

    // Lifetime of self could supercede the running stream, so we would create a clone of the model
    // which will guarantee dropping the model at the termination of the stream and depleated reference counter.
    //
    // See the following for more info:
    // - https://stackoverflow.com/questions/72403669/tokio-tonic-how-to-fix-this-error-self-has-lifetime-life0-but-it-needs
    let model = self.model.clone();
    let output_stream = async_stream::try_stream! {
      while let Some(req) = req_stream.next().await {
        let req = req?;

        debug!("Yoinked request -> {} | img = {}B", req.device, req.image.len());

        // Run inference on the input image.
        yield match model.run(req.image) {
          Ok(output) => {
            // By default, no detections, returns an empty image to save on bandwidth.
            let mut resp = ClassifyImageResponse{
              device: req.device,
              ..Default::default()
            };

            if output.detections.len() > 0 {
              for detection in output.detections {
                for prediction in detection.predictions {
                  resp.matches.push(prediction.class_id as f32);
                  resp.match_scores.push(prediction.confidence as f32);
                  resp.labels.push(prediction.label);
                }
              }

              // Include the mutated image with the higher resolution.
              if output.resized_img_vec.len() > output.img_vec.len() {
                resp.image = output.resized_img_vec.to_vec();
              } else {
                resp.image = output.img_vec.to_vec();
              }

              info!("Detection: labels={:?} | ids={:?} | confidence={:?}", resp.labels, resp.matches, resp.match_scores);
            }
            resp
          }
          Err(err) => Err(
            Status::unknown(
              format!("Inference failed on device={} -> {:?}", req.device, err)
            )
          )?,
        }
      }
    };

    Ok(Response::new(
      Box::pin(output_stream) as Self::ClassifyImageStream
    ))
  }
}
