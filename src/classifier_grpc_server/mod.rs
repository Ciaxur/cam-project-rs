pub mod classifier {
  tonic::include_proto!("classifier");
}

use std::pin::Pin;

use classifier::image_classifier_server::ImageClassifier;
use classifier::{ClassifyImageRequest, ClassifyImageResponse};
use log::info;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

#[derive(Debug, Default)]
pub struct ClassifierServer {}
type ResponseStream = Pin<Box<dyn Stream<Item = Result<ClassifyImageResponse, Status>> + Send>>;

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
    let output_stream = async_stream::try_stream! {
      while let Some(req) = req_stream.next().await {
        let req = req?;
        info!("Yoinked request -> {:?}", req);
        yield ClassifyImageResponse{
          device: format!("server response to -> {}", req.device),
          ..Default::default()
        };
      }
    };
    Ok(Response::new(
      Box::pin(output_stream) as Self::ClassifyImageStream
    ))
  }
}
