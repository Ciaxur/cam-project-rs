pub mod classifier_grpc_server;
pub mod ort_backend;

use std::{process::exit, time::Duration};

use anyhow::{Error, Result};
use classifier_grpc_server::classifier::image_classifier_server::ImageClassifierServer;
use classifier_grpc_server::ClassifierServer;
use env_logger::Env;
use log::{error, info};
use opencv::core::{MatTraitConst, Vector};
use ort_backend::YoloOrtModel;
use tokio::{select, time::sleep};
use tokio_stream::{Stream, StreamExt};
use tonic::transport::Server;

// Client
use classifier_grpc_server::classifier::{
  image_classifier_client::ImageClassifierClient, ClassifyImageRequest,
};
// TODO: args.

fn client_req_iter() -> impl Stream<Item = ClassifyImageRequest> {
  tokio_stream::iter(1..usize::MAX).map(|i| ClassifyImageRequest {
    device: format!("test[{i}] yo"),
    ..Default::default()
  })
}

async fn run_test_client() -> Result<(), Error> {
  // Arbitrarily wait for server to start.
  sleep(Duration::from_secs(1)).await;
  info!("Starting client");

  // Alright let's start the fun!
  let mut client = ImageClassifierClient::connect("http://[::1]:9000").await?;
  let stream_in = client_req_iter()
    .take(10)
    .throttle(Duration::from_millis(250));
  let response = client.classify_image(stream_in).await?;

  let mut response_stream = response.into_inner();
  while let Some(resp) = response_stream.next().await {
    let resp = resp?;
    info!("Client received -> {:?}", resp);
  }
  Ok(())
}

async fn run_server() -> Result<(), Error> {
  let addr = "[::1]:9000".parse().unwrap();
  let svc = ClassifierServer::default();

  info!("Server listeining on {}", addr);
  Server::builder()
    .add_service(ImageClassifierServer::new(svc))
    .serve(addr)
    .await?;
  Ok(())
}

async fn run_model() -> Result<(), Error> {
  // TODO: move into arg or something.
  let onnx_model_path = "./models/yolov8/yolov8n.onnx";

  let yolo_model = YoloOrtModel::new(onnx_model_path.to_string(), 0.75)?;
  info!("Version -> {}", yolo_model.version()?);

  // Load in the image and resize it to match expected input shape.
  // NOTE: this matches what the client sends.
  let img = opencv::imgcodecs::imread("video0.jpg", opencv::imgcodecs::IMREAD_COLOR)?;
  let mut _img_mat = img.clone();

  let mut image_vec: Vector<u8> = Vector::new();
  let encode_flags: Vector<i32> = Vector::new();
  opencv::imgcodecs::imencode(".jpg", &_img_mat, &mut image_vec, &encode_flags)?;

  let client_req_img: Vec<u8> = image_vec.to_vec();

  // client_req_img.
  info!("Image loaded -> ({}, {})", img.cols(), img.rows());
  let out = yolo_model.run(client_req_img)?;

  // DEBUG:
  opencv::highgui::named_window("yeet", opencv::highgui::WINDOW_AUTOSIZE)?;
  opencv::highgui::imshow("yeet", &out.img_mat)?;
  opencv::highgui::wait_key(0)?;

  Ok(())
}

#[tokio::main]
async fn main() {
  // Initialize global logger. Logger value can be set via the 'RUST_LOG' environment variable.
  env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

  select! {
    ret = run_server() => {
      match ret {
        Ok(_) => exit(0),
        Err(err) => {
          error!("Server failed to start: {err}");
          exit(1)
        }
      }
    },
    ret = run_test_client() => {
      match ret {
        Ok(_) => exit(0),
        Err(err) => {
          error!("Client failed to start: {err}");
          exit(1)
        }
      }
    },
  }
}
