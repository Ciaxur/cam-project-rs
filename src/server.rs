pub mod classifier_grpc_server;
pub mod ort_backend;
pub mod utils;

use anyhow::{Error, Result};
use clap::Parser;
use classifier_grpc_server::classifier::image_classifier_server::ImageClassifierServer;
use classifier_grpc_server::ClassifierServer;
use env_logger::Env;
use log::{error, info};
use std::process::exit;
use tokio::select;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
  /// Server port to listen on.
  #[arg(short, long, required = true)]
  port: u16,

  /// Prediction percentage threshold.
  #[arg(short, long, default_value_t = 0.75)]
  confidence: f64,

  /// Path to ONNX YOLO Model.
  #[arg(short, long, required = true)]
  model_filepath: String,

  /// Path to storage directory for which to store detected images in.
  #[arg(short, long, required = true)]
  image_store_dir: String,
}

/// Helper function which starts the gRPC server.
async fn run_server() -> Result<(), Error> {
  // Grab passed in cli args, forwarding them to the main execution logic.
  let args = Args::parse();

  // Server config.
  let addr = format!("[::1]:{}", args.port).parse().unwrap();
  let svc = ClassifierServer::new(args.model_filepath, args.confidence, args.image_store_dir)?;

  info!("Server listeining on {}", addr);
  Server::builder()
    .concurrency_limit_per_connection(32)
    .add_service(ImageClassifierServer::new(svc))
    .serve(addr)
    .await?;
  Ok(())
}

#[tokio::main(flavor = "multi_thread")]
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
  }
}
