pub mod classifier_grpc_server;
pub mod ort_backend;
pub mod pb;

use anyhow::{Error, Result};
use server::config::server::ServerConfig;
use clap::Parser;
use classifier_grpc_server::ClassifierServer;
use env_logger::Env;
use log::{error, info};
use pb::classifier::image_classifier_server::ImageClassifierServer;
use std::process::exit;
use tokio::select;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
  /// Server port to listen on.
  #[arg(short, long, required = true)]
  port: u16,

  /// File path to configuration json file.
  /// See config/mod.rs for more info.
  #[arg(short, long, required = true)]
  config: String,
}

/// Helper function which starts the gRPC server.
async fn run_server() -> Result<(), Error> {
  // Grab passed in cli args, forwarding them to the main execution logic.
  let args = Args::parse();

  // Load server config.
  info!("Using configuration file from -> {}", args.config);
  let config = ServerConfig::parse(args.config.clone())?;

  // Server config.
  let addr = format!("0.0.0.0:{}", args.port).parse().unwrap();
  let svc = ClassifierServer::new(
    config.model_filepath,
    config.confidence,
    config.image_store_dir,
    config.accepted_match_classes,
  )?;

  info!("Server listening on {}", addr);
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
