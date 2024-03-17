use apartment_cam_client::run;
use env_logger::Env;
use log::error;
use std::process::exit;

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
  // Initialize global logger. Logger value can be set via the 'RUST_LOG' environment variable.
  env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

  match run().await {
    Ok(_) => exit(0),
    Err(e) => {
      error!("Process failed: {:?}", e);
      exit(1);
    }
  }
}
