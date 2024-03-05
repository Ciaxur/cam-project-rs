/// Implementation for hosting a tcp server to stream a video device.
use actix_web::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
use actix_web::{get, web, App, Handler, HttpResponse, HttpServer, Responder};
use async_stream::stream;
use log::{debug, error, info};
use std::convert::Infallible;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

// Multi-part constants.
static MULTIPART_BOUNDARY: &str = "01multipart_boundry_seperation10";
static CORS_ALLOW_ORIGIN: &str = "*";

pub struct CameraServer {
  camera_chan_rx: mpsc::Receiver<Vec<u8>>,
}

impl CameraServer {
  pub fn new(rx: mpsc::Receiver<Vec<u8>>) -> Self {
    Self { camera_chan_rx: rx }
  }

  pub async fn start(self, port: u16) {
    info!("Listening on :{port}");
    let _ = HttpServer::new(|| {
      App::new()
        .app_data(web::Data::new(&self.camera_chan_rx)) // WIP: figure this out.
        .route("/", web::get().to(serve))
    })
    .bind(("127.0.0.1", port))
    .unwrap()
    .run()
    .await;
  }
}

async fn serve(chan: web::Data<mpsc::Receiver<Vec<u8>>>) -> impl Responder {
  let mixed_multipart_mime = format!(
    "{}/x-mixed-replace;boundary={}",
    mime::MULTIPART,
    MULTIPART_BOUNDARY
  );

  let stream = stream! {
    yield Ok::<_, Infallible>(web::Bytes::from("data"));
  };

  HttpResponse::Ok()
    .content_type(mixed_multipart_mime)
    .insert_header((ACCESS_CONTROL_ALLOW_ORIGIN, CORS_ALLOW_ORIGIN))
    .streaming(stream)
}

/*
 async fn generate_data() -> Vec<u8> {
   // Do "work"...
   tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
   let data: &str = "yee data";

   let mut buffer: Vec<u8> = Vec::new();
   let now = SystemTime::now();
   let epoch_timestamp = now.duration_since(UNIX_EPOCH).unwrap();

   // Write multi-part chunk.
   buffer
     .write(format!("\r\n--{MULTIPART_BOUNDARY}\r\n").as_bytes())
     .unwrap();
   buffer
     .write(
       format!(
         "Content-Type: image/jpeg\r\nContent-Length: {}\r\nX-Timestamp: {}.{}\r\n\r\n",
         data.len(),
         epoch_timestamp.as_secs(),
         epoch_timestamp.subsec_micros(),
       )
       .as_bytes(),
     )
     .unwrap();
   buffer.write(data.as_bytes()).unwrap();

   buffer
 }



*/
