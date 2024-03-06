/// Implementation for hosting a tcp server to stream a video device.
use crate::data_struct::ring_buffer::RingBuffer;
use actix_web::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
use actix_web::{http::header::ContentType, web, App, HttpResponse, HttpServer, Responder};
use async_stream::stream;
use log::info;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct CameraServer {
  pub cam_buffer: Arc<RwLock<RingBuffer<Vec<u8>>>>,
  port: u16,
}

impl CameraServer {
  pub fn new(rb_capacity: usize, port: u16) -> Self {
    Self {
      cam_buffer: Arc::new(RwLock::new(RingBuffer::new(rb_capacity))),
      port,
    }
  }

  /// Starts camera server, which passes in the RingBuffer used
  /// to for responses.
  pub async fn start(&self) {
    info!("Listening on :{}", self.port);
    let data = web::Data::new(self.cam_buffer.clone());

    let _ = HttpServer::new(move || {
      App::new()
        .app_data(web::Data::clone(&data))
        .route("/", web::get().to(serve))
    })
    .bind(("127.0.0.1", self.port))
    .unwrap()
    .run()
    .await;
  }
}

async fn serve(data: web::Data<RwLock<RingBuffer<Vec<u8>>>>) -> impl Responder {
  let stream = stream! {
    let mut data = data.write().await;
    loop {
      match data.pop().await {
        Ok(img) => yield Ok::<_, Infallible>(web::Bytes::from(img)),
        Err(_) => todo!()
      }
    }
  };

  HttpResponse::Ok()
    .content_type(ContentType::octet_stream())
    .insert_header((ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
    .streaming(stream)
}
