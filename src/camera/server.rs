/// Implementation for hosting a tcp server to stream a video device.
use crate::data_struct::ring_buffer::RingBuffer;
use actix_web::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
use actix_web::{http::header::ContentType, web, App, HttpResponse, HttpServer, Responder};
use async_stream::stream;
use log::info;
use std::convert::Infallible;
use std::sync::{Arc, RwLock};

pub struct CameraServer {
  cam_buffer: Arc<RingBuffer<Vec<u8>>>,
}

impl CameraServer {
  pub fn new() -> Self {
    Self {
      cam_buffer: Arc::new(RingBuffer::new(69)),
    }
  }

  /// Starts camera server, which passes in the RingBuffer used
  /// to for responses.
  ///
  /// Args:
  ///   - port: Port to listen on.
  pub async fn start(&self, port: u16) {
    info!("Listening on :{port}");
    let data = web::Data::new(RwLock::new(self.cam_buffer.clone()));

    let _ = HttpServer::new(move || {
      App::new()
        .app_data(web::Data::clone(&data))
        .route("/", web::get().to(serve))
    })
    .bind(("127.0.0.1", port))
    .unwrap()
    .run()
    .await;
  }
}

async fn serve(data: web::Data<RwLock<RingBuffer<Vec<u8>>>>) -> impl Responder {
  let stream = stream! {
    let mut data = data.write().unwrap();
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
