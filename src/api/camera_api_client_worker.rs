use crate::api::classifier_grpc::classifier::ClassifyImageRequest;
use crate::api::interfaces::CameraStreamResponse;
use crate::api::{AdjustedCameraBuffer, CameraApi};

use log::{error, info, warn};
use tokio::sync::mpsc;

/// Starts the Camera API client within two tokio tasks (consumer, producer) for which
/// consumes images from an open stream, from the given api endpoint, writing those images
/// to a channel. Those images are then consumed to apply adjustments to, like pre-processing,
/// for which to then write to a Sender channel to be classified.
///
/// Args:
/// * camera_api: Camera API instance reference.
/// * camera_api_tx: Sender channel for writing consumed images to.
/// * camera_api_rx: Receiver channel for consuming images from to apply adjustments to.
/// * classify_request_tx: Sender channel for writing adjusted images being classified.
///
/// Returns:
/// * A Tokio task for the producer.
/// * A Tokio task for the consumer.
pub async fn start_camera_api_client(
  client_api: &CameraApi,
  camera_api_tx: mpsc::Sender<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
  mut camera_api_rx: mpsc::Receiver<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
  classify_request_tx: mpsc::Sender<ClassifyImageRequest>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
  // CameraAPI Producer.
  let mut producer_client_api = client_api.clone();
  let producer_task = tokio::task::spawn(async move {
    info!("CameraAPI Client: Started client stream");

    loop {
      let status = producer_client_api
        .run_camera_stream(camera_api_tx.clone())
        .await
        .map_err(|err| err.to_string());

      match status {
        Ok(_) => {
          warn!("CameraAPI Client: Exited, likely due to timeout, restart client");
        }
        Err(err) => {
          error!("CameraAPI Client: {err}");
          if err.contains("error reading a body from connection") {
            info!("CameraAPI Client: Restarting camera stream");
            continue;
          }
          break;
        }
      }
    }

    warn!("CameraAPI Client: Terminating producer");
  });

  // CameraAPI Consumer.
  let consumer_task = tokio::task::spawn(async move {
    info!("CameraAPI Client: Started image stream consumer");
    let mut buffer: Vec<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)> = vec![];

    while camera_api_rx.recv_many(&mut buffer, 20).await > 0 {
      // Interate over all images, applying adjustments.
      for (image, adjusted_image) in buffer.to_owned() {
        if let Some(a_img_buffs) = adjusted_image {
          for a_img in a_img_buffs {
            info!(
              "CameraAPI Client: Consumed adjusted camera buffer {}[{}]",
              a_img.cam_name, a_img.cam_ip,
            );

            let send_status = classify_request_tx.send(ClassifyImageRequest {
              image: a_img.image_buff,
              device: a_img.device_name,
            });

            if let Err(err) = send_status.await {
              error!("CameraAPI Client: Failed to write adjusted image to classification channel -> {err}");
              break;
            }
          }
        } else {
          for (cam_ip, cam_elt) in image.cameras {
            warn!(
              "CameraAPI Client: Consumed un-adjusted camera buffer {}[{}]",
              cam_elt.name, cam_ip
            );
          }
        }
      }

      buffer.clear();
    }

    info!("CameraAPI Client: Terminating consumer");
  });

  return (producer_task, consumer_task);
}