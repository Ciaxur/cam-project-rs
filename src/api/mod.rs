pub mod camera_api_client_worker;
pub mod classifier_worker;
pub mod interfaces;
pub mod ndarray_util;
use anyhow::{Error, Result};
use base64::{engine::general_purpose as b64, Engine as _};
use interfaces::{CameraEntry, CameraListResponse, CameraSnapResponse, CameraStreamResponse};
use log::{debug, info, warn};
use ndarray_util::{
  image_bytes_to_ndarray, ndarray_crop_image, ndarray_rotate_image, ndarray_to_image_bytes,
};
use reqwest::{header, Client, Method};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc::Sender;
use tokio::time::{Duration, Instant};

use crate::config::ConfigCameraAPI;

/// Helper function for loading in certificates for mTLS.
///
/// # Arguments
/// * `creds` - The client credentials instance used for initializing TLS credentials.
///
/// # Returns
/// * A key-pair identity and a trusted certificate.
fn load_certificates(
  creds: ClientCredentials,
) -> Result<(reqwest::Identity, reqwest::Certificate), Error> {
  // Construct trusted CA bundle.
  let trusted_ca_buffer = std::fs::read(creds.trusted_ca_filepath)?;
  let ca = reqwest::Certificate::from_pem(&trusted_ca_buffer).unwrap();

  // Create a rustls identity pair.
  // rustls expects a single pem file with both public and private keys.
  let mut id_buff = Vec::<u8>::new();
  File::open(creds.identity_cert_filepath)?.read_to_end(&mut id_buff)?;
  File::open(creds.identity_key_filepath)?.read_to_end(&mut id_buff)?;

  let cert_key_pair = reqwest::Identity::from_pem(&id_buff)?;

  Ok((cert_key_pair, ca))
}

/// This contains client credential info used to configure the Camera Client
/// with mTLS.
#[derive(Clone, Debug)]
pub struct ClientCredentials {
  pub trusted_ca_filepath: String,
  pub identity_cert_filepath: String,
  pub identity_key_filepath: String,
}

/// This contains options used to configure initializing a CameraApi instance.
#[derive(Clone, Debug)]
pub struct CameraApiOptions {
  pub endpoint: String,
  pub credentials: ClientCredentials,
  pub target_fps: f64,
}

/// Structure containing the final adjusted camera image buffer along with
/// the metadata.
#[derive(Clone, Debug)]
pub struct AdjustedCameraBuffer {
  pub image_buff: Vec<u8>,
  pub cam_ip: String,
  pub cam_name: String,
  pub device_name: String,
}

#[derive(Clone, Debug)]
pub struct CameraApi {
  client: Client,
  endpoint: String,
  target_fps: f64,

  // Clone of the camera api configuration.
  api_config: ConfigCameraAPI,

  // Cached atmoic Camera List and Entries from server response.
  camera_list: Arc<RwLock<CameraListResponse>>,
  camera_entries: Arc<RwLock<HashMap<String, CameraEntry>>>,
  camera_entries_last_updated: Option<Instant>,
}

// Static duration for when cached data is considered stale.
static STALE_CACHE_DURATION: Duration = Duration::from_secs(60);

impl CameraApi {
  /// Constructor which creates an HTTP Client with mTLS and configures the client
  /// based on given options.
  ///
  /// # Args:
  /// * opts - Camera Options used to configure the HTTP client.
  ///
  /// # Returns:
  /// * CameraApi instance.
  pub fn new(opts: CameraApiOptions, api_config: &ConfigCameraAPI) -> Result<CameraApi, Error> {
    // Construct credentials.
    let (video_identity, trusted_cas) = load_certificates(opts.credentials)?;

    // Construct HTTP client.
    let mut headers = header::HeaderMap::new();
    headers.insert(
      header::CONTENT_TYPE,
      header::HeaderValue::from_static("application/json"),
    );

    // Construct HTTP Client.
    // BUG: native-tls does not load multiple CAs from a bundle. So use rulstls instead.
    // https://github.com/seanmonstar/reqwest/issues/1260
    let client = Client::builder()
      .use_rustls_tls()
      .tls_built_in_root_certs(false)
      .https_only(true)
      .default_headers(headers)
      .tcp_keepalive(Duration::from_secs(600))
      .http2_keep_alive_interval(Duration::from_secs(2))
      .http2_keep_alive_timeout(Duration::from_secs(600))
      .add_root_certificate(trusted_cas.clone())
      .identity(video_identity.clone())
      .build()
      .expect("Failed to create http client");

    Ok(Self {
      client,
      endpoint: opts.endpoint,
      target_fps: opts.target_fps,
      api_config: api_config.clone(),
      camera_entries: Arc::new(RwLock::new(HashMap::new())),
      camera_list: Arc::new(RwLock::new(CameraListResponse { cameras: vec![] })),
      camera_entries_last_updated: None,
    })
  }

  /// Generic request helper function, which invokes the given api endpoint
  /// and deserializes the response body based on the type T.
  ///
  /// Defaults to a 5s timeout, which is returned if the request is not successful.
  ///
  /// # Args:
  /// * api_endpoint - The endpoint uri on the api.
  /// * method - HTTP Method type.
  /// * payload - Serialized payload body.
  ///
  /// # Returns:
  /// * Deserialized interface struct of type T.
  async fn _invoke_api<'a, T>(
    &self,
    api_endpoint: &String,
    method: Method,
    payload: String,
  ) -> Result<T>
  where
    // https://serde.rs/lifetimes.html
    // DeserializeOwned is a trait, which indicates that this type does not have
    // borrowed references such as &str.
    T: serde::de::DeserializeOwned + std::fmt::Debug,
  {
    // Construct endpoint.
    let server_endpoint = self.endpoint.clone();
    let endpoint = format!("https://{server_endpoint}/{api_endpoint}");

    // Issue request and deserialize the response body.
    let res = self
      .client
      .request(method, endpoint)
      .timeout(Duration::from_secs(5))
      .body(payload)
      .send()
      .await?;
    debug!("Client request '{api_endpoint}' succeeded -> {res:?}");

    // Since we're using a generic with a set lifetime, we need serde_json to perform
    // a zero-copy deserialization on the body, which is done by having the generic type
    // implement DeserializeOwned.
    // .json() uses serde under the hood, which does the following:
    // let body = res.bytes().await?;
    // let body_vec: Vec<u8> = body.to_vec();
    // let deserialized_res: T = serde_json::from_slice(&body_vec)?;
    let deserialized_res: T = res.json().await?;
    debug!("GET response deserialized -> {deserialized_res:?}");

    Ok(deserialized_res)
  }

  /// Generic request helper function, which invokes the given api endpoint
  /// and returns unserialized body.
  ///
  /// Defaults to a 5s timeout, which is returned if the request is not successful.
  ///
  /// # Args:
  /// * api_endpoint - The endpoint uri on the api.
  /// * method - HTTP Method type.
  /// * payload - Serialized payload body.
  ///
  /// # Returns:
  /// * Deserialized response body.
  async fn _invoke_api_to_bytes(
    &self,
    api_endpoint: &String,
    method: Method,
    payload: String,
  ) -> Result<Vec<u8>> {
    // Construct endpoint.
    let server_endpoint = self.endpoint.clone();
    let endpoint = format!("https://{server_endpoint}/{api_endpoint}");

    // Issue request and deserialize the response body.
    let res = self
      .client
      .request(method, endpoint)
      .timeout(Duration::from_secs(5))
      .body(payload)
      .send()
      .await?;
    debug!("Client request '{api_endpoint}' succeeded -> {res:?}");

    Ok(res.bytes().await?.to_vec())
  }

  /// Pushes a message notification to API which gets relayed to the stored
  /// telegram chatId.
  ///
  /// Args:
  /// * telegram_chat_id: Telegram ChatID to push a message notification to.
  /// * message: Message to include in the chat.
  /// * image: Jpeg image byte array.
  pub async fn push_msg_notification(
    self,
    telegram_chat_id: u64,
    message: String,
    image: Vec<u8>,
  ) -> Result<()> {
    // Base64-encode the image.
    let image_b64 = b64::STANDARD.encode(image);

    // Construct request payload.
    let req = interfaces::TelegramMessageRequest {
      chat_id: telegram_chat_id,
      image_b64,
      message,
    };
    let payload = serde_json::to_string(&req).map_err(|err| {
      Error::msg(format!(
        "Push Notification: Internal failure to serialize request payload -> {err}"
      ))
    })?;

    // Construct api endpoint.
    let _ = self
      ._invoke_api_to_bytes(
        &self.api_config.post_telegram_endpoint_path,
        Method::POST,
        payload,
      )
      .await
      .map_err(|err| {
        Error::msg(format!(
          "Push Notification: Failed to invoke notification. From server -> {err}",
        ))
      })?;

    Ok(())
  }

  /// Invokes a GET request to the API endpoint, retrieving the list of cameras stored
  /// on the server-side, deserializes the payload body and returns it.
  ///
  /// # Returns:
  /// * Deserialized response.
  async fn list_cameras(&self) -> Result<CameraListResponse> {
    // Resuse internal generic function to invoke an API request.
    let cameras = self
      ._invoke_api(
        &self.api_config.get_camera_list_endpoint_path,
        Method::GET,
        "{}".to_string(),
      )
      .await?;
    Ok(cameras)
  }

  // / Grabs a snapshot of all the cameras connected to the API server.
  pub async fn snap_cameras(&self) -> Result<CameraSnapResponse> {
    // Resuse internal generic function to invoke an API request.
    let camera_snaps = self
      ._invoke_api(
        &self.api_config.get_camera_snap_endpoint_path,
        Method::GET,
        "{}".to_string(),
      )
      .await?;
    Ok(camera_snaps)
  }

  /// Invokes a GET request to the API endpoint, retrieving the list of cameras stored
  /// on the server-side, storing the response in an in-memory HashMap as a cache/reference.
  async fn update_camera_list_cache(&mut self) -> Result<()> {
    // Only update camera entries if stale.
    if self.camera_entries_last_updated.is_some()
      && self.camera_entries_last_updated.unwrap().elapsed() < STALE_CACHE_DURATION
    {
      return Ok(());
    }

    // Request a fresh list of cameras from the server.
    let new_camera_list = self.list_cameras().await?;
    self.camera_entries_last_updated = Some(Instant::now());

    // Update camera list.
    match self.camera_list.write() {
      Ok(mut camera_list) => {
        *camera_list = new_camera_list.clone();
      }
      Err(err) => {
        warn!("Failed to update camera list cache: {err}");
        return Err(Error::msg(err.to_string()));
      }
    }

    // Construct HashMap.
    match self.camera_entries.write() {
      Ok(mut camera_entries_mp) => {
        // Construct camera hashmap mapping camera ip to its corresponding camera entry.
        for camera_elt in new_camera_list.cameras {
          camera_entries_mp.insert(camera_elt.ip.clone(), camera_elt);
        }
        info!("CameraEntry HashMap cache updated");
        Ok(())
      }
      Err(err) => {
        warn!("Failed to construct CameraEntry HashMap cache: {err}");
        Err(Error::msg(err.to_string()))
      }
    }
  }

  /// Adjusts the given image response according to the stored camera configuration,
  /// generating an array of adjusted images.
  ///
  /// Args:
  /// * cam_response: The camera stream response instance for which to apply adjustments to
  ///
  /// Returns:
  /// * Array of adjusted camera image buffers
  async fn apply_image_adjustments(
    &mut self,
    cam_response: Arc<&CameraStreamResponse>,
  ) -> Result<Vec<AdjustedCameraBuffer>> {
    // Make sure to update camera cache.
    let _ = self.update_camera_list_cache().await.map_err(|err| {
      Error::msg(format!(
        "Camera Adjustment: Failed to update camera list cache: {err}"
      ))
    })?;

    // Find corresponding adjustment config.
    let cam_configs = self.camera_entries.read().map_err(|err| {
      Error::msg(format!(
        "Camera Adjustment: Internal error locking camera entries: {err}"
      ))
    })?;

    // Adjust each image.
    let mut adjusted_image_buffers: Vec<AdjustedCameraBuffer> = Vec::new();
    for (cam_ip, cam_elt) in cam_response.cameras.clone().into_iter() {
      if let Some(cam_config) = cam_configs.get(&cam_ip) {
        debug!("Camera Adjustments: Adjusting {}[{}]", cam_elt.name, cam_ip);

        // Decode the image, then parse and convert to ndarray for manual modification.
        let img_raw = b64::STANDARD.decode(cam_elt.data_b64)?;
        debug!("Image[{cam_ip}] base64 decoded -> {}B", img_raw.len());

        // Adjust the image according the stored adjustment configs.
        let (mut image_ndarray, image_result) = image_bytes_to_ndarray(&img_raw)?;
        let (width, height) = image_result.dimensions();
        debug!(
          "Image[{cam_ip}] parsed -> {}x{} | ndarray -> {:?}!",
          width,
          height,
          image_ndarray.shape()
        );

        // Crop the image.
        image_ndarray = ndarray_crop_image(image_ndarray, image_result, &cam_config.adjustment);

        // Apply image rotation.
        if cam_config.adjustment.image_rotate != 0.0 {
          image_ndarray = ndarray_rotate_image(image_ndarray, cam_config.adjustment.image_rotate);
        }

        // Buffer adjusted image.
        adjusted_image_buffers.push(AdjustedCameraBuffer {
          cam_ip: cam_ip.clone(),
          cam_name: cam_elt.name.clone(),
          device_name: format!("{}:{}", cam_elt.name.clone(), cam_ip.clone()),
          image_buff: ndarray_to_image_bytes(image_ndarray)?,
        });
      } else {
        warn!(
          "Camera Adjustments: Failed to find camera {}[{}]",
          cam_elt.name, cam_ip,
        );
      }
    }

    Ok(adjusted_image_buffers)
  }

  /// Opens a stream with the backend up, streaming images from the server.
  /// Pretty much like subscribing to the endpoint.
  ///
  /// Args:
  /// * stream_tx: Sender channel for which to write consumes images to.
  pub async fn run_camera_stream(
    &mut self,
    stream_tx: Sender<(CameraStreamResponse, Option<Vec<AdjustedCameraBuffer>>)>,
  ) -> Result<()> {
    let server_endpoint = self.endpoint.clone();
    let endpoint = format!(
      "https://{server_endpoint}/{}",
      self.api_config.get_camera_stream_endpoint_path
    );

    // Open an http request with the server, then continuously consume response chunks.
    let res = self.client.get(endpoint).body("{}").send().await?;

    // Early return on failure.
    if !res.status().is_success() {
      return Err(Error::msg(format!(
        "ClientAPI: Camera stream  failed with status code: {status}",
        status = res.status(),
      )));
    }

    info!("ClientAPI: Response headers -> {:?}", res.headers());
    let mut multipart_reader = MultiPartReader::new(res)?;

    // Calculate expected time delay to match target FPS.
    let mut start_time = Instant::now();
    let frame_delay = Duration::from_secs_f64(1.0 / self.target_fps);

    // Throughput statistics.
    let mut total_bytes_sent: usize = 0;
    let mut stat_check_time = Instant::now();

    // Instead of throttling the reader, drop images until delay matches target FPS.
    while let Some(chunk) = multipart_reader.next().await? {
      // Check statistics every second.
      if stat_check_time.elapsed().as_millis() >= 1000 {
        let total_kb_sent = total_bytes_sent / 1024;
        info!("CameraAPI: Produced ~{}KB/s", total_kb_sent);

        // Reset.
        stat_check_time = Instant::now();
        total_bytes_sent = 0;
      }

      // Restrict consumption to match target FPS.
      let elapsed_time = start_time.elapsed();
      let fps: f64 = math::round::floor(1.0 / elapsed_time.as_secs_f64(), 2);
      debug!(
        "ClientAPI: elapsed_seconds={}s | fps_now={}",
        elapsed_time.as_secs_f64(),
        fps,
      );
      if elapsed_time < frame_delay {
        continue;
      }

      // If we reach here, then target FPS has been met, proceed to produce image for consumers.
      start_time = Instant::now();
      info!(
        "ClientAPI: FPS target met -> elapsed_seconds={}s | fps_now={} | target_elapsed_seconds={}s | target_fps={}",
        elapsed_time.as_secs_f64(),
        fps,
        frame_delay.as_secs_f64(),
        self.target_fps,
      );

      // Unmarshal image buffers.
      match serde_json::from_slice::<CameraStreamResponse>(&chunk) {
        Ok(stream_response) => {
          debug!(
            "ClientAPI: Wrote deserialized camera stream response to tx channel[{}/{}]",
            stream_tx.max_capacity() - stream_tx.capacity(),
            stream_tx.max_capacity(),
          );

          // Adjust the images.
          let adjusted_images_res = self
            .apply_image_adjustments(Arc::new(&stream_response))
            .await;
          match adjusted_images_res {
            Ok(adjusted_image_buffers) => {
              // Keep track of produced buffer statistics.
              let buffer_byte_size: usize = adjusted_image_buffers
                .iter()
                .map(|elt| elt.image_buff.clone())
                .map(|buff_vec| buff_vec.len())
                .sum();
              total_bytes_sent += buffer_byte_size;

              // Ship produced buffers to consumers.
              stream_tx
                .send((stream_response, Some(adjusted_image_buffers)))
                .await?;
            }
            Err(err) => {
              warn!("ClientAPI: Image buffer adjustments failed. Only including un-adjusted buffer: {err}");
              stream_tx.send((stream_response, None)).await?;
            }
          }
        }
        // We don't want to shutdown the stream if we fail to parse a packet.
        // Packet loss/corruption could occur.
        Err(err) => {
          warn!(
            "ClientAPI: Camera Stream Response failed to deserialize: {}",
            err.to_string()
          );
        }
      }
    }

    warn!("ClientAPI: Closing camera stream.");
    Ok(())
  }
}

struct MultiPartReader {
  response: reqwest::Response,
  boundary: String,
  buffer: Vec<u8>,
}

impl MultiPartReader {
  /// Constructs a Multipart reader from the response instance, grabbing ownership of
  /// the given response instance.
  ///
  /// Args:
  /// * response: Response instance used to consume multipart packets.
  fn new(response: reqwest::Response) -> Result<Self> {
    // Grab the boundary header.
    let boundary = response
      .headers()
      .get("content-type")
      .and_then(|v| v.to_str().ok())
      .and_then(|v| v.split("=").nth(1))
      .unwrap_or_default();
    info!("Found boundary -> '{boundary}'");

    Ok(Self {
      boundary: boundary.to_owned(),
      response,
      buffer: vec![],
    })
  }

  /// Consumes the next multi-part packet from the stored response instance.
  ///
  /// # Returns:
  /// * Consumed payload packet.
  async fn next(&mut self) -> Result<Option<Vec<u8>>> {
    debug!("MultiPartReader: Buffer size -> {}B", self.buffer.len());

    // Algorithm:
    // Find boundary
    // Right find \r\n\r\n
    // Buffer until boundary is found again
    while let Some(chunk) = self.response.chunk().await? {
      // Buffer the chunk.
      let chunk_vec = chunk.to_vec();
      self.buffer.extend(chunk_vec.iter());

      // Convert buffer into string for parsing.
      let buffer = self.buffer.clone();
      let s = String::from_utf8_lossy(&buffer);

      // Find boundaries.
      let chunk_split_boundary: Vec<&str> =
        s.split(format!("--{}", self.boundary).as_str()).collect();

      // We must have at least one completed chunk.
      // We would have 2 elements if we split on boundary,
      // 3+ elements means we found two boundaries.
      debug!(
        "MultiPartReader: Boundary chunks -> {}",
        chunk_split_boundary.len()
      );
      if chunk_split_boundary.len() <= 2 {
        continue;
      }

      // The first element contains remaining headers & payload.
      let chunk_payload = chunk_split_boundary[1].split("\r\n\r\n").nth(1);

      // Verify payload is not malformed.
      return match chunk_payload {
        Some(payload) => {
          // Update stored buffer. We have at least 3 elements based on
          // check above, clear and store internal buffer with remaining chunks.
          self.buffer = chunk_split_boundary[2..].concat().as_bytes().to_vec();

          // Return result.
          Ok(Some(payload.as_bytes().to_owned()))
        }
        None => Err(Error::msg("Maformed payload chunk")),
      };
    }
    Ok(None)
  }
}
