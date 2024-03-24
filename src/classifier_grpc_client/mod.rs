pub mod classifier {
  tonic::include_proto!("classifier");
}

use anyhow::{Error, Result};
use classifier::image_classifier_client::ImageClassifierClient;
use classifier::{ClassifyImageRequest, ClassifyImageResponse};
use log::{debug, error, info, warn};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Endpoint};

#[derive(Debug)]
pub struct ClassifierClient {
  // gRPC server's endpoint for which to connect the client to.
  endpoint: String,

  // Vector of tracked active tokio tasks. This is used to track
  // and stop active workers.
  tokio_tasks: Vec<tokio::task::JoinHandle<()>>,

  // gRPC channel buffer size.
  channel_buff_size: usize,
}

impl ClassifierClient {
  /// Instantiates a new client instance to handle image classification
  /// requests and responses with the given server endpoint.
  ///
  /// Args:
  /// * endpoint: Endpoint to the gRPC classifier server.
  /// * chan_buff_size: gRPC channel buffer size.
  pub fn new(endpoint: String, channel_buff_size: usize) -> ClassifierClient {
    Self {
      endpoint,
      tokio_tasks: vec![],
      channel_buff_size,
    }
  }

  /// Stops all active tokio tasks.
  pub fn stop(&mut self) {
    let _ = self.tokio_tasks.iter().map(|task| task.abort());
    self.tokio_tasks.clear();
  }

  /// Starts the gRPC classifier client with the instantiated server endpoint,
  /// using a pair of sender and receiver channels for which to be used to supply a
  /// stream of requests for the client to issue and a stream of responses for
  /// which to fill the channel with for consumption.
  ///
  /// Args:
  /// * tx_response: Sender channel for which to push requests into for the client to issue
  ///   to the server.
  /// * rx_request: mpsc::Receiver channel used to consume pushed requests to proxy to the classifier server.
  pub async fn start(
    &mut self,
    tx_response: mpsc::Sender<ClassifyImageResponse>,
    rx_request: mpsc::Receiver<ClassifyImageRequest>,
  ) -> Result<()> {
    // Construct a channel endpoint for which to be used to create a channel from.
    let channel_endpoint: Endpoint = Channel::from_shared(self.endpoint.clone())?
      .concurrency_limit(10)
      .connect_timeout(Duration::from_secs(5))
      .buffer_size(self.channel_buff_size);

    // Start tokio workers which handle the stream.
    let task = tokio::spawn(stream_call_with_retry(
      channel_endpoint,
      tx_response,
      rx_request,
      15,
    ));

    // Track active workers.
    self.tokio_tasks.push(task);

    Ok(())
  }
}

/// Helper function which invokes a stream classifier call with retry logic.
///
/// Retry logic is based on a 2s cooldown with 200ms backoff. If we fail within 2s of every
/// retry attempt, then we fall over. If we succeed at least once, and the cooldown period
/// passed, then the retry attempts reset.
///
/// Args:
/// * endpoint: Constructed Endpoint instance used to create a channel.
/// * tx_response: Sender mpsc channel to send server responses on.
/// * rx_request: Receiver mpsc channel to read and proxy requests to server from.
/// * max_retries: Maximum number of retries before falling over.
async fn stream_call_with_retry(
  endpoint: Endpoint,
  tx_response: mpsc::Sender<ClassifyImageResponse>,
  rx_request: mpsc::Receiver<ClassifyImageRequest>,
  max_retries: u64,
) {
  // Track number of retry attempts with basic backoff.
  // The maximum number of retries signify multiple attempts within
  // an arbitrary cooldown threashold from the last retry attempt.
  //
  // If the cooldown is met, then we the retry counter resets.
  let cooldown_threashold = Duration::from_secs(2);
  let backoff_duration = Duration::from_millis(200);
  let mut last_retry_attempt = Instant::now();
  let mut retries_left = max_retries;
  let rw_rx_request = RwLock::new(rx_request);

  loop {
    match stream_call(&endpoint, &tx_response, &rw_rx_request).await {
      Ok(()) => {
        break;
      }
      Err(err) => {
        error!("{err}");
        retries_left -= 1;

        // Check if we've run out retry attempts.
        if retries_left <= 0 {
          warn!(
            "ClassifierClient: Ran out of retry attempts [{}/{}]",
            retries_left, max_retries
          );

          // We're responsible for closing the recv channel!
          rw_rx_request.write().await.close();
          break;
        }

        // Proceed with retry logic.
        info!(
          "ClassifierClient: Attempting retry logic with -> retries=[{}/{}] | cooldown={:?} | backoff={:?}",
          retries_left,
          max_retries,
          cooldown_threashold,
          backoff_duration,
        );

        // Reset retries if cooldown threadshold met.
        if last_retry_attempt.elapsed() > cooldown_threashold {
          retries_left = max_retries;
        }
        tokio::time::sleep(backoff_duration).await;
        last_retry_attempt = Instant::now();
      }
    }
  }
  warn!("ClassifierClient: Terminated grpc client consumer");
}

/// Internal helper function used to start a request stream that
/// proxies requests from the given mpsc channel to the gRPC server endpoint,
/// then sends responses on the given mpsc channel.
///
/// This function wraps the given receiver channel internally to prevent
/// it from closing upon a gRPC connection failure. The caller is responsible
/// for closing and draining the channel.
///
/// Args:
/// * endpoint: Constructed Endpoint instance used to create a channel.
/// * tx_response: Sender mpsc channel to send server responses on.
/// * rx_request: Receiver mpsc channel to read and proxy requests to server from.
async fn stream_call(
  endpoint: &Endpoint,
  tx_response: &mpsc::Sender<ClassifyImageResponse>,
  rw_rx_request: &RwLock<mpsc::Receiver<ClassifyImageRequest>>,
) -> Result<()> {
  // Establish a connection using the given endpoint.
  let channel = endpoint.connect().await?;
  let mut client = ImageClassifierClient::new(channel);
  let mut rx_request = rw_rx_request.write().await;

  // Create another mpsc to act as a middleman for proxying ClassifyImageRequest from
  // the given mpsc rx channel.
  //
  // Reason for doing so is cause the mpsc channel closes if the gRPC client connection
  // faults. To mitigate that, we're running a concurrent task that wraps around
  // the given rx request channel, such that if it closes, it doesn't disrupt the
  // caller. Plus gives us an opportunity to retry.
  let (tx, rx) = mpsc::channel::<ClassifyImageRequest>(tx_response.capacity());

  // Write at least one request to the buffer.
  if let Some(req) = rx_request.recv().await {
    tx.send(req).await?;
  } else {
    return Err(Error::msg("ClassifierClient: Request proxy channel closed"));
  }

  // Wrap tokio's receiver channel around a Stream.
  let stream_wrapped_rx_request = ReceiverStream::new(rx);

  // Open a stream of requests from the channel to the server.
  info!("ClassifierClient: Starting producer");
  let classify_response = client
    .classify_image(stream_wrapped_rx_request)
    .await
    .map_err(|err| Error::msg(format!("Failed to invoke classify_image: {err}")))?;

  // Start the stream.
  /*
    Message response info:
      Result::Err(val) means a gRPC error was sent by the sender instead of a valid response message.
        Refer to Status::code and Status::message to examine possible error causes.
      Result::Ok(None) means the stream was closed by the sender and no more messages will be delivered.
        Further attempts to call Streaming::message will result in the same return value.
      Result::Ok(Some(val)) means the sender streamed a valid response message val.
  */
  info!("ClassifierClient: Starting consumer");
  let mut resp_stream = classify_response.into_inner();
  loop {
    // Concurrently consume server responses and client requests via
    // tokio context select statement.
    tokio::select! {
      state = resp_stream.message() => {
        match state {
          Ok(resp_msg) => {
            if let Some(msg) = resp_msg {
              debug!(
                "ClassifierClient: Received -> (device{} | matches={:?} | scores={:?})",
                msg.device, msg.matches, msg.match_scores,
              );

              tx_response.send(msg).await.map_err(|err| {
                Error::msg(
                  format!(
                    "ClassifierClient: Failed to write response to channel -> {}",
                    err
                  )
                )
              })?;
            } else {
              warn!("ClassifierClient: Client closed stream. Terminating comsumer nominally.");
              break;
            }
          }
          Err(rpc_err) => {
            return Err(Error::msg(
              format!(
                "ClassifierClient: Closing stream due to an rpc error[code={}] -> {}",
                rpc_err.code(),
                rpc_err.message()
              )
            ));
          }
        }
      }

      state = rx_request.recv() => {
        // Simply proxy the request.
        match state {
          Some(classify_req) => {
            tx.send(classify_req).await.map_err(|err| {
              Error::msg(format!(
                "ClassifierClient: Request tx proxy channel error -> {err}"
              ))
            })?;
          }
          None => {
            return Err(Error::msg("ClassifierClient: Request proxy channel closed"));
          }
        }
      },
    }
  }
  Ok(())
}
