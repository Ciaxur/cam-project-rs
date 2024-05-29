use log::{error, info};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug)]
pub struct StorageManagerFile {
  // File extension.
  pub ext: String,

  // Device name.
  pub device_name: String,

  // File data.
  pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct StorageManager {
  // Path to save images to.
  // Images are stored under a sub-directory by date '%Y-%m-%d'.
  // Image names follow 'hooman-<DATE>-<DEVICE_NAME>'.
  // Date is saved as RFC3339 using seconds, ie '2018-01-26T18:30:09Z'.
  // Example saved image would be '2018-01-26/hooman-2018-01-26T18:30:09.123Z-video0.jpeg'.
  image_storage_path: String,

  // Size of the channel buffer.
  // mpsc channels are used to listen and invoke bulk file writes.
  buffer_size: usize,

  // Tracked running tokio tasks.
  tokio_tasks: Vec<JoinHandle<()>>,
}

impl StorageManager {
  /// Creates a new Storage manager that tracks storing images
  /// to the given path.
  ///
  /// Image storage structure:
  /// - Images are stored under a sub-directory by date '%Y-%m-%d'.
  /// - Image names follow 'hooman-<DATE>-<DEVICE_NAME>'.
  /// - Date is saved as RFC3339 using seconds, ie '2018-01-26T18:30:09Z'.
  ///
  /// Args:
  /// * image_storage_path: Path to save images to.
  /// * buffer_size: Size of the channel buffer.
  pub fn new(image_storage_path: String, buffer_size: usize) -> Self {
    Self {
      image_storage_path,
      buffer_size,
      tokio_tasks: vec![],
    }
  }

  /// Starts a background worker which consumes images in bulk to be written
  /// disk from the sender channel.
  ///
  /// Returns:
  /// * mpsc sender channel to write images being saved.
  pub fn start(&mut self) -> mpsc::Sender<StorageManagerFile> {
    let (image_tx, image_rx) = mpsc::channel::<StorageManagerFile>(self.buffer_size);
    let buffer_size = self.buffer_size;
    let image_storage_path = self.image_storage_path.clone();

    let task = tokio::spawn(async move {
      let mut rx = image_rx;
      let mut image_buffer: Vec<StorageManagerFile> = vec![];

      while rx.recv_many(&mut image_buffer, buffer_size).await > 0 {
        // Create a sub-directory based on the current date.
        let subdir_name = chrono::Local::now().format("%Y-%m-%d").to_string();
        let abs_storage_path = format!("{}/{}", image_storage_path, subdir_name);
        if let Err(err) = tokio::fs::create_dir(abs_storage_path.clone()).await {
          if err.kind() != std::io::ErrorKind::AlreadyExists {
            error!(
              "StorageManager: Failed to create sub-directory '{}' under '{}' -> {}",
              subdir_name, image_storage_path, err
            );
            continue;
          }
        }

        // Grab current timestamp used on all the files.
        let timestamp = chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        // Write all images to file.
        for file in image_buffer.iter() {
          // Construct image path.
          let image_path = format!(
            "{}/hooman-{}-{}.{}",
            abs_storage_path, timestamp, file.device_name, file.ext
          );
          info!("StorageManager: Saving file {}", image_path);
          let _ = tokio::fs::write(image_path, file.data.to_owned()).await;
        }

        image_buffer.clear();
      }
    });
    self.tokio_tasks.push(task);

    return image_tx;
  }

  /// Stops all running tasks.
  pub fn stop(&mut self) {
    let _ = self.tokio_tasks.iter().map(|task| task.abort());
    self.tokio_tasks.clear();
  }
}
