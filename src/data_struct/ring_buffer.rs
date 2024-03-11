use std::fmt::Debug;
use tokio::sync;

pub enum RingBufferResult<T> {
  Some(T),
  Empty,
}

pub struct RingBuffer<T: Clone + Debug + Default> {
  capacity: usize,
  buffer: sync::RwLock<Vec<T>>,
  head: usize,
  tail: usize,
}

impl<T: Clone + Debug + Default> RingBuffer<T> {
  pub fn new(capacity: usize) -> Self {
    let mut v = Vec::<T>::new();
    v.resize(capacity, T::default());

    Self {
      buffer: sync::RwLock::new(v),
      capacity,
      head: 0,
      tail: 0,
    }
  }

  /// Pushes the given value into the RingBuffer.
  ///
  /// Args:
  /// * value: Value to add into the RingBuffer.
  pub async fn push(&mut self, value: T) {
    let mut buffer = self.buffer.write().await;

    let next_tail = (self.tail + 1) % self.capacity;
    buffer[next_tail] = value;

    // Buffer was full and we started overwriting head.
    if next_tail == self.head {
      self.head = (self.head + 1) % self.capacity;
    }
    self.tail = next_tail;
  }

  /// Returns the head value from the stored RingBuffer instance if
  /// available.
  pub async fn pop(&mut self) -> RingBufferResult<T> {
    if self.is_empty() {
      return RingBufferResult::Empty;
    }

    let buffer = self.buffer.read().await;
    let buffer_idx = self.head;
    self.head = (self.head + 1) % self.capacity;
    let value = buffer[buffer_idx].clone();
    RingBufferResult::Some(value)
  }

  /// Returns whether the RingBuffer is empty or not.
  pub fn is_empty(&self) -> bool {
    self.head == self.tail
  }
}
