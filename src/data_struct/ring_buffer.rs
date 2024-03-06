use std::{
  collections::VecDeque,
  fmt::Debug,
  sync::{Condvar, Mutex},
};
use tokio::sync;

use anyhow::{Error, Result};

pub struct RingBuffer<T: Clone + Debug> {
  capacity: usize,
  buffer: sync::RwLock<VecDeque<T>>,
  head: usize,
  tail: usize,

  // Conditional variables to wait on.
  not_empty: Condvar,
  lock: Mutex<bool>,
}

impl<T: Clone + Debug> RingBuffer<T> {
  pub fn new(capacity: usize) -> Self {
    Self {
      buffer: sync::RwLock::new(VecDeque::<T>::new()),
      capacity,
      head: 0,
      tail: 0,
      not_empty: Condvar::new(),
      lock: Mutex::new(false),
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
    self.not_empty.notify_one();
  }

  /// Returns the head value from the stored RingBuffer instance if
  /// available.
  pub async fn pop(&mut self) -> Result<T, Error> {
    let buffer = self.buffer.read().await;
    while self.is_empty() {
      let lock = self.lock.lock().unwrap();
      let _ = self.not_empty.wait(lock);
    }

    let buffer_idx = self.head;
    self.head = (self.head + 1) % self.capacity;
    let value = buffer[buffer_idx].clone();
    Ok(value)
  }

  /// Returns whether the RingBuffer is empty or not.
  pub fn is_empty(&self) -> bool {
    self.head == self.tail
  }
}
