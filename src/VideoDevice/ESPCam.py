#!/usr/bin/env python3.10
import logging
import socket
from collections import deque
from io import BytesIO
from threading import Thread
from time import sleep


class ESP32_CAM():
    device_name: str
    _device_name: str
    _device_ip: str
    _device_port: int
    _data_end: str = b"\r\nDone\r\n"
    _sock: socket.socket
    _running: bool = False
    _image_query_thread: Thread = None
    _verbose: bool

    # Image queue buffer.
    _queue_size = 5     # Buffer latest 5 images.
    _queue: deque

    def __init__(self, device_ip: str, device_port: int, device_name="", verbose=False) -> None:
        self._device_name = device_name
        self._device_ip = device_ip
        self._device_port = device_port
        self._queue = deque()
        self._verbose = verbose

        # Construct device name.
        self.device_name = device_ip if device_name == "" else f"{device_name}:{device_ip}"

    def _start_image_query_thread(self) -> None:
        logging.info('[Image Query] Attempting to start image query thread')
        if not self._image_query_thread:
            self._running = True

            # Thread logic.
            def _thread_impl():
                logging.info('[Image Query] Running image query thread')
                try:
                    buffer = BytesIO()

                    # Send arbitrary data to let the device know client's ready.
                    self._sock.sendall(b"I'm ready!")

                    while self._running:
                        # Buffer the data.
                        data = self._sock.recv(4096)
                        if len(data) == 0:
                            raise Exception('No data received')

                        buffer.write(data)

                        # Check if the completed image data was recieved in the buffer.
                        first_data_end_index = buffer.getvalue().find(self._data_end)
                        if (first_data_end_index != -1):
                            # Queue the image.
                            image = buffer.getvalue()[:first_data_end_index]
                            self._queue.append(image)
                            if (self._verbose):
                                logging.debug(f"[Image Query] Image size: {first_data_end_index}B.")

                            # Keep the queue within the size limit.
                            if len(self._queue) > self._queue_size:
                                self._queue.popleft()

                            # Reset the buffer, keeping existing data after the newline.
                            buffer.seek(first_data_end_index + len(self._data_end))
                            existing_data = buffer.read()
                            buffer.close()
                            buffer = BytesIO(existing_data)
                except socket.timeout:
                    logging.error('[Image Query] Thread client socket timed out')
                except Exception as e:
                    logging.error(f'[Image Query] Failed to recieve image: {str(e)}')
                finally:
                    logging.info('[Image Query] Image Query stopping')
                    self.stop()

            # Start the thread.
            logging.info('Starting image thread')
            self._image_query_thread = Thread(target=_thread_impl)
            self._image_query_thread.start()

    def start(self) -> None:
        """
            Start the socket and buffer incoming images.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._device_ip, self._device_port))
        self._sock.settimeout(1)
        self._start_image_query_thread()

    def stop(self) -> None:
        if self._sock:
            logging.info(f"[Image Query:stop] Stopping: {self._device_ip}")
            self._sock.close()
            self._sock = None
            self._running = False
            try:
                self._image_query_thread.join()
            except Exception as e:
                logging.error(f"[Image Query:stop] Failed to join thread: {e}")
            finally:
                self._image_query_thread = None

    def is_running(self) -> bool:
        return self._running

    def clear_images(self) -> None:
        """Clears images in the queue."""
        self._queue.clear()

    def get_image(self, blocking=True) -> bytes:
        """
            Gets the latest image from the queue.

            Args:
                blocking: State of blocking current thread for 5s until an image has filled the queue.
        """
        # Block waiting on queue with 5s timeout.
        if blocking:
            for _ in range(5 * 100):
                if len(self._queue) > 0:
                    break
                sleep(1 / 100)
        return self._queue.pop() if len(self._queue) else b""
