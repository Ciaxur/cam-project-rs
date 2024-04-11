# The Camera Project

Camera object detection client and server project. Server provides an API for running
the object detection model and the client handles providing the images to be used as input.

# Usage

Both client & server require OpenCV.

- `ArchLinux` -> https://archlinux.org/packages/extra/x86_64/opencv

## Client

Client consumes images from local camera device and from an HTTP endpoint.
The client uses a `json config`. See [config_example](./config_client_example.json) for more info.

```sh
# Build the client.
$ cargo build --release

# Run the client.
$ aptcam_client -c config.json
```

## Server

The server is a gRPC server which provides an endpoint for which to run a model on supplied
images within an open stream.
The server uses a `json config`. See [config_example](./config_server_example.json) for more info.


```sh
# Build the server
$ cargo build --bin server --release

# Run the server
$ aptcam_server \
  --port 6969 \
  -c config.json
```

# Models

Additional models can be found here:

- https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model
- https://github.com/GuodongQi/yolo3_tensorflow
- https://github.com/ViswanathaReddyGajjala/SSD_MobileNet
- Labels https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8

# Resources

- https://docs.opencv.org/3.4
- https://www.ultralytics.com/
