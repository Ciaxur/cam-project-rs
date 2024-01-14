# Aparmtment Camera
Camera object detection client and server project. Server provides an API for running
the object detection model and the client handles providing the images to be used as input.

*WARNING*: This project is not generic and is dependent on 4bit.


# Benchmarks
## YoloV8
`Git`: https://github.com/ultralytics/assets
`CPU`: 0.2s | 130% CPU
`Intel GPU`: 2s | 100% CPU | ?% GPU

## yolos-tiny
Git: https://huggingface.co/hustvl/yolos-tiny
`Intel GPU`: 0.8s | 200% CPU | ?% GPU

## resnet-50
Git: https://huggingface.co/microsoft/resnet-50
`Intel GPU`: 0.1s | 154% CPU | ?% GPU


# Usage
Both client & server require OpenCV.
- `ArchLinux` -> https://archlinux.org/packages/extra/x86_64/opencv

## Client
Client consumes images from local camera device and from an HTTP endpoint.
The client uses a `json config`. See [config_example](./config_example.json) for more info.
```sh
# Build the client.
$ cargo build --release

# Run the client.
$ aptcam_client -c config.json
```

## Server
The server is a gRPC server which provides an endpoint for which to run a model on supplied
images within an open stream.
```sh
# Create a python virtual environment
$ python -m venv opencv-venv
$ source ./opencv-venv/bin/activate

# Install modules
$ python -m pip install ./requirements.txt

# Run the server
$ python ./src/server.py \
  --threshold=0.7 \
  --image_store_dir=/path/to/matched/images/store \
  --port 6969
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
