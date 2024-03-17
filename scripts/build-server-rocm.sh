#!/usr/bin/env bash
set -e

## Vars ##
DOCKER_IMAGE_TAG="onnxruntime-rocm"
TEMP_CONTAINER_NAME="tmp-$DOCKER_IMAGE_TAG"
ONNX_GIT_REV="1eb67a07caca2fa9561af03ac47f23f5cc0cdd41"
WORKSPACE_DIR="$(realpath $(dirname $0)/..)"

# ONNXRuntime paths
ONNX_WORKSPACE_DIR="$WORKSPACE_DIR/onnxruntime-workspace"
ONNX_LD_LIB_DIR="$ONNX_WORKSPACE_DIR/onnxruntime-1.18.0"

# ROCm paths
ROCM_LD_LIB_DIR="$ONNX_WORKSPACE_DIR/rocm-6.0.0"


# Create workspace to build ONNXRuntime in.
echo "Creating workspace -> $ONNX_WORKSPACE_DIR"
mkdir "$ONNX_WORKSPACE_DIR"
cd "$ONNX_WORKSPACE_DIR"

# Build ONNX from source with ROCm support.
git clone https://github.com/microsoft/onnxruntime
git checkout $ONNX_GIT_REV
cd onnxruntime
docker build -t $DOCKER_IMAGE_TAG -f dockerfiles/Dockerfile.rocm .

# Create a temporary container
echo "Creating temporary ONNXRuntime container[$TEMP_CONTAINER_NAME] to grab the build libraries from"
docker create --name $TEMP_CONTAINER_NAME onnxruntime-rocm

# Grab the built release
cd "$ONNX_WORKSPACE_DIR"
echo "Copying built release from the container -> $ONNX_WORKSPACE_DIR..."
docker cp $TEMP_CONTAINER_NAME:/code/onnxruntime/build/Linux/Release .

# Copy needed files from the release.
echo "Creating ld library directory -> $ONNX_LD_LIB_DIR"
mkdir "$ONNX_LD_LIB_DIR"
cd "$ONNX_LD_LIB_DIR"

mkdir lib include
find "$ONNX_WORKSPACE_DIR/Release" -name '*.so' | xargs -I{} cp -v {} ./lib

# Now grab includes form the container.
docker cp $TEMP_CONTAINER_NAME:onnxruntime/include/onnxruntime/core ./include

# Yoink ROCm .so files.
mkdir "$ROCM_LD_LIB_DIR"
cd "$ROCM_LD_LIB_DIR"
mkdir lib

ROCM_LIB_PATHS=(
  /opt/rocm-6.0.0/lib/librocblas.so.4
  /opt/rocm-6.0.0/lib/librocblas.so.4.0.60000
  /opt/rocm-6.0.0/lib/libMIOpen.so
  /opt/rocm-6.0.0/lib/libMIOpen.so.1
  /opt/rocm-6.0.0/lib/libMIOpen.so.1.0.60000
  /opt/rocm-6.0.0/lib/libhipfft.so
  /opt/rocm-6.0.0/lib/libhipfft.so.0.1.60000
  /opt/rocm-6.0.0/lib/libhipfft.so.0
  /opt/rocm-6.0.0/lib/libroctracer64.so.4
  /opt/rocm-6.0.0/lib/libroctracer64.so
  /opt/rocm-6.0.0/lib/libroctracer64.so.4.1.60000
  /opt/rocm-6.0.0/lib/libamdhip64.so.6.0.60000
  /opt/rocm-6.0.0/lib/libamdhip64.so.6
  /opt/rocm-6.0.0/lib/libroctx64.so.4
  /opt/rocm-6.0.0/lib/libroctx64.so.4.1.60000
  /opt/rocm-6.0.0/lib/librocfft.so.0.1.60000
  /opt/rocm-6.0.0/lib/librocfft.so.0
  /opt/rocm-6.0.0/lib/libhiprtc.so.6.0.60000
  /opt/rocm-6.0.0/lib/libhiprtc.so.6
)
for file in "${ROCM_LIB_PATHS[@]}"; do
  docker cp $TEMP_CONTAINER_NAME:"$file" ./lib
done


# Finally. Compile and run the server binary.
export LD_LIBRARY_PATH="$ONNX_LD_LIB_DIR:$ROCM_LD_LIB_DIR:$LD_LIBRARY_PATH"

echo "When running the server, export linked libraries: 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH'"
cargo run --release --bin server

# Clean up.
echo "Cleaning up temp container..."
docker stop $TEMP_CONTAINER_NAME
docker rm $TEMP_CONTAINER_NAME


# Make sure the user wants to clean up the image, as the build takes a WHILE
# and the image is pretty large.
read -r -p "Proceed to clean up and delete docker image '$DOCKER_IMAGE_TAG'? [Y/n] " response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
  echo "Deleting $DOCKER_IMAGE_TAG image"
  docker rmi $DOCKER_IMAGE_TAG
fi

