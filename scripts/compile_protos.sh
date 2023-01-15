#!/usr/bin/env bash

SCRIPTS_DIR=$(realpath "$(dirname "$0")")
PROTO_DIR="$SCRIPTS_DIR/../proto"
SRC_DIR="$SCRIPTS_DIR/../src"

echo "Compiling *.proto..."
find "$PROTO_DIR" -maxdepth 1 -name '*.proto' | xargs -I {} \
  python -m grpc_tools.protoc \
    -I="$PROTO_DIR" \
    --python_out="$SRC_DIR" \
    --grpc_python_out="$SRC_DIR" \
    {};