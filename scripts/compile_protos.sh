#!/usr/bin/env bash

SCRIPTS_DIR=$(realpath "$(dirname "$0")")
PROTO_DIR="$SCRIPTS_DIR/../proto"
ROOT_DIR="$SCRIPTS_DIR/.."

echo "[python] Compiling *.proto..."
find "$PROTO_DIR" -maxdepth 1 -name '*.proto' | xargs -I {} \
  python -m grpc_tools.protoc \
    -I="$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    {};


echo "[golang] Compiling *.proto..."
find "$PROTO_DIR" -maxdepth 1 -name '*.proto' | xargs -I {} \
  protoc \
    -I="$PROTO_DIR" \
    --go_out="$ROOT_DIR" \
    {};