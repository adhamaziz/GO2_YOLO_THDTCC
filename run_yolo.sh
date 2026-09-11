#!/usr/bin/env bash
# Launch script for the YOLO subscriber container on the Go2's Jetson.
#
# Usage:
#   ./run_yolo.sh                          # defaults: all classes
#   CLASSES_FILTER=0 ./run_yolo.sh         # first model class only
#   MODEL_PATH=/workspace/fire_extinguisher_best.pt ./run_yolo.sh
#   CONF_THRES=0.6 CLASSES_FILTER=0 ./run_yolo.sh
#
set -e

# Resolve paths relative to this script's own location, not the caller's
# current directory -- so this works correctly no matter where you run
# `./run_yolo.sh` from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-go2-yolo:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-yolo_go2}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"          # match go2_ws and your zenoh bridge config

# Host-side weights file (defaults to the fine-tuned model that lives in
# this project's models/ folder) and where it gets mounted inside the
# container. Override HOST_MODEL_PATH if you want to point at a different
# .pt file without editing this script.
HOST_MODEL_PATH="${HOST_MODEL_PATH:-${SCRIPT_DIR}/models/fire_extinguisher_best.pt}"
CONTAINER_MODEL_PATH="/workspace/fire_extinguisher_best.pt"
MODEL_PATH="${MODEL_PATH:-${CONTAINER_MODEL_PATH}}"

CONF_THRES="${CONF_THRES:-0.5}"
INPUT_TOPIC="${INPUT_TOPIC:-/camera/color/image_raw}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/go2/yolo/image_annotated/compressed}"
DET_TOPIC="${DET_TOPIC:-/go2/yolo/detections}"
CLASSES_FILTER="${CLASSES_FILTER:-}"          # e.g. "0" = first model class only. Empty = all classes.

# --- 3D back-projection (new) -----------------------------------------
# enable_depth:=true / align_depth.enable:=true must be set on the
# RealSense launch for this to actually produce poses -- see
# OPERATIONS_README.md section 3.5.
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/aligned_depth_to_color/image_raw}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
TARGET_POSE_TOPIC="${TARGET_POSE_TOPIC:-/yolo/target_pose}"
TARGET_FRAME="${TARGET_FRAME:-map}"
CAMERA_FRAME="${CAMERA_FRAME:-camera_color_optical_frame}"

if [ ! -f "${HOST_MODEL_PATH}" ]; then
  echo "ERROR: model file not found at ${HOST_MODEL_PATH}"
  echo "       Set HOST_MODEL_PATH=/path/to/your.pt if it lives elsewhere."
  exit 1
fi

echo "Launching ${CONTAINER_NAME}"
echo "  ROS_DOMAIN_ID   = ${ROS_DOMAIN_ID}"
echo "  model (host)    = ${HOST_MODEL_PATH}"
echo "  model (in container) = ${MODEL_PATH}"
echo "  input_topic     = ${INPUT_TOPIC}"
echo "  conf_thres      = ${CONF_THRES}"
echo "  classes_filter  = ${CLASSES_FILTER:-<all>}"
echo "  output image    = ${IMAGE_TOPIC}"
echo "  output detections = ${DET_TOPIC}"
echo "  depth_topic     = ${DEPTH_TOPIC}"
echo "  camera_info_topic = ${CAMERA_INFO_TOPIC}"
echo "  target_frame    = ${TARGET_FRAME}"
echo "  camera_frame    = ${CAMERA_FRAME}"
echo "  output target_pose = ${TARGET_POSE_TOPIC}"

# Build the -p args array so we only pass classes_filter when it's actually
# set -- an empty `-p classes_filter:=` (no value after :=) is rejected by
# ROS2's parameter parser outright.
ROS_ARGS=(
  -p "model_path:=${MODEL_PATH}"
  -p "conf_thres:=${CONF_THRES}"
  -p "input_topic:=${INPUT_TOPIC}"
  -p "image_topic:=${IMAGE_TOPIC}"
  -p "detections_topic:=${DET_TOPIC}"
  -p "depth_topic:=${DEPTH_TOPIC}"
  -p "camera_info_topic:=${CAMERA_INFO_TOPIC}"
  -p "target_pose_topic:=${TARGET_POSE_TOPIC}"
  -p "target_frame:=${TARGET_FRAME}"
  -p "camera_frame:=${CAMERA_FRAME}"
)
if [ -n "${CLASSES_FILTER}" ]; then
  # Wrap in single quotes so ROS2's YAML-based param parser treats this as
  # a string (e.g. "0") instead of inferring an integer type, which would
  # conflict with the string-typed default declared in the node.
  ROS_ARGS+=(-p "classes_filter:='${CLASSES_FILTER}'")
fi

docker run -it --rm \
  --runtime nvidia \
  --network host \
  --name "${CONTAINER_NAME}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -v "${HOST_MODEL_PATH}:${CONTAINER_MODEL_PATH}" \
  "${IMAGE_NAME}" \
  --ros-args "${ROS_ARGS[@]}"