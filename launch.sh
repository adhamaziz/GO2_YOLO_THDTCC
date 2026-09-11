#!/usr/bin/env bash
# Friendly launcher for the go2-yolo container.
#
# Usage:
#   ./launch.sh                    # all 80 COCO classes, default settings
#   ./launch.sh --person           # person only (class 0)
#   ./launch.sh --classes 0,2      # person + car
#   ./launch.sh --conf 0.6 --person
#   ./launch.sh --build            # rebuild the image first, then launch
#
set -e

IMAGE_NAME="go2-yolo:latest"
CONTAINER_NAME="yolo_go2"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
CONF_THRES="0.5"
CLASSES_FILTER=""
INPUT_TOPIC="/camera/color/image_raw"
IMAGE_TOPIC="/go2/yolo/image_annotated"
DET_TOPIC="/go2/yolo/detections"
MODEL_PATH="yolov8n.pt"
DO_BUILD=0

usage() {
  echo "Usage: $0 [--person] [--classes 0,2,...] [--conf 0.5] [--topic /camera/color/image_raw] [--build]"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --person)     CLASSES_FILTER="0"; shift ;;
    --classes)    CLASSES_FILTER="$2"; shift 2 ;;
    --conf)       CONF_THRES="$2"; shift 2 ;;
    --topic)      INPUT_TOPIC="$2"; shift 2 ;;
    --build)      DO_BUILD=1; shift ;;
    -h|--help)    usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# --- Pre-flight checks -----------------------------------------------------

if [ "${DO_BUILD}" = "1" ]; then
  echo "==> Building ${IMAGE_NAME} (this can take a while, see Dockerfile.subscriber comments)"
  docker build -f Dockerfile.subscriber -t "${IMAGE_NAME}" .
fi

if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "==> Image ${IMAGE_NAME} not found. Building it now..."
  docker build -f Dockerfile.subscriber -t "${IMAGE_NAME}" .
fi

echo "==> Checking for ${INPUT_TOPIC} on the host ROS2 graph (5s timeout)..."
if command -v ros2 > /dev/null 2>&1; then
  if timeout 5 ros2 topic list 2>/dev/null | grep -qx "${INPUT_TOPIC}"; then
    echo "    Found it -- realsense node looks like it's running."
  else
    echo "    WARNING: ${INPUT_TOPIC} not seen on the host. Is your realsense node running?"
    echo "    Continuing anyway -- the container will just wait for frames."
  fi
else
  echo "    (ros2 CLI not on PATH in this shell, skipping check)"
fi

# --- Build the -p args array (only include classes_filter if set) ----------

ROS_ARGS=(
  -p "model_path:=${MODEL_PATH}"
  -p "conf_thres:=${CONF_THRES}"
  -p "input_topic:=${INPUT_TOPIC}"
  -p "image_topic:=${IMAGE_TOPIC}"
  -p "detections_topic:=${DET_TOPIC}"
)
if [ -n "${CLASSES_FILTER}" ]; then
  ROS_ARGS+=(-p "classes_filter:='${CLASSES_FILTER}'")
fi

echo "==> Launching ${CONTAINER_NAME}"
echo "    ROS_DOMAIN_ID   = ${ROS_DOMAIN_ID}"
echo "    input_topic     = ${INPUT_TOPIC}"
echo "    conf_thres      = ${CONF_THRES}"
echo "    classes_filter  = ${CLASSES_FILTER:-<all 80 COCO classes>}"
echo "    output image    = ${IMAGE_TOPIC}"
echo "    output detections = ${DET_TOPIC}"
echo

exec docker run -it --rm \
  --runtime nvidia \
  --network host \
  --name "${CONTAINER_NAME}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  "${IMAGE_NAME}" \
  --ros-args "${ROS_ARGS[@]}"