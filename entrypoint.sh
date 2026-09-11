#!/bin/bash
set -e
source /opt/ros/humble/install/setup.bash
source /opt/cyclonedds_ws/install/setup.bash
exec python3 /workspace/yolo_subscriber_node.py "$@"