# YOLOv8n + RealSense on Go2 Jetson Orin Nano (Docker)

## What this does
Single container, running on the Go2's onboard Jetson Orin Nano:
- Opens the RealSense color stream directly (`pyrealsense2`)
- Runs YOLOv8n inference per frame
- Publishes:
  - `/go2/yolo/image_annotated` (`sensor_msgs/Image`) — frame with boxes drawn, for viewing
  - `/go2/yolo/detections` (`vision_msgs/Detection2DArray`) — structured boxes/classes/scores

Your existing zenoh bridge carries these topics to your laptop's ROS2 Humble docker,
where you view them with `rqt_image_view` or `rviz2`. No changes needed on the
zenoh side as long as topic names/domain match what the bridge is configured to relay.

## Before building

1. **Check your L4T / JetPack version** on the Go2's Jetson:
   ```bash
   cat /etc/nv_tegra_release
   ```
   Then pick a matching base image tag in the Dockerfile from
   https://github.com/dusty-nv/jetson-containers (search `ros:humble`).
   Getting this wrong is the most common reason these images fail to build/run.

2. **RealSense on Jetson is the real gotcha.** `pip install pyrealsense2` gives you
   an x86_64 wheel that will not work on the Orin Nano. You need librealsense +
   its Python bindings built for aarch64/L4T. Follow Intel's Jetson install guide:
   https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
   Build it once, then either bake it into the Docker image (add the build steps
   to the Dockerfile) or bind-mount a prebuilt install into the container.

## Build

```bash
docker build -t go2-yolo:latest .
```

## Run (on the Go2, in your Docker environment)

```bash
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  --name yolo_go2 \
  go2-yolo:latest
```

Notes on the flags:
- `--runtime nvidia` — required for GPU/CUDA access to run YOLO at usable FPS.
- `--network host` — **important**. ROS2/DDS discovery relies on multicast; in
  Docker's default bridge network mode, your zenoh bridge (or other DDS
  participants on the host) generally won't discover topics inside the
  container. Host networking sidesteps this. If you truly need bridge
  networking, you'll need to configure DDS discovery (e.g. Cyclone DDS peer
  lists / unicast discovery) explicitly.
- `-v /dev/bus/usb:/dev/bus/usb` + `--privileged` — USB passthrough so the
  container can see the RealSense camera. You can narrow `--privileged` down
  to specific `--device` flags once you know the camera's device nodes.
- Match `ROS_DOMAIN_ID` (env var) to whatever your zenoh-bridge-ros2dds config
  expects, if it's not domain 0.

## View on your laptop (ROS2 Humble docker)

```bash
# quick view
ros2 run rqt_image_view rqt_image_view /go2/yolo/image_annotated

# or in rviz2: Add -> By topic -> /go2/yolo/image_annotated -> Image
rviz2
```

Confirm the topic is visible first with:
```bash
ros2 topic list
ros2 topic hz /go2/yolo/image_annotated
```
If it doesn't show up, the issue is almost always DDS/zenoh discovery, not YOLO.

## Performance tip
YOLOv8n on Orin Nano via plain PyTorch will run, but for real-time FPS export
to TensorRT once, then load the `.engine` file instead of the `.pt`:
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='engine')  # produces yolov8n.engine
```
Then set the `model_path` parameter to `yolov8n.engine` when launching the node.