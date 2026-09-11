#!/usr/bin/env python3
"""
Subscribes to an existing RealSense image topic (published by your already-running
realsense node), runs YOLOv8n inference, and publishes:
  - annotated image (sensor_msgs/CompressedImage, JPEG) -> for rviz2 / rqt_image_view
  - structured detections (vision_msgs/Detection2DArray)
  - 3D target pose in map frame (geometry_msgs/PoseStamped) -> for the SWAP FSM,
    via depth back-projection at the best detection's bbox center

No RealSense SDK / pyrealsense2 dependency needed here anymore -- this node
just consumes whatever your camera driver is already publishing.

Output image is compressed JPEG rather than raw -- this container relays
over the zenoh bridge back to your laptop, so compression matters more
here than it would for a purely local/simulated pipeline.

REAL-HARDWARE DEPTH NOTES (ported from the sim's yolo_detector.py, which
assumed float-meters depth -- neither assumption holds here unmodified):
  - Depth topic here is /camera/aligned_depth_to_color/image_raw, which
    means it's already pixel-aligned to the COLOR image -- so the color
    camera's own intrinsics (camera_info_topic) apply directly, with no
    separate depth-to-color reprojection step needed.
  - RealSense aligned-depth topics are typically encoded 16UC1, i.e.
    integer millimeters, NOT float meters. This code checks msg.encoding
    and converts accordingly -- getting this wrong silently produces
    target poses ~1000x too far away, with no error thrown anywhere.
"""
import math

import numpy as np
import cv2

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose

from ultralytics import YOLO


class YoloSubscriberNode(Node):
    def __init__(self):
        super().__init__('yolo_subscriber_node')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('image_topic', '/go2/yolo/image_annotated/compressed')
        self.declare_parameter('detections_topic', '/go2/yolo/detections')
        self.declare_parameter('jpeg_quality', 80)
        # Comma-separated class IDs to keep, e.g. "0" for the first class
        # only. Empty string = keep all classes the loaded model knows.
        # NOTE: this also gates which detections are eligible for
        # /yolo/target_pose below, via ultralytics' own `classes=` predict
        # arg -- result.boxes only ever contains already-filtered classes.
        self.declare_parameter('classes_filter', '')

        # --- 3D back-projection (new) ---
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_pose_topic', '/yolo/target_pose')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        # Sanity bounds on accepted depth readings (metres). Real sensors
        # regularly return 0 (no return) or nonsense far values at edges/
        # reflective surfaces -- reject rather than publish garbage poses.
        self.declare_parameter('min_valid_depth_m', 0.15)
        self.declare_parameter('max_valid_depth_m', 8.0)

        model_path = self.get_parameter('model_path').value
        self.conf_thres = self.get_parameter('conf_thres').value
        input_topic = self.get_parameter('input_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        image_topic = self.get_parameter('image_topic').value
        det_topic = self.get_parameter('detections_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        classes_filter_str = self.get_parameter('classes_filter').value
        if classes_filter_str.strip():
            self.classes_filter = [int(c) for c in classes_filter_str.split(',')]
        else:
            self.classes_filter = None  # None = all classes

        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.min_valid_depth_m = self.get_parameter('min_valid_depth_m').value
        self.max_valid_depth_m = self.get_parameter('max_valid_depth_m').value

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        # For real-time speed on Orin Nano, export once to TensorRT and point
        # model_path at the .engine file instead of the .pt:
        #   YOLO('yolov8n.pt').export(format='engine')
        self.model = YOLO(model_path)
        self.get_logger().info(
            f'Model classes: {self.model.names}. '
            f'Class filter: {self.classes_filter if self.classes_filter else "all classes"}'
        )
        self.bridge = CvBridge()

        self.latest_depth = None
        self.camera_info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Two separate QoS profiles: BEST_EFFORT for subscribing (matches
        # the camera driver's conventional sensor-data QoS), and RELIABLE
        # for our own outgoing publishers. This matters: a RELIABLE-
        # requesting subscriber (rqt_image_view's default) will NOT match
        # a BEST_EFFORT publisher at all in DDS -- no error, just silence,
        # which looks exactly like "no output" with no clue why.
        sub_qos = QoSProfile(depth=5)
        sub_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sub_qos.history = HistoryPolicy.KEEP_LAST

        pub_qos = QoSProfile(depth=5)
        pub_qos.reliability = ReliabilityPolicy.RELIABLE
        pub_qos.history = HistoryPolicy.KEEP_LAST

        self.image_pub = self.create_publisher(CompressedImage, image_topic, pub_qos)
        self.det_pub = self.create_publisher(Detection2DArray, det_topic, pub_qos)
        self.pose_pub = self.create_publisher(PoseStamped, target_pose_topic, pub_qos)

        # Depth + camera_info: BEST_EFFORT to match the RealSense driver's
        # own publishing convention (same reasoning as sub_qos above).
        self.create_subscription(Image, depth_topic, self.depth_cb, sub_qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.caminfo_cb, sub_qos)

        if self.use_compressed:
            self.get_logger().info(f'Subscribing (compressed) to: {input_topic}')
            self.sub = self.create_subscription(
                CompressedImage, input_topic, self.compressed_callback, sub_qos)
        else:
            self.get_logger().info(f'Subscribing (raw) to: {input_topic}')
            self.sub = self.create_subscription(
                Image, input_topic, self.image_callback, sub_qos)

        self.get_logger().info(
            f'3D back-projection: depth_topic={depth_topic}, '
            f'camera_info_topic={camera_info_topic}, target_frame={self.target_frame}, '
            f'camera_frame={self.camera_frame} -> publishing {target_pose_topic}'
        )

    def depth_cb(self, msg: Image):
        self.latest_depth = msg

    def caminfo_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def compressed_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.run_inference(frame, msg.header)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.run_inference(frame, msg.header)

    def run_inference(self, frame, header):
        results = self.model.predict(
            frame, conf=self.conf_thres, classes=self.classes_filter, verbose=False)
        result = results[0]

        annotated = result.plot()  # BGR numpy array, boxes drawn in
        annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

        # JPEG-encode directly via cv2 rather than cv_bridge.cv2_to_imgmsg --
        # avoids a known cv_bridge/opencv-python type-code conflict when the
        # two coexist (pip-installed opencv-python from ultralytics vs. the
        # apt/source-built OpenCV cv_bridge itself was compiled against),
        # and produces a much smaller message for the zenoh bridge to relay.
        success, encoded = cv2.imencode(
            '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if success:
            img_msg = CompressedImage()
            img_msg.header = header  # keep original timestamp/frame_id
            img_msg.format = 'jpeg'
            img_msg.data = encoded.tobytes()
            self.image_pub.publish(img_msg)
        else:
            self.get_logger().warn('JPEG encoding failed, skipping this frame\'s debug image')

        det_array = Detection2DArray()
        det_array.header = header

        best_box = None
        best_conf = 0.0

        for box in result.boxes:
            cx, cy, bw, bh = box.xywh[0].tolist()

            det = Detection2D()
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = bw
            det.bbox.size_y = bh

            hyp = ObjectHypothesisWithPose()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            hyp.hypothesis.class_id = self.model.names[cls_id]
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det_array.detections.append(det)

            # result.boxes is already restricted to classes_filter by the
            # `classes=` arg passed to model.predict() above -- no need to
            # re-check class membership here, just take highest confidence.
            if conf > best_conf:
                best_conf = conf
                best_box = (cx, cy)

        self.det_pub.publish(det_array)

        if best_box is not None:
            self.publish_target_pose(best_box, header)

    def publish_target_pose(self, pixel, header):
        if self.latest_depth is None or self.camera_info is None:
            return

        cx, cy = pixel
        px, py = int(cx), int(cy)

        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
        if not (0 <= py < depth_img.shape[0] and 0 <= px < depth_img.shape[1]):
            return

        raw_val = float(depth_img[py, px])
        if raw_val <= 0.0 or math.isnan(raw_val) or math.isinf(raw_val):
            return

        # 16UC1 = integer millimetres (typical RealSense aligned-depth output).
        # 32FC1 = already float metres. Anything else is unexpected -- warn
        # once rather than silently guessing.
        encoding = self.latest_depth.encoding
        if encoding == '16UC1':
            depth_m = raw_val / 1000.0
        elif encoding == '32FC1':
            depth_m = raw_val
        else:
            self.get_logger().warn(
                f'Unrecognized depth encoding "{encoding}", assuming millimetres. '
                'Verify this against your actual depth topic if poses look wrong.',
                throttle_duration_sec=10.0
            )
            depth_m = raw_val / 1000.0

        if not (self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m):
            return

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        x = (px - ppx) * depth_m / fx
        y = (py - ppy) * depth_m / fy
        z = depth_m

        camera_pose = PoseStamped()
        camera_pose.header = header
        camera_pose.pose.position.x = x
        camera_pose.pose.position.y = y
        camera_pose.pose.position.z = z
        camera_pose.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, header.stamp,
                timeout=Duration(seconds=0.5)
            )
        except (LookupException, ExtrapolationException) as e:
            # Exact-timestamp lookup commonly fails here because AMCL only
            # republishes map->odom periodically (often a few Hz, tied to
            # scan-matching), while the camera captures at ~8Hz -- so most
            # image timestamps land ahead of the latest map correction
            # available. Falling back to the latest available transform
            # (rclpy.time.Time() == "latest") trades a small timing error
            # (bounded by AMCL's own update rate) for actually getting a
            # pose most frames, instead of failing on nearly every one.
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, self.camera_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.5)
                )
                self.get_logger().warn(
                    f'Exact-stamp TF lookup failed ({e}); used latest available '
                    'transform instead -- pose may be off by up to AMCL\'s update '
                    'interval if the robot is moving quickly.',
                    throttle_duration_sec=5.0
                )
            except (LookupException, ExtrapolationException) as e2:
                self.get_logger().warn(f'TF lookup failed for YOLO target pose (both exact and latest): {e2}')
                return

        world_pose_geom = do_transform_pose(camera_pose.pose, transform)

        out = PoseStamped()
        out.header.frame_id = self.target_frame
        out.header.stamp = header.stamp
        out.pose = world_pose_geom
        self.pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()