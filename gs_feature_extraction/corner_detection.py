#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import random

class CornerDetectionNode(Node):
    def __init__(self):
        super().__init__('corner_detection_node')

        self.subscription = self.create_subscription(
            Image,
            '/gs_depth_image',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(Image, '/gs_corner_img', 10)

        self.bridge = CvBridge()
        self.max_corners = 20  # nombre max de coins à détecter
        self.quality_level = 0.01
        self.min_distance = 10
        self.block_size = 3
        self.use_harris = False
        self.k = 0.04

        self.get_logger().info("Shi-Tomasi corner detector node ready.")


    def image_callback(self, msg):
        self.get_logger().info(f"Image encoding: {msg.encoding}")
        # try:
        #     frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
        # except Exception as e:
        #     self.get_logger().error(f"CV Bridge error: {e}")            
        #     return
            
        cv_image = self.bridge.imgmsg_to_cv2(msg)

        if msg.encoding == 'mono16':
            frame = cv2.convertScaleAbs(cv_image, alpha=(255/65356))#conversion d'une image de 16 bits à 8 bits
        else:
            frame = cv_image.copy()

        #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Détection des coins
        corners = cv.goodFeaturesToTrack(
            frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris,
            k=self.k
        )

        # Affichage des coins détectés
        if corners is not None:
            for i in range(corners.shape[0]):
                x, y = corners[i, 0]
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv.circle(frame, (int(x), int(y)), 4, color, -1)

        # Affichage pour debug
        cv.imshow("Corners", frame)
        cv.waitKey(1)

        # Publication ROS
        try:
            msg_out = self.bridge.cv2_to_imgmsg(frame, encoding='passthrough')
            self.publisher.publish(msg_out)
        except Exception as e:
            self.get_logger().error(f"CV Bridge publish error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CornerDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
            