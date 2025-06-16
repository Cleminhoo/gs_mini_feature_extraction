#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')

        self.subscription = self.create_subscription(
            Image,
            '/gs_mini_image',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(Image, '/gs_opt_flow_img', 10)

        self.bridge = CvBridge()
        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.color = np.random.randint(0, 255, (100, 3))

        # Params
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def image_callback(self, msg):
        self.get_logger().info(f"Image encoding: {msg.encoding}")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            self.mask = np.zeros_like(frame)
            return

        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

            output = cv.add(frame, self.mask)
            cv.imshow("Optical Flow", output)
            cv.waitKey(1)

            # Publier l'image traitée (optionnel)
            try:
                msg_out = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
                self.publisher.publish(msg_out)
            except Exception as e:
                self.get_logger().error(f"CV Bridge publish error: {e}")

            # Mettre à jour les données pour la prochaine itération
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
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
