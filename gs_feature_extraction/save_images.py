import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class DepthImageSaver(Node):
    def __init__(self):
        super().__init__('depth_image_saver')

        # Declare a parameter for the output directory
        self.declare_parameter('output_dir', 'saved_depth_images')
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value

        # Create the directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"Saving depth images to: {self.output_dir}")

        # Initialize subscriber and utilities
        self.subscription = self.create_subscription(
            Image,
            '/gs_mini_img',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.counter = 0

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Build filename and save the image
            filename = f"depth_{self.counter:04d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, cv_image)
            self.get_logger().info(f"Saved: {filename}")

            self.counter += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
