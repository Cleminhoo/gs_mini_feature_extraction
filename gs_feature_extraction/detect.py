#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
from .utils import *
from .models import UnetPlusPlus, PSPNet, DeepLabV3Plus, UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class PanelLineDetector(Node):
    def __init__(self):
        super().__init__('panels_lines_detection')
        
        # Initialisation
        self.flag = False
        self.bridge = CvBridge()

        # Publisher
        self.detection_img_pub = self.create_publisher(Image, '/gs_img_nn', 10)

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/gs_mini_img',
            self.callback,
            10
        )

        # Chargement du modèle
        PATH = "/root/gelsight_ros2/ros2_vbtsensors/src/gs_mini_feature_extraction/gs_feature_extraction/"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        n_epoch = 49
        
        
        self.model = UnetPlusPlus("resnet18", "imagenet", in_channels=3, out_channels=1).to(self.DEVICE)
        load_checkpoint(torch.load(PATH + f"epochs/checkpoint_epoch_{n_epoch}.pth.tar"), self.model)
        self.test_transform = A.Compose([
	    A.PadIfNeeded(min_height=256, min_width=320, border_mode=0, value=0),  # padding a tamaño mínimo
	    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
	    ToTensorV2(),
	])

        self.get_logger().info(f"Network model loaded on {self.DEVICE}")

        # Timer pour déclencher l'analyse périodiquement
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30 Hz

    def timer_callback(self):
        self.flag = True

    def callback(self, msg):
        if not self.flag:
            return
        self.flag = False

        # Traitement de l'image
        #image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = self.test_transform(image=image)['image']
        image = image.to(self.DEVICE).unsqueeze(0)

        pred = torch.sigmoid(self.model(image))
        pred = (pred > 0.4).float()

        detection = pred.cpu().detach().numpy()
        final_mask = detection[0][0].astype(np.uint8) * 255




        # Publication du masque
        image_msg = self.bridge.cv2_to_imgmsg(final_mask, "mono8")
        image_msg.header.stamp = msg.header.stamp
        image_msg.header.frame_id = "os_sensor"
        self.detection_img_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PanelLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
