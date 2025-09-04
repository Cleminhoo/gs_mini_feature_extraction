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

import time
import cv2
import math

count = 0

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
        global count
      #  if not self.flag:
       #     return
      #  self.flag = False

        folder = "/root/gelsight_ros2/results_txt/"

        # Traitement de l'image
        #image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        h, w = image.shape[:2]

        start = time.time()
        image = self.test_transform(image=image)['image']
        image = image.to(self.DEVICE).unsqueeze(0)

        pred = torch.sigmoid(self.model(image))
        pred = (pred > 0.4).float()

        detection = pred.cpu().detach().numpy()
        final_mask = detection[0][0].astype(np.uint8) * 255
        contours_thresh, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colourmask= final_mask.copy()
        colourmask = cv2.cvtColor(colourmask, cv2.COLOR_GRAY2RGB)
        out_path = folder + "detect.txt"

        
        time_exec = 0
        flag_area = True
        r = 0
        alpha = 0
        cx = 0 
        cy = 0
        rightmost = [0, 0]
        leftmost = [0, 0]

        for cnt in contours_thresh:
            area = cv2.contourArea(cnt)
            if area >= 20:
                flag_area = False
                # x_values = cnt[:, :, 0]
                # y_values = cnt[:, :, 1]

                # min_x = x_values.min()
                # max_x = x_values.max()
                # min_y = y_values.min()
                # max_y = y_values.max()
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                
                alpha= math.pi/2 +math.atan2((leftmost[0]-rightmost[0]),(leftmost[1]-rightmost[1]))
            
                #coordonnées du centroïde 
                cx = int((leftmost[0]+rightmost[0])/2)
                cy = int((leftmost[1]+rightmost[1])/2)

                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

                lefty = int((-x * vy / vx) + y)
                righty = int(((final_mask.shape[1] - x) * vy / vx) + y)

                point1 = (final_mask.shape[1] - 1, righty)
                point2 = (0, lefty)
                cv2.line(colourmask, point1, point2, (255, 0, 0), 1)
                cv2.circle(colourmask, leftmost, 5, (0, 0, 255), -1)
                cv2.circle(colourmask, rightmost, 5, (0, 255, 0), -1)
                cv2.circle(colourmask, (cx,cy), 5, (0, 255, 0), -1)

                cv2.circle(colourmask, (int(w/2),int(h/2)), 5, (255,0,0), 1)
                cv2.line(colourmask,(int(w/2),int(h/2)),(cx,cy),(0,0,255 ),1)

                r=abs(math.sin(alpha)*(leftmost[0]+rightmost[0]-w)+math.cos(alpha)*(leftmost[1]+rightmost[1]-h))/2
                #print('r: ',r)
                #print('a:', alpha)

                #with out_path.open('a', encoding='utf-8') as f:
                time_exec = 1000*(time.time() - start)
                

            
            #else:
            #    with open(out_path,"a") as f:
             #       f.write(f" {0},{0},{0},{0},{0},{0},{0},{0},{0}\n")

        if not contours_thresh:            
            #print(out_path)
            #with out_path.open('a', encoding='utf-8') as f:
            with open(out_path,"a") as f:
                f.write(f" {0},{0},{0},{0},{0},{0},{0},{0},{0}\n")
        else:
            if flag_area:
                with open(out_path,"a") as f:
                    f.write(f" {0},{0},{0},{0},{0},{0},{0},{0},{0}\n")
            else:
                with open(out_path,"a") as f:
                #   for b, c in enumerate(i, start=0):
                    f.write(f"{r},{alpha},{cx},{cy},{rightmost[0]},{rightmost[1]},{leftmost[0]},{leftmost[1]},{time_exec}\n")
        # Publication du masque
        print( "Process time: " + str(time_exec))
        count=count+1
        print(count)
        image_msg = self.bridge.cv2_to_imgmsg(colourmask, "rgb8")
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
