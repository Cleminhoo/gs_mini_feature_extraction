import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
import math

class ImageSubscriberPublisher(Node):
    def __init__(self):
        super().__init__('image_subscriber_publisher')

        self.image_h = 240
        self.image_w = 320

        self.subscription = self.create_subscription(
            Image, '/gs_depth_image', self.image_callback, 10
        )
        self.publisher = self.create_publisher(Image, '/gs_new_img', 10)
        self.coord_publisher = self.create_publisher(Float32MultiArray, '/gs_feature_coords2', 10)

        self.bridge = CvBridge()
        self.get_logger().info('Node initialized: Subscribed to /gs_depth_image and publishing to /gs_new_img')

    def publish_feature_coords(self, x_center, y_center, point1, point2, alpha, depth_data_p1,depth_data_p2, r):
        values = [
            float(x_center), float(y_center),
            float(point1[0]), float(point1[1]),
            float(point2[0]), float(point2[1]),
            float(alpha), float(depth_data_p1),float(depth_data_p2),
            float(r)
        ]

        if any(not math.isfinite(v) for v in values):
            return

        msg = Float32MultiArray()
        msg.data = values
        self.coord_publisher.publish(msg)

    def image_callback(self, msg):
        self.get_logger().info(f"Image encoding: {msg.encoding}")
        cv_image = self.bridge.imgmsg_to_cv2(msg)

        if msg.encoding == 'mono16':
            cv_image_8 = cv2.convertScaleAbs(cv_image, alpha=(255 / 65536))
        else:
            cv_image_8 = cv_image.copy()

        _, binary_thresh = cv2.threshold(cv_image_8, 20, 255, cv2.THRESH_BINARY)

       

        thinned = cv2.ximgproc.thinning(binary_thresh)
        edges = cv2.Canny(thinned, 100, 120)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_thresh, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = cv2.cvtColor(cv_image_8, cv2.COLOR_GRAY2BGR)
        h, w = cv_image.shape[:2]
        #cx, cy = w // 2, h // 2
        #cv2.rectangle(filtered, (cx - 50, cy - 50), (cx + 50, cy + 50), (0, 0, 255), 1)

        min_area = 500
        x_center, y_center = 0, 0
        point1, point2 = (0, 0), (0, 0)
        alpha = 0

        for cnt in contours_thresh:
            if cv2.contourArea(cnt) >= min_area:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (x_center, y_center), (a_len, b_len), angle = ellipse
                    cv2.ellipse(filtered, ellipse, (0, 255, 0), 1)

                    # Asegurarse que a_len sea el eje mayor
                    if a_len < b_len:
                        a_len, b_len = b_len, a_len
                        angle += 90.0
                    # Semieje mayor
                    a = a_len / 2.0

                    # Ángulo en radianes
                    alpha = math.radians(angle)



                    ################ Desplazamiento de los puntos de interseccion de la elipse y el eje mayor para poder sacar un dato mejor de profundidad:
                    # Factor de desplazamiento (0.0 = extremo exacto, 1.0 = centro)
                    f = 0.2

                    # Punots de interseccion desplazados
                    dx_new = a * (1 - f) * math.cos(alpha)
                    dy_new = a * (1 - f) * math.sin(alpha)

                    # Puntos desplazados hacia el centro
                    point1 = (int(x_center + dx_new), int(y_center + dy_new))
                    point2 = (int(x_center - dx_new), int(y_center - dy_new))


                    # # Dirección del eje mayor
                    # dx = a * math.cos(alpha)
                    # dy = a * math.sin(alpha)

                    # # Puntos extremos del eje mayor
                    # point1 = (int(x_center + dx), int(y_center + dy))
                    # point2 = (int(x_center - dx), int(y_center - dy))
                    
                    cv2.circle(filtered, (point1[0], point1[1]), 2, (255, 255, 255), -1)
                    cv2.circle(filtered, (point2[0], point2[1]), 2, (255, 0, 255), -1)
                    #cv2.circle(filtered, (x_center, y_center), 2, (0, 255, 255), -1)


                except cv2.error:
                    pass

                #limitacion de los puntos 1 y 2
                point1 = (min(max(point1[1], 0), self.image_h-1),min(max(point1[0], 0), self.image_w-1))
                point2 = (min(max(point2[1], 0), self.image_h-1),min(max(point2[0], 0), self.image_w-1))

                print("Punto_1x:",  point1[0] )
                print("Punto_1y:",  point1[1] )
                print("Punto_2x:",  point2[0] )
                print("Punto_2y:",  point2[1] )


                if 0 <= y_center < h and 0 <= x_center < w:
                    depth_data_p1 = float(cv_image_8[point1[0],point1[1]])
                    depth_data_p2 = float(cv_image_8[point2[0],point2[1]])
                    print("D_p1: ",depth_data_p1)
                    print("D_p2: ",depth_data_p2)
                    #depth_data = 0.0
                else:
                    self.get_logger().warn("Center out of image bounds.")
                    depth_data_p1 = 0.0
                    depth_data_p2 = 0.0

                r = math.sqrt((x_center-160)**2+(y_center-140)**2)#calcul de la distance pour effectuer des comparaisons avec les autres modèles
                print(r)

                self.publish_feature_coords(x_center, y_center, point1, point2, math.pi-alpha, depth_data_p1,depth_data_p2,r)

        filtered_msg = self.bridge.cv2_to_imgmsg(filtered, encoding='bgr8')
        self.publisher.publish(filtered_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriberPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
