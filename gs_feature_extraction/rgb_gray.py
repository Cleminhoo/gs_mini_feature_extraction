import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubscriberPublisher(Node):
    def __init__(self):
        super().__init__('image_subscriber_publisher')

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/gs_mini_img',
            self.image_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(
            Image,
            '/gs_gray_img',
            10
        )

        # CV Bridge
        self.bridge = CvBridge()
        self.get_logger().info('Node initialized: Subscribed to /gs_mini_img and publishing to /gs_gray_img')

    def image_callback(self, msg):
        try:
            self.get_logger().info(f"Image encoding: {msg.encoding}")

            # Décodage de l'image ROS
            if msg.encoding == 'bgr8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg)

            # 1. Conversion en niveaux de gris
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 2. Amélioration du contraste (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_image)

            # 3. Réduction du bruit
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.2)

            # 4. Détection de contours avec Canny (ajusté pour plus de sensibilité)
            edges = cv2.Canny(blurred, threshold1=50, threshold2=120)
            kernel = np.ones((3, 3), np.uint8)
            dilation = cv2.dilate(edges,kernel,iterations = 1)

             # 6. Filtrage par aires
            contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)  # pour dessiner en couleur

            min_area = 10  # ← seuil ajustable selon ton capteur
            for cnt in contours:
                area = cv2.contourArea(cnt)
                print(area)
                if area >= min_area:
                    cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                else:
                    cv2.drawContours(cv_image, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

                

            # 7. Conversion finale en niveaux de gris
            final_edges = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

            # 5. Affichage
#            cv2.imshow('Contours améliorés', edges)
#            cv2.waitKey(1)

            # 6. Publication
            filtered_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher.publish(filtered_msg)

        except Exception as e:
            
            self.get_logger().error(f"Erreur dans image_callback: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriberPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
