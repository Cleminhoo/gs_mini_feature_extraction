#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import random
#from scipy.interpolate import splprep, splev
from scipy.spatial import distance


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



        #frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        #frame_bn = cv_image.copy()*0

        _,th1 = cv.threshold(frame,20,255,cv.THRESH_BINARY)
        # thresh_mean = cv2.adaptiveThreshold(cv_image_8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
        #                                     blockSize=11,  # taille du voisinage (doit être impair)
        #                                     C=2            # constante soustraite à la moyenne
        #                                     )

        frame = cv.bitwise_and(frame,frame,mask = th1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

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
            points = []
            for i in range(corners.shape[0]):
                x, y = corners[i, 0]
                points.append((x,y))
                color = (0,255, 0)
                cv.circle(frame_rgb, (int(x), int(y)), 2, color, -1)

            # Convertir en array numpy
            points = np.array(points)
            x_vals = points[:, 0]
            y_vals = points[:, 1]


            ############### Filtrado de datos:##############################

            # --- Filtrado estadístico multivariado (Mahalanobis) ---
           

            # Calculamos la media (centroide) y covarianza de los puntos
            mean = np.mean(points, axis=0)
            cov = np.cov(points.T)
            inv_cov = np.linalg.inv(cov)

            # Distancia de Mahalanobis de cada punto al centroide
            mahal_dist = np.array([
                distance.mahalanobis(p, mean, inv_cov) for p in points
            ])

            # Umbral para definir outliers (ajustable: cuanto mayor, más tolerante)
            threshold = np.mean(mahal_dist) + 2*np.std(mahal_dist)

            # Filtramos los inliers
            inliers = mahal_dist < threshold
            points_filtered = points[inliers]

            # Actualizamos los puntos
            points = points_filtered

            ###############################################################


            if len(points) >= 5:
                ellipse = cv.fitEllipse(points.astype(np.float32))
                center, axes, angle = ellipse
                x_c, y_c = center
                width, height = axes  # width: eje mayor, height: eje menor (según rotación)

                # Dibujo de la elipse
                cv.ellipse(frame_rgb, ellipse, (0, 255, 255), 1)

                # Dibujo del centro
                cv.circle(frame_rgb, (int(x_c), int(y_c)), 3, (0, 0, 255), -1)

                # Convertimos el ángulo a radianes
                theta = np.deg2rad(angle)

                # Eje mayor (la mitad porque queremos ir desde el centro hacia los extremos)
                dx_major = (width / 2) * np.cos(theta)
                dy_major = (width / 2) * np.sin(theta)
                pt1_major = (int(x_c - dx_major), int(y_c - dy_major))
                pt2_major = (int(x_c + dx_major), int(y_c + dy_major))
                cv.line(frame_rgb, pt1_major, pt2_major, (255, 0, 0), 2)

                # Eje menor
                dx_minor = (height / 2) * np.cos(theta + np.pi/2)
                dy_minor = (height / 2) * np.sin(theta + np.pi/2)
                pt1_minor = (int(x_c - dx_minor), int(y_c - dy_minor))
                pt2_minor = (int(x_c + dx_minor), int(y_c + dy_minor))
                cv.line(frame_rgb, pt1_minor, pt2_minor, (0, 255, 0), 2)


            # # Ajustement par une droite : y = m*x + b
            # if len(x_vals) >= 2:  # au moins 2 points requis
            #     m, b = np.polyfit(x_vals, y_vals, 1)

            #     # Calculer 2 points pour tracer la droite
            #     x0, x1 = int(np.min(x_vals)), int(np.max(x_vals))
            #     y0, y1 = int(m * x0 + b), int(m * x1 + b)

            #     # Tracer la droite sur l’image
            #     cv.line(frame_rgb, (x0, y0), (x1, y1), (0, 0, 255), 1)

        # Affichage pour debug
        cv.imshow("Corners", frame_rgb)
        cv.waitKey(1)

        # Publication ROS
        try:
            msg_out = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='passthrough')
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
            