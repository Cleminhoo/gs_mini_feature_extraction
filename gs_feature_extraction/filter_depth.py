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

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/gs_depth_image',
            self.image_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(
            Image,
            '/gs_new_img',
            10
        )

        self.coord_publisher = self.create_publisher(Float32MultiArray, '/gs_feature_coords', 10)

        # CV Bridge
        self.bridge = CvBridge()
        self.get_logger().info('Node initialized: Subscribed to /gs_depth_image and publishing to /gs_new_img')

    def publish_feature_coords(self, x_center, y_center, point1, point2,alpha , depth_data ):
        msg = Float32MultiArray()
        msg.data = [float(x_center), float(y_center), float(point1[0]), float(point1[1]), float(point2[0]), float(point2[1]),float(alpha), float(depth_data)]
        self.coord_publisher.publish(msg)


    def image_callback(self, msg):
        if(1):
            self.get_logger().info(f"Image encoding: {msg.encoding}")
            
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            if msg.encoding == 'mono16':
                cv_image_8 = cv2.convertScaleAbs(cv_image, alpha=(255/65356))#conversion d'une image de 16 bits à 8 bits
            else:
                cv_image_8 = cv_image.copy()

            

            _,th1 = cv2.threshold(cv_image_8,20,255,cv2.THRESH_BINARY)
            thresh_mean = cv2.adaptiveThreshold(cv_image_8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                blockSize=11,  # taille du voisinage (doit être impair)
                C=2            # constante soustraite à la moyenne
            )
            thresh_gauss = cv2.adaptiveThreshold(
                cv_image_8, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )

            




            # 3. Réduction du bruit
            blurred = cv2.GaussianBlur(cv_image_8, (5, 5), 1.2)
            # 4. Détection de contours avec Canny (ajusté pour plus de sensibilité)
            thinned = cv2.ximgproc.thinning(th1)
            #edges = cv2.Canny(blurred, threshold1=50, threshold2=120)
            edges = cv2.Canny(thinned, threshold1=50, threshold2=120)
            kernel = np.ones((3, 3), np.uint8)
            dilation = cv2.dilate(edges,kernel,iterations = 2)

             # 6. Filtrage par aires
            contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours obtenus à partir du th1 pour tracer la nouvelle ellipse
            filtered = cv2.cvtColor(cv_image_8, cv2.COLOR_GRAY2BGR)  # pour dessiner en couleur

            min_area = 1000  # ← seuil ajustable selon ton capteur
            x_center=0
            y_center=0
            point1=[0,0]
            point2=[0,0]
            alpha = 0

          #  for cnt2 in contours2:
          #      area1 = cv2.contourArea(cnt2)
          #      if area1 >= min_area:                    
          #          try:
          #              ellipse = cv2.fitEllipse(cnt2)
          #              x_center = int(ellipse[0][0])
          #              y_center = int(ellipse[0][1])
          #              cv2.ellipse(filtered,ellipse,(0,255,0),2)
           #         except:
           #             None
                


            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    cv2.drawContours(filtered, [cnt], -1, (0, 0, 255), 1)
                    M = cv2.moments(cnt)
                    print( M )
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    cv2.circle(filtered,(cx,cy),5,(0,255,255),1) 


            # Approximation d’une ligne sur le premier contour
                    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cv_image_8.shape[1] - x) * vy / vx) + y)

                    point1 = (cv_image_8.shape[1] - 1, righty)
                    point2 = (0, lefty)
                    #ellipse = cv2.fitEllipse(cnt)
                    #x_center = int(ellipse[0][0])
                    #y_center = int(ellipse[0][1])
                    alpha= math.atan2((point1[1]-point2[1]),(point1[0]-point2[0]))
                    print(alpha)

                    try:
                        cv2.line(filtered, point1, point2, (255, 0, 0), 2)
                        #cv2.ellipse(filtered,ellipse,(0,255,0),2)
                        cv2.circle(filtered,(x_center,y_center),2,(0,255,255),-1) # paramètres : image dans laquelle on trace le cercle, coordonnées du centre, rayon, couleur du cercle en BGR(ici en vert) et épaisseur du trait ici -1 cercle plein. 

                    except:
                        None

                #send depth information
                try:
                    depth_data = cv_image_8[y_center] [x_center] # extraire la coordonnée z de l'image 
                except:
                    depth_data = 0.0

                self.publish_feature_coords(x_center, y_center, point1, point2,alpha, depth_data)#Publication du message qui affiche les coordonnées du cercle,les deux extremités de la droite et l'angle alpha.

            
            # 7. Conversion finale en niveaux de gris
            final_edges = cv2.cvtColor(thinned, cv2.COLOR_GRAY2RGB)


            # 8. Publication
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
