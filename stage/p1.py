import cv2

camera = cv2.VideoCapture(1) #On sélectionne la position de la caméra qu'on veut utiliser ici on prend le 1 qui correspond au capteur, tandis que le 0 correspond à la caméra de notre pc
img_counter = 0

while True:
    ret, frame = camera.read() #on récupére l'information lié à la caméra 
    frame = cv2.resize(frame,(320,240))
    print(frame.shape)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) #On va appliquer différents thresholds qui ont des propriétés différentes, par exemple pour th1 on fait passer l'image en noir et blanc en fonction de la valur des pixels(voir site)
    ret, th2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
    if not ret:
        print("failed to grab frame")
        break
    
    cv2.imshow("gelsight", frame) #On affiche les différentes images que l'on veut 
    cv2.imshow("gelsight1", th1)
    cv2.imshow("gelsight2", th2)
    cv2.imshow("gelsight3", th3)
    cv2.imshow("gelsight4", th4)
    cv2.imshow("gelsight5", th5)

    if cv2.waitKey(1) & 0xFF == ord('q'): #Lorsque l'on appuie sur la touche q on enlève les images  et cv2.waitkey(1) va montrer l'image pendant 1 ms
        break
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cv2.destroyAllWindows()


