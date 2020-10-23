import numpy as np
import cv2
import tensorflow as tf
model=tf.keras.models.load_model('mnist_model700.h5') 

img = np.zeros([250,250],dtype='uint8')
windowName='Canvas'
cv2.namedWindow(windowName)


def draw(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags==cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img,(x,y),8,(255,255,255),-1)

cv2.setMouseCallback(windowName,draw)

while(1):
    cv2.imshow(windowName,img)
    if cv2.waitKey(1)==ord('p'):
        image_resize=cv2.resize(img[:,:],(28,28)).reshape(1,28,28)
        output=model.predict_classes(image_resize)
        print("Digit identified:",output)
    elif cv2.waitKey(1)==ord('c'):
        img[:,:]=0
    elif cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
