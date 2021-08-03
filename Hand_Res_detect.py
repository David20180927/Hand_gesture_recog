import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import cv2 as cv
import keras

from keras.models import model_from_json
#315, 94, 37
#skin color
hmin, smin, vmin = 0, 58, 50
Hmax, Smax, Vmax = 30, 255, 255
lower_hsv = np.array([hmin, smin, vmin])
upper_hsv = np.array([Hmax, Smax, Vmax])
dict = {0: 'palm', 1: '1', 2: 'fist',3: 'fist_moved',4: 'thumb',5: 'index',6: 'ok',7: 'palm_moved',8: 'c',9: 'down'}

def preprocess(action_frame):
    blur = cv.GaussianBlur(action_frame, (3,3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lower_color, upper_color)
    blur = cv.medianBlur(mask, 5)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv.dilate(blur, kernel)
    return hsv_d

#dir of database
dirmodel = 'C:/Users/user/Desktop/Research/hand_ges_database/model/'
json_file = open(dirmodel+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(dirmodel+"model.h5")
print("Loaded model from disk")

cap = cv.VideoCapture(0)

while(1):
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    edge = cv.inRange(hsv, lower_hsv, upper_hsv)
    frame = preprocess(frame)
    arr = np.array(frame)
    img = Image.fromarray(arr).convert('L')
    img = img.resize((320, 120))
    arr = np.array(img)
    x = [arr]
    if x is not None:
        x = np.array(x, dtype = 'float32')            
        x = x.reshape((1, 120, 320, 1))
        x /= 255
        p = loaded_model.predict_classes(x)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame,'ges detected: ' + dict[int(p)],(20,20), font, 0.6,(255,255,255),2,cv.LINE_AA)  
    else:
        cv.putText(frame,'ges detected: ' + 'None',(20,20), font, 0.6,(255,255,255),2,cv.LINE_AA) 
    cv.imshow('frame',frame)
    cv.imshow('edge',edge)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv.destroyAllWindows() 



   


