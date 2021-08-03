import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


from keras import layers
from keras import models


#dir of database
dirdata = 'C:/Users/user/Desktop/Research/hand_ges_database/leapGestRecog/00'
dirmodel = 'C:/Users/user/Desktop/Research/hand_ges_database/model/'

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir(dirdata):
    if not j.startswith('.'): # If running this code locally, this is to 
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1


x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('C:/Users/user/Desktop/Research/hand_ges_database/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('C:/Users/user/Desktop/Research/hand_ges_database/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('C:/Users/user/Desktop/Research/hand_ges_database/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count

            
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size



#use keras
y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255


#

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))

[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))

# serialize model to JSON
model_json = model.to_json()
with open(dirmodel+ "model.json", "w") as json_file:
    json_file.write(model_json)

    
# serialize weights to HDF5
model.save_weights(dirmodel+"model.h5")
print("Saved model to disk")

##
img = Image.open('C:/Users/user/Desktop/Research/hand_ges_database/leapGestRecog/04/08_palm_moved/frame_04_08_0026.png').convert('L')
img = img.resize((320, 120))
arr = np.array(img)
x = [arr]
x = np.array(x, dtype = 'float32')
x = x.reshape((1, 120, 320, 1))
x /= 255

p = loaded_model.predict_classes(x)
