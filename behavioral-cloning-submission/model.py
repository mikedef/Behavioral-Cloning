import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
tf.python.control_flow_ops = tf

#original is model2.py on GPU computer

# Model training variables
model_name = 'model33.h5'
epochs = 2
BATCH_SIZE = 64 # augment and process images in batches of 256 to keep memory use lower

# No generator

# Image augmentation variables
correction = 0.25 #0.15

samples = []
# Read in CSV
with open('/home/mikedef/disk/CarND-Term1/CarND-Behavioral-Cloning/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#print(samples[0])
#del(samples[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Helper functions
# flip images horizontally
#def flip_image(img, steering_angle):
#    flip = cv2.flip(img, 1)
#    steering_angle = steering_angle * -1)
#    return flip, steering_angle

# Generator
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                # Center Image
                center = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)#YUV) 
                center_angle = float(batch_sample[3])
                images.append(center)
                angles.append(center_angle)
                # Flip images and measurements
                images.append(cv2.flip(center,1))
                angles.append(center_angle*-1.0)

                # Left Image
                left = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)#YUV)
                left_angle = center_angle + correction
                images.append(left)
                angles.append(left_angle)
                # Flip images and measurements
                images.append(cv2.flip(left,1))
                angles.append(left_angle*-1.0)

                # Right Image
                right = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)#YUV)
                images.append(right)
                right_angle  = center_angle - correction
                angles.append(right_angle)
                # Flip images and measurements
                images.append(cv2.flip(right,1))
                angles.append(right_angle*-1.0)

            # Make data numpy arrays to feed into Keras
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=BATCH_SIZE)
#validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


# Since images were taken on local machine grab the path name minus local machine name
images = []
measurements = []
del samples[0]
for line in samples:
# Center image
    source_path = line[0]
    #print('source path:', source_path)
    filename = source_path.split('/')[-1]
    #print('filename:', filename)
    current_path = 'data/IMG/' + filename
    #print('current path:', current_path)
    center = cv2.imread(current_path)
    center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
#    center = cv2.cvtColor(center, cv2.COLOR_BGR2YUV) # as per Nvidia pape#r
    images.append(center)
    # get stearing measurement
    steering_center = float(line[3])
    measurements.append(steering_center)   
    # Flip images and measurements
    images.append(cv2.flip(center,1))
    measurements.append(steering_center*-1.0)##

# Left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    left = cv2.imread(current_path)
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
#    left = cv2.cvtColor(left, cv2.COLOR_BGR2YUV)
    images.append(left)
    steering_left   = steering_center + correction
    measurements.append(steering_left)
    # Flip images and measurements
    images.append(cv2.flip(left,1))
    measurements.append(steering_left*-1.0)

# Right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    right = cv2.imread(current_path)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
#    right = cv2.cvtColor(right, cv2.COLOR_BGR2YUV)
    images.append(right)
    steering_right  = steering_center - correction
    measurements.append(steering_right)
    # Flip images and measurements
    images.append(cv2.flip(right,1))
    measurements.append(steering_right*-1.0)

# Make data numpy arrays to feed into Keras
X_train = np.array(images)
y_train = np.array(measurements)
#print(X_train.shape, y_train.shape)

# Create a basic NN in Keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocessing Layers
# Creates a layer that normalizes each image
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
# Crops the image such that the sky and hood of the vehicle are removed
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Nvidia Model
# 24, kernel_size(5,5), strides(2,2), activation layer (relu)
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(64,3,3, activation='elu'))
model.add(Convolution2D(64,3,3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

# MSE and not cross entropy because it's a regression network
model.compile(loss='mse', optimizer='adam')
# Fit the modelm
#history_object = model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)//BATCH_SIZE)*BATCH_SIZE, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

model.save(model_name)

print(model_name)
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
