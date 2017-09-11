import csv
import cv2
import numpy as np
import math
from image_processing import jitter, crop_scaleImage
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation,Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


lines = []
with open ('../data/driving2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

init_measure = []
for line in lines:
    measurement = float(line[3])
    init_measure.append(measurement)

init_measure = np.array(init_measure)

num_bins = 25
avg_samples_per_bin = len(init_measure)/num_bins 
hist, bins = np.histogram(init_measure, num_bins)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, color = 'red')
#plt.plot((np.min(init_measure), np.max(init_measure)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

keep_probs = []
target = avg_samples_per_bin * 0.5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))

remove_list = []
for i in range(len(init_measure)):
    for j in range(num_bins):
        if init_measure[i] > bins[j] and init_measure[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)

#image_paths = np.delete(image_paths, remove_list, axis=0)
#init_measure = np.delete(init_measure, remove_list)

for i in sorted(remove_list, reverse=True): 
    del lines[i]


train_lines, validation_lines = train_test_split(lines, test_size=0.2)

init_measure = []
for line in lines:
    measurement = float(line[3])
    init_measure.append(measurement)

init_measure = np.array(init_measure)

num_bins = 25
avg_samples_per_bin = len(init_measure)/num_bins


width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
hist, bins = np.histogram(init_measure, num_bins)
plt.bar(center, hist, align='center', width=width, color='red')
#plt.plot((np.min(init_measure), np.max(init_measure)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

plt.show()

correction = 0.25
steering_c = 0.0
steering_l = steering_c+correction
steering_r = steering_c-correction

steering = (steering_c,steering_l,steering_r)

new_col = 200
new_row = 66


a_images, a_measurements = [],[]
for line in train_lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/driving2/IMG/' + filename
        image = cv2.imread(current_path)
        image_t = crop_scaleImage(image,new_col,new_row)
        image_t = cv2.cvtColor(image_t,cv2.COLOR_BGR2RGB)
        a_images.append(image_t)
        #print(line[3], source_path)
        measurement = float(line[3])
        measurement = measurement+steering[i]
        a_measurements.append(measurement)
        
        timg,tmea = jitter(image_t,measurement)
        #timg = crop_scaleImage(image,new_col,new_row)

        a_images.append(timg)
        a_measurements.append(tmea)
                
        if abs(measurement) > 0.0:
#        if i==0 :
            
            #a_images.append(crop_scaleImage(cv2.flip(image,1),new_col,new_row))
            a_images.append(cv2.flip(image_t,1))
            a_measurements.append(measurement*-1.0)
            
            timg,tmea = jitter(cv2.flip(image_t,1),measurement*-1.0)
            #timg = crop_scaleImage(image,new_col,new_row)

            a_images.append(timg)
            a_measurements.append(tmea)



x_train = np.array(a_images)
y_train = np.array(a_measurements)

init_measure = []
for line in y_train:
    measurement = line
    init_measure.append(measurement)

init_measure = np.array(init_measure)

num_bins = 25
avg_samples_per_bin = len(init_measure)/num_bins
hist, bins = np.histogram(init_measure, num_bins)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, color='red')
#plt.plot((np.min(init_measure), np.max(init_measure)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

a_images, a_measurements = [],[]

for line in validation_lines:

    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = '../data/driving2/IMG/' + filename
    image = cv2.imread(current_path)
    image = crop_scaleImage(image,new_col,new_row)
    a_images.append(image)
    measurement = float(line[3])
    a_measurements.append(measurement)
    
    
x_val = np.array(a_images)
y_val = np.array(a_measurements)

x_val = x_val.reshape(1,x_val.shape[0],x_val.shape[1],x_val.shape[2],x_val.shape[3])
y_val = y_val.reshape(1,y_val.shape[0])
x_val = tuple(x_val)
y_val = tuple(y_val)
valid = x_val+y_val

from keras.layers.core import Dropout
model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(new_row,new_col,3)))
#model.add(Cropping2D(cropping=((30,0),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu",W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu",W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu",W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="elu",W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="elu",W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,W_regularizer=l2(0.001)))

model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(50,W_regularizer=l2(0.001)))

model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(10,W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = Adam(lr=1e-4))

model.fit(x_train,y_train,validation_data=valid, shuffle=True, verbose=1,nb_epoch=20)

model.save('model.h5')
