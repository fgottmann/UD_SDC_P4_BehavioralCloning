### this file contains the model generation
### it was generated from the contents of the attached juypter notebook 'Model.ipynb.
### As i was not sure if i should put it all in one file for submission on udacity,
### you find here the data import, data preprocessing, model and training functions implemented


####################################
###### ---- Data Import ---- #######
####################################

# import neccessary libs
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import imageio


# function to get time difference
# assume measurements were conduct within one day
def time_diff(cur,base):
    diff  = (float(cur[-1]) - float(base[-1]))/1000.0
    diff += (float(cur[-2]) - float(base[-2]))*1.0
    diff += (float(cur[-3]) - float(base[-3]))*60.0
    diff += (float(cur[-4]) - float(base[-4]))*60.0*60.0
    if (cur[-5] != base[-5]): # not the same day
        diff += 24.0*60.0*60.0

    return diff

# initialize data
input_data = [] # initalize input data
data_dir   = 'data' # where are the datasets
run_names  = ['run1','run2'] # what are their names

# some preprocessing properties to minimize file size
pixels_cut_from_top = 65
pixels_cut_from_bot = 20
new_img_size        = (128,64)

# iterate trough the files
for run in run_names:
     # initialize data for specific run
    run_data = {'time':[],'im_center':[],'im_left':[],'im_right':[],'steer':[],'throttle':[],'brake':[],'speed':[]}
    timestamp_base = None # reset timestamp
    show_1_plt = 0        # reset plot function

    print("Importing run: '{}'".format(run))

    # read file
    with open(data_dir + '/' + run + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            # get current timestamp
            timestamp = line[0][:-4].split('_')[1:]
            if (timestamp_base is None):
                run_data['time'].append(0) # every dataset start at t=0
                timestamp_base = timestamp
            else:
                run_data['time'].append(time_diff(timestamp,timestamp_base))

            # import images
            img = imageio.imread(data_dir + '/' + run + '/' + line[0])
            img = img[pixels_cut_from_top:(img.shape[0] - 1 - pixels_cut_from_bot),:,:]
            img = cv2.resize(img, new_img_size, interpolation = cv2.INTER_AREA)
            run_data['im_center'].append(img)
            img = imageio.imread(data_dir + '/' + run + '/' + line[1])
            img = img[pixels_cut_from_top:(img.shape[0] - 1 - pixels_cut_from_bot),:,:]
            img = cv2.resize(img, new_img_size, interpolation = cv2.INTER_AREA)
            run_data['im_left'].append(img)
            img = imageio.imread(data_dir + '/' + run + '/' + line[2])
            img = img[pixels_cut_from_top:(img.shape[0] - 1 - pixels_cut_from_bot),:,:]
            img = cv2.resize(img, new_img_size, interpolation = cv2.INTER_AREA)
            run_data['im_right'].append(img)
            # import measurements
            run_data['steer'].append(float(line[3]))
            run_data['throttle'].append(float(line[4]))
            run_data['brake'].append(float(line[5]))
            run_data['speed'].append(float(line[6]))

            if show_1_plt == 0:
                fig,ar = plt.subplots(1,2,figsize=(10,10))
                ar[0].set_title('original image')
                ar[0].imshow(imageio.imread(data_dir + '/' + run + '/' + line[0]))
                ar[1].set_title('stored image')
                ar[1].imshow(run_data['im_center'][0])
                plt.show()
                show_1_plt = 1

        # append run to overall data
        print("Length of run ('{}') is {}.".format(run,len(run_data['speed'])))
        input_data.append(run_data)

# save input data
with open('input.p', 'wb') as f:
    pickle.dump(input_data, f)

print("Input data stored in input.p")

###########################################
###### ---- Data Preprocessing ---- #######
###########################################

# importing libs
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# importing data
with open('input.p','rb') as f:
    input_data = pickle.load(f)
print('Input data restored!')

# generate filter for measurements
fs = 15.0
ff = 0.5
bf = 2.0 * ff / fs
filt_steer_b, filt_steer_a = signal.butter(2, bf)


training_data = {'img':[],'steer':[],'throttle':[],'brake':[],'speed':[]}
discard_first_el = 20
discard_last_el = 20

for run in input_data:
    # center
    training_data['img']      += run['im_center'][discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['steer'])
    training_data['steer']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['throttle'])
    training_data['throttle'] += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['brake'])
    training_data['brake']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['speed'])
    training_data['speed']    += data.tolist()[discard_first_el:-discard_last_el]
    # left
    training_data['img']      += run['im_left'][discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['steer'])  + 0.3
    training_data['steer']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['throttle'])
    training_data['throttle'] += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['brake'])
    training_data['brake']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['speed'])
    training_data['speed']    += data.tolist()[discard_first_el:-discard_last_el]
    # right
    training_data['img']      += run['im_right'][discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['steer']) - 0.3
    training_data['steer']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['throttle'])
    training_data['throttle'] += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['brake'])
    training_data['brake']    += data.tolist()[discard_first_el:-discard_last_el]
    data = signal.filtfilt(filt_steer_b,filt_steer_a,run['speed'])
    training_data['speed']    += data.tolist()[discard_first_el:-discard_last_el]



training_data['img']      = np.array(training_data['img'])
training_data['steer']    = np.array(training_data['steer'])
training_data['throttle'] = np.array(training_data['throttle'])
training_data['brake']    = np.array(training_data['brake'])
training_data['speed']    = np.array(training_data['speed'])

plt.plot(training_data['steer'])
plt.show()

print('Data for training generated!')
print()

# save train data
with open('train.p', 'wb') as f:
    pickle.dump(training_data, f)

print("Training data stored in train.p")

########################################
##### ---- Model Architecture ---- #####
########################################
##### used model from nvidia's autopilot project: https://github.com/0bserver07/Nvidia-Autopilot-Keras
########################################

import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

import cv2
import numpy as np
import json

# from Dan Does Data VLOG
import math
import h5py
import glob
from tqdm import tqdm
import scipy
from scipy import misc

import matplotlib.pyplot as plt
plt.ion()


# frame size
nrows = 64
ncols = 128

# model start here
model = Sequential()

model.add(BatchNormalization(epsilon=0.001, axis=3,input_shape=(nrows,ncols,3)))

model.add(Convolution2D(24,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(36,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(48,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Convolution2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


model.summary()
# Save model to JSON
with open('autopilot_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

print("Driver Model stored!")

########################################
##### ---- Training the model ---- #####
########################################
## Parameters reused from NVIDIA's approach
# importing files
########################################

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import keras
import keras.models as models
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

# importing data
with open('train.p','rb') as f:
    training_data = pickle.load(f)
print('Training data restored!')

model = Sequential()
with open('autopilot_basic_model.json') as model_file:
    model = models.model_from_json(model_file.read())


# checkpoint
#filepath="weights/weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])

nb_epoch = 10
batch_size = 64

print(training_data['steer'][:].shape)
print(training_data['img'].shape)

model.fit(training_data['img'], training_data['steer'], #callbacks=callbacks_list,
          batch_size =batch_size, nb_epoch=nb_epoch, verbose=1,
          validation_split=0.25,shuffle=True)

# export weights and model
model.save_weights('weights/model_weights.hdf5')
model.save('model.h5')
