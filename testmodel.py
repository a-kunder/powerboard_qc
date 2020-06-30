#!/usr/bin/env python
#include <iostream.h>

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os

#
parser = argparse.ArgumentParser(description='Train and test a model for identifying IDs')
parser.add_argument('trainDir', help='Directory with train dataset')
parser.add_argument('testDir', help='Directory with test dataset')

args=parser.parse_args()

# Load and label the dataset
def get_label(file_path):
    file_name=tf.strings.split(file_path, os.path.sep)[-1]
    digit=tf.strings.substr(file_name, 0, 1)
    return tf.strings.to_number(digit)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (100, 100))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img/255

    return img, label

train_ds=tf.data.Dataset.list_files('{}/*.png'.format(args.trainDir))
test_ds =tf.data.Dataset.list_files('{}/*.png'.format(args.testDir ))
IMG_HEIGHT = 374
IMG_WIDTH = 650

labelled_train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
labelled_test_ds  = test_ds .map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(type(labelled_train_ds))

# Prepare data for training
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        if n>=image_batch.shape[0]:
            continue
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n][:,:,0])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

training_size = 15000

batch_train_ds = labelled_train_ds.shuffle(1000).take(training_size).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
batch_test_ds  = labelled_test_ds .take(100).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

image_batch, label_batch = next(iter(batch_train_ds))
show_batch(image_batch.numpy(), label_batch.numpy())

# Make model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

#tensorboard plots
'''log_dir = f"logs/fit_20_updated/{training_size}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
histogram_freq=1)'''

print (f"Training size = {training_size}\n")

#Train
checkpointfile='checkpoint0'
if os.path.exists(checkpointfile+'.index'):
    model.load_weights(checkpointfile)
model.fit     (batch_train_ds, epochs=10)#, callbacks=[tensorboard_callback])
model.save_weights(checkpointfile)

#Test
model.evaluate(batch_test_ds , verbose=2)

image_batch, label_batch = next(iter(batch_test_ds))
predict_batch=np.argmax(model.predict(image_batch),axis=1)
show_batch(image_batch, predict_batch)

