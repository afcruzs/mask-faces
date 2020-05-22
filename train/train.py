#!/usr/bin/env python
# coding: utf-8

# do not bother with tf if args are not correct
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="data path", required=True)
parser.add_argument("--train_folder", type=str, help="local train folder to data_path", required=False, default='train')
parser.add_argument("--dev_folder", type=str, help="local dev folder to data_path", required=False, default='dev')
parser.add_argument("--test_folder", type=str, help="local test folder to data_path", required=False, default='test')
parser.add_argument("--epochs", type=int, help="epochs", required=True)
parser.add_argument("--batch_size", type=int, help="Batch size", required=False, default=64)
parser.add_argument('--local', action='store_true', help='Send data to wandb', required=False, default=False)
args = parser.parse_args()

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D,  Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn import simple_cnn
from wandb.keras import WandbCallback
from tqdm.auto import tqdm
import numpy as np
import time
import os
import random
import wandb

if __name__ == '__main__':
    params = dict(epochs = args.epochs, cnn1=32, cnn2=32, cnn3=64, cnn4=64, dropout=0.5, batch=args.batch_size)
    
    if not args.local:
        wandb.init(project="mask-faces", config=params, name='small_cnn')

    INPUT_IMG_SIZE = (160, 160) 
    model = simple_cnn(INPUT_IMG_SIZE)

    batch_size = params['batch']

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            os.path.join(args.data_path, args.train_folder),  # this is the target directory
            target_size=INPUT_IMG_SIZE,  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            os.path.join(args.data_path, args.dev_folder),
            target_size=INPUT_IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary')

    # this is a similar generator, for validation data
    test_generator = test_datagen.flow_from_directory(
            os.path.join(args.data_path, args.test_folder),
            target_size=INPUT_IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary')

    filepath = "small-cnn-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    n_train_samples = sum([len(files) for r, d, files in os.walk(os.path.join(args.data_path, args.train_folder))])
    n_dev_samples = sum([len(files) for r, d, files in os.walk(os.path.join(args.data_path, args.dev_folder))])
    train_steps = n_train_samples // batch_size
    validation_steps = n_dev_samples // batch_size
    print(f'{n_train_samples} in Train set')
    print(f'{n_dev_samples} in Train set')
    print(f'Optimization steps | Train: {train_steps}, Validation: {validation_steps}')

    callbacks=[checkpoint]
    if not args.local:
        callbacks.append(WandbCallback(data_type="image", labels=['mask', 'nomask'], generator=validation_generator))

    model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint, WandbCallback(data_type="image", labels=['mask', 'nomask'], generator=validation_generator)])