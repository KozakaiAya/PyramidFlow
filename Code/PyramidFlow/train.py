import tensorflow as tf
import keras
import numpy as np
import cv2
import os

import model
import custom_loss
import myconfig
import utils

my_model = model.pyramid_model(input_shape=(myconfig.image_h, myconfig.image_w, 12))

optimizer = keras.optimizers.nadam(
    lr=myconfig.lr,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    schedule_decay=0.004)

loss = custom_loss.superloss_12

my_model.compile(loss=loss, optimizer=optimizer)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(myconfig.model_save_path, 'weights.hdf5'),
        monitor='loss',
        save_best_only=True,
        verbose=1),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(myconfig.model_save_path, 'weights_val_best.hdf5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss', verbose=1, factor=0.5, patience=5, min_delta=0.0005),
    keras.callbacks.TensorBoard(log_dir=myconfig.log_path),
    keras.callbacks.EarlyStopping(
        monitor='loss', patience=20, verbose=1, mode='auto'),
    keras.callbacks.CSVLogger(
        filename=os.path.join(myconfig.log_path, 'stats_per_epoch.csv'), append=False)
]

#keras.utils.plot_model(my_model, to_file=os.path.join(myconfig.model_save_path, 'model.png'), show_shapes=True)

train_data_count = len(utils.get_frame_tuple_list(myconfig.dataset_path))
val_data_count = len(utils.get_frame_tuple_list(myconfig.valset_path))
batch_size = 32

my_model.summary()


hist = my_model.fit_generator(
    generator=utils.data_generator(myconfig.dataset_path, batch_size=batch_size),
    steps_per_epoch=train_data_count // batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1,
    validation_data=utils.data_generator(myconfig.valset_path, batch_size=4),
    validation_steps=val_data_count // 4)

