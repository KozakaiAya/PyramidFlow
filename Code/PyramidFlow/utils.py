import os
import cv2
import random
import numpy as np

import myconfig

def get_frame_tuple_list(path):
    frame_tuple_list = []
    with open(os.path.join(path, 'frame_list.txt'), 'r') as f:
        name = f.readlines()
        frame_tuple_list = [x.strip() for x in name]

    return frame_tuple_list

def resize(img):
    return cv2.resize(img, (myconfig.image_w, myconfig.image_h), interpolation=cv2.INTER_LANCZOS4)

def data_generator(data_path, batch_size=32):
    frame_list = get_frame_tuple_list(data_path)
    frame_count = len(frame_list)
    batch_count = frame_count // batch_size
    while True:
        random.shuffle(frame_list)
        batch_data_list = []
        batch_target_list = []
        for idx in range(batch_count * batch_size):
            img1_name = os.path.join(data_path, frame_list[idx] + '_0.png')
            img2_name = os.path.join(data_path, frame_list[idx] + '_1.png')
            img3_name = os.path.join(data_path, frame_list[idx] + '_2.png')
            img4_name = os.path.join(data_path, frame_list[idx] + '_3.png')
            img5_name = os.path.join(data_path, frame_list[idx] + '_4.png')

            img1 = resize(cv2.imread(img1_name))
            img2 = resize(cv2.imread(img2_name))
            img3 = resize(cv2.imread(img3_name))
            img4 = resize(cv2.imread(img4_name))
            img5 = resize(cv2.imread(img5_name))

            data = np.concatenate((img1, img2, img4, img5), axis=2)
            batch_data_list.append(data)
            batch_target_list.append(img3)

            if len(batch_data_list) == batch_size:
                batch_data = np.stack(batch_data_list, axis=0)
                batch_target = np.stack(batch_target_list, axis=0)

                batch_data_list = []
                batch_target_list = []

                yield batch_data.astype('float32') / 255.0, batch_target.astype('float32') / 255.0


            