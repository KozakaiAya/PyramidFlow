import cv2
import numpy as np
import os
import threading
import queue
import skimage.measure

video_clip_path = '../../Video Clips'
dataset_save_path = '../../Dataset'
frame_tuple = 5
same_scene_max = 3
ssim_min = 0.73
ssim_max = 0.995

class ownerLock():
    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None

    def acquire(self, thread, blocking=True):
        self.owner = thread
        return self.lock.acquire(blocking)

    def release(self):
        self.owner = None
        return self.lock.release()

    def get_owner(self):
        return self.owner


def is_image_similar(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim = skimage.measure.compare_ssim(img1, img2)
    return (ssim_min <= ssim <= ssim_max)

def is_same_scene(frame_list):
    ret = True
    for i in range(1, frame_tuple - 1):
        if not (is_image_similar(frame_list[i - 1], frame_list[i]) and is_image_similar(frame_list[i], frame_list[i + 1])):
            ret = False
            break

    return ret 

tuple_id = 0
video_clip_queue = queue.Queue()
tuple_id_mutex = ownerLock()     

def process_video(clip):
    global tuple_id
    print("Start to process:", clip.encode('utf-8'))
    try:
        vid = cv2.VideoCapture(clip)
        frame_list = []
        last_scene = 0
        same_scene_count = 0
        has_last_scene = False
        while True:
            ret, image = vid.read()
            if not ret:
                break
            image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
            frame_list.append(image)
            if len(frame_list) == frame_tuple:
                # Check whether reached same_scene_max
                we_should_test_it = True
                if has_last_scene:
                    if is_image_similar(frame_list[0], last_scene):
                        same_scene_count += 1
                        last_scene = frame_list[0]
                        if same_scene_count > same_scene_max:
                            we_should_test_it = False
                    else:
                        has_last_scene = True
                        last_scene = frame_list[0]
                        same_scene_count = 1
                else:
                    has_last_scene = True
                    last_scene = frame_list[0]
                    same_scene_count = 1

                if we_should_test_it:
                    if is_same_scene(frame_list):
                        tuple_id_mutex.acquire(threading.current_thread())
                        tuple_id += 1
                        #raise ValueError
                        print(tuple_id)
                        tuple_id_mutex.release()
                        idx = 0
                        for img in frame_list:
                            img_name = str(tuple_id).zfill(7) + '_' + str(idx) + '.png'
                            idx += 1
                            cv2.imwrite(os.path.join(dataset_save_path, img_name), img)
                
                frame_list = []

    except Exception as e:
        print(e)
        if threading.current_thread() is tuple_id_mutex.get_owner():
            print("Mutex Release")
            tuple_id_mutex.release()

def start_process_thread():
    while not video_clip_queue.empty():
        video_path = video_clip_queue.get()
        process_video(video_path)
        if tuple_id > 20000:
            return

    return



video_name_list = os.listdir(video_clip_path)
for name in video_name_list:
    video_clip_queue.put(os.path.join(video_clip_path, name))

thread_list = []
for _ in range(7):
    thread = threading.Thread(target=start_process_thread)
    thread.start()
    thread_list.append(thread)

for th in thread_list:
    th.join()

with open(os.path.join(dataset_save_path, 'frame_list.txt'), 'w') as f:
    for i in range(1, tuple_id + 1):
        f.write(str(i).zfill(7) + '\n')



        