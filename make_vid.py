import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

target_dir = 'test_data'

# find images in the color folder
img_paths_color = glob.glob(os.path.join(target_dir, 'rgb', '**', '*.png'), recursive=True)
img_paths_color.sort()  

# find images in the depth folder
img_paths_depth = glob.glob(os.path.join(target_dir, 'depth', '**', '*.png'), recursive=True)
img_paths_depth.sort()

# find images in the ir folder
img_paths_ir = glob.glob(os.path.join(target_dir, 'ir', '**', '*.png'), recursive=True)
img_paths_ir.sort()

# define a function to make a video from images in the folder
def make_video(img_folder, video_name):
    
    img_array = []
    for filename in tqdm(img_folder):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        print(height, width)
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release() 
    
#create new folder for videos
if not os.path.exists(os.path.join(target_dir, 'videos')):
    os.makedirs(os.path.join(target_dir, 'videos'))
    
# make videos from images in the folder
make_video(img_paths_color, os.path.join(target_dir, 'videos', 'color.avi'))
make_video(img_paths_depth, os.path.join(target_dir, 'videos', 'depth.avi'))
make_video(img_paths_ir, os.path.join(target_dir, 'videos', 'ir.avi'))
