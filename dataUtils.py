import os
import glob
import cv2
import random
from PIL import Image
from tqdm import tqdm




def save_to_frame(input_dir, save_dir):
    """
    input:
        input_dir: directory to the video files
        save_dir: directory to save the frames converted from the video
    output:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    for path in tqdm(glob.glob(input_dir + '/*')):
        fname = os.path.basename(path).split('.')[0]
        os.makedirs(os.path.join(save_dir, fname), exist_ok=True)
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 1 == 0:
                #print("count = {}".format(count))
                cv2.imwrite("{}/{}/{}-{}.jpg".format(save_dir, fname, fname, str(count).zfill(4)), image)     # save frame as JPEG file      
            success, image = vidcap.read()
            count += 1



def gen_list(input_dir, merge_size):
    merge_list = []
    img_list = []
    for vidPath in tqdm(sorted(glob.glob(input_dir + '/*'))):
        img_list.append(vidPath)
        if len(img_list) == merge_size:
            merge_list.append(img_list)
            img_list = []
    return merge_list


def merge_frames(input_dir, save_dir, merge_img_dim):
    """
    input:
        input_dir: directory to the video frames created from save_to_frame function
        save_dir: directory to save the merged frames
        merge_img_dim: list of the size of the merged image (i.e. [5,3] or [3,2])
    output:
        None

    """
    r = merge_img_dim[0]
    c = merge_img_dim[1]
    merge_size = r * c

    os.makedirs(save_dir, exist_ok=True)
    for folderPath in tqdm(glob.glob(input_dir + '/*')):
        folderName = os.path.basename(folderPath).split('.')[0]
        merge_list = gen_list(folderPath, merge_size) 

        img = Image.open(merge_list[0][0])
        (w, h) = img.size
        result_w = w * c
        result_h = h * r
        result_img = Image.new('RGB', (result_w, result_h))

        for count, each in enumerate(merge_list):
            idx = 0
            for i in range(r):
                for j in range(c):
                    print(each[idx])
                    image = Image.open(each[idx])
                    result_img.paste(im=image, box=(j*w, i*h))
                    idx += 1

            merge_img_name = os.path.join(save_dir, "{}_merge_{}.jpg".format(folderName, count))
            result_img.save(merge_img_name)
            