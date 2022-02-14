import cv2
import os
from matplotlib import image
import numpy as np
from PIL import Image

def convertColor(image_path, out_path, str_index) :
    depth_image = cv2.imread(image_path)
    image_color=cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=15),cv2.COLORMAP_JET)
    out_image = Image.fromarray(image_color)
    out_image.save(out_path +'/img_' + str_index + '.png')

if __name__ == '__main__' :
    test_path = './test_result/'
    out_path = 'color_out'
    ref_path = 'ref_out'
    if not os.path.exists(out_path) :
        os.mkdir(out_path)
    if not os.path.exists(ref_path) :
        os.mkdir(ref_path)

    for index in range(len(os.listdir(test_path))) :
        str_index = '0'
        for i in range(5 - len(str(index))) :
            str_index += '0'
        str_index += str(index) #00xxxx
        rgb_image_dir = './test_result/img_{}.png'.format(str_index)
        depth_map = './nyuv2/test/{}_depth.png'.format(str_index)

        convertColor(rgb_image_dir, out_path, str_index)
        convertColor(depth_map, ref_path, str_index)
