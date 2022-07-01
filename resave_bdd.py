# -*- coding: utf-8 -*-
# @Time : 2020/8/11 下午5:47
# @Author :libin
# @File : jpg2bgr.py
# @Software: PyCharm
import cv2
from numpy import *
import numpy as np
import glob
import os
from tqdm import tqdm

from utils.general import imread, imwrite

class JPG2BGR(object):
    def __init__(self, basedir, test_path, save_path='/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/images_bgr/', image_size=640):
        self.img_size = image_size  # save bgr size
 
        self.imgpath = basedir
        self.path = test_path
        self.save_path = save_path

    # def yolo_letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    #     # Resize and pad image while meeting stride-multiple constraints
    #     shape = im.shape[:2]  # current shape [height, width]
    #     if isinstance(new_shape, int):
    #         new_shape = (new_shape, new_shape)

    #     # Scale ratio (new / old)
    #     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    #     if not scaleup:  # only scale down, do not scale up (for better val mAP)
    #         r = min(r, 1.0)

    #     # Compute padding
    #     ratio = r, r  # width, height ratios
    #     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    #     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    #     if auto:  # minimum rectangle
    #         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    #     elif scaleFill:  # stretch
    #         dw, dh = 0.0, 0.0
    #         new_unpad = (new_shape[1], new_shape[0])
    #         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    #     dw /= 2  # divide padding into 2 sides
    #     dh /= 2

    #     if shape[::-1] != new_unpad:  # resize
    #         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    #     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    #     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    #     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    #     cv2.imwrite("test.jpg", im)
    #     return im, ratio, (dw, dh)
    
 
    def letterbox(self,image):
        """
        letterbox
        :param image: source box
        :return:
        """
        ih = iw = self.img_size
        h, w, _ = image.shape
 
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0,dtype=np.uint8)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
 
        return image_paded
 
    def reverseletterbox(self,image,target_shape):
        """
        :param image:
        :param target_shape: target image shape(h,w,c)
        :return:
        """
        ih,iw,_=image.shape
        h, w, _ = target_shape
        scale=min(ih/h,iw/w)
 
        n_h,n_w=int(h*scale),int(w*scale)
        dh,dw=(ih-n_h)//2,(iw-n_w)//2
        new_img=image[dh:dh+n_h,dw:dw+n_w,:]
        targetimg=cv2.resize(new_img, (w, h))
        return targetimg
 
 
    def jpg2bgr(self):
        save_img_size = self.img_size
        imgpath = self.imgpath
        imgfile = glob.glob(os.path.join(imgpath,"*.jpg"))
        for jpg in tqdm(imgfile):
            img = cv2.imread(jpg)
            n_img = self.letterbox(img)
 
            if n_img is None:
                print("img is none")
            else:
 
                # cv2.imwrite(jpg, n_img)
                (B, G, R) = cv2.split(n_img)
 
                savepath = self.save_path + os.path.basename(jpg)[:-3] + 'bgr'
                with open(savepath, 'wb')as fp:
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(B[i, j])
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(G[i, j])
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(R[i, j])
 
                # print("save success")
 
 
    def transform(self,test=False):
        if not test:
            self.jpg2bgr()
        else:
            self.bgr2rgb()
 
 
    def bgr2rgb(self,shape=(720, 1280, 3)):
 
        path = self.path
        imgsize = self.img_size
 
        f = open(path, 'rb')
 
        src=np.zeros(shape, np.uint8)
 
        src = cv2.resize(src, (imgsize, imgsize))
 
        print(src.shape)
 
        B, G, R = cv2.split(src)
 
        data = f.read(imgsize * imgsize * 3)
        for j in range(imgsize):
            for i in range(imgsize):
                R[j, i] = data[j * imgsize + i]
                G[j, i] = data[j * imgsize + i + imgsize * imgsize]
                B[j, i] = data[j * imgsize + i + imgsize * imgsize * 2]
 
 
        newimg = cv2.merge([R,G,B])
        newimg=self.reverseletterbox(newimg,shape)
        cv2.waitKey(0)
        cv2.imwrite('./Test.jpg',newimg)
 
        f.close()
        cv2.waitKey(0)
 
 
if __name__ == '__main__':
    preprocess=JPG2BGR("/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/images", "/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/images_bgr/1.bgr")

    # img = cv2.imread("/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/images/1.jpg")
    # preprocess.yolo_letterbox(img)
    # preprocess.letterbox(img)

    preprocess.transform(test=False)