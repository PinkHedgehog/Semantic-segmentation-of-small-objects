import numpy as np
import os
import PIL
from PIL import Image, ImageColor, ImageEnhance, ImageFilter
import openslide
import cv2
import random
import multiprocessing as mp
import cythonmagic as cy
import matplotlib.pyplot as plt
import torch
import skimage
import Helpers.helpers as h
from skimage import io
from skimage import data
import skimage.color as col
import segmentation_models_pytorch
#from singleSlide import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, for our server either "0" , "1", "2" or "3";
os.environ["CUDA_VISIBLE_DEVICES"]="2";

# device  =torch.device("cuda:0")
# t = torch.ones((1,1)).to(device)
# t = torch.ones((1,1), device = device)
    


class PieceOfSlide:

    
    def __init__(
        self,
        img,
        orig_dim=(35856, 32482),
        model='./best_model.pth'
    ):
        self.image = None
        self.filename = None
        if type(img) == str:
            self.image = cv2.imread(img)
            self.filename = img
        elif type(img) == PIL.Image.Image:
            self.image = PieceOfSlide.__pil_to_cv2(img)
        elif type(img) == np.ndarray:
            self.image = img.copy()
        self.orig_dim = orig_dim
        self.model = None
        if type(model) == str:
            self.model = torch.load(model)
        elif type(model) == segmentation_models_pytorch.unet.model.Unet:
            self.model = model

################################################################        
        
    def image_show(self):
        plt.figure(figsize=(64, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.show()
        
    def image_show_jupyter(self):
        t = PieceOfSlide.__cv2_to_pil(self.image)
        display(t)
    
    def __image_show(image):
        plt.figure(figsize=(64, 10))
        plt.imshow(image)
        plt.show()
    
################################################################    
    
    def predict_mask(self, **flags):
        if self.image.shape[0] > 512 or self.image.shape[1] > 512:
            im0 = cv2.resize(self.image, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            im0 = self.image
        im0 = np.array(im0 / 255.0, dtype='float32')
        if flags.get('device'):
            device = flags['device']
            if device[:4] == 'cuda' or device[:3] == 'gpu':
                self.model.to('cuda')
                x_tensor_0 = torch.from_numpy(im0.transpose(2, 0, 1)).to(device).unsqueeze(0)
                pr_mask = self.model.predict(x_tensor_0)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            else:
                model.to('cpu')
                x_tensor_0 = torch.from_numpy(im0.transpose(2, 0, 1)).unsqueeze(0)
                pr_mask = self.model.predict(x_tensor_0)
                pr_mask = (pr_mask.squeeze().numpy().round())
        else:
            model.to('cpu')
            x_tensor_0 = torch.from_numpy(im0.transpose(2, 0, 1)).unsqueeze(0)
            pr_mask = self.model.predict(x_tensor_0)
            pr_mask = (pr_mask.squeeze().numpy().round())
        if flags.get('vis') == True or flags.get('show') == True:
            self.__visualize(
                image=im0[:, :, ::-1], 
                predicted_mask=pr_mask
            )
        
        
        
        if flags.get('save'):
            path_to_save=flags['save']
            kek = np.array(pr_mask*255, dtype='uint8')
            cv2.imwrite(path_to_save, kek)

        return pr_mask
    
    def predict_mask_gpu(self, **flags):

        im0 = self.image.copy()
        im0 = np.array(im0 / 255.0, dtype='float32')
        DEVICE = 'cuda'
        x_tensor_0 = torch.from_numpy(im0.transpose(2, 0, 1)).to(DEVICE).unsqueeze(0)
        pr_mask = self.model.predict(x_tensor_0)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        if flags.get('vis') == True or flags.get('show') == True:
            self.__visualize(
                image=im0[:, :, ::-1], 
                #ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )
        
        
        
        if flags.get('save'):
            path_to_save=flags['save']
            kek = np.array(pr_mask*255, dtype='uint8')
            cv2.imwrite(path_to_save, kek)
        
        return pr_mask


################################################################    
    
    def get_masked(self, mask=None, transform=None, show=True, copy=True, device='cuda'):
        if mask == None:
            mask = self.predict_mask(device=device)
        
        mask_inds = np.where(mask > 0)
        if copy:
            t = self.image.copy()
        else:
            t = self.image
        if transform:
            t = transform(t)
        if t.shape[0] != mask.shape[0]:
            t = cv2.resize(t, mask.shape, interpolation=cv2.INTER_AREA)
        t[mask_inds] = 255
        if show:
            PieceOfSlide.__image_show(t)
        return t
    
    def get_masked_gpu(self, mask=None, transform=None, show=True, copy=False):
        if mask == None:
            mask = self.predict_mask_gpu()
        
        mask_inds = np.where(mask > 0)
        if copy:
            t = self.image.copy()
        else:
            t = self.image
        if transform:
            t = transform(t)
        t[mask_inds] = 255
        if show:
            PieceOfSlide.__image_show(t)
        return t
    
    def get_kernels(self, show=True):
        mask = np.where(self.image[:, :, 0] >= 90)
        t = self.image.copy()
        t[mask] = 255
        if show:
            PieceOfSlide.__image_show(t)
        return t
    
    def count_epithelium_kernels(self):
        mask = self.predict_mask()
        mask_inds = np.where(mask > 0)
        mask_inds0 = np.where(mask == 0)
        
        masked = self.get_kernels(show=False)
        masked[mask_inds] = 255
        
        print('masked')
        image_show(masked[:, :, ::-1])  
        
        inds = np.where(masked[:, :, 2] > 80)
        masked[inds] = 255
          
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray  
 
        THRESHOLD_VALUE = np.unique(gray)[10]
        ret, gray2 = cv2.threshold(gray.copy(), THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        erosion = cv2.erode(gray2.copy(), kernel, iterations=1)
        #image_show(erosion)
        src = gray2.copy() #erosion.copy()#cv2.imread('/directorypath/image.bmp')
        #src = erosion.copy()
        
        connectivity = 4
        
        output = cv2.connectedComponentsWithStats(src, connectivity, cv2.CV_32S)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        centroids = output[3]
        
        coef = self.orig_dim[0] * self.orig_dim[1] / (35856.0 * 32482.0)
        new_labels = self.remove_invalid_areas(labels, coef)
        KERNEL_SIZE = self.extract_single_kernel(labels, coef)
        print('number of kernels:', np.count_nonzero(new_labels) / KERNEL_SIZE)
        
        
        f = np.vectorize(lambda x: (255- 255//num_labels*2* x, 255-255//num_labels*x, round(np.sqrt(36*x*(255-255//num_labels*x)))))
        t = np.array(f(new_labels), dtype='uint8')
        t = t.transpose(1, 2, 0)
        image_show(t)
        return [np.count_nonzero(new_labels) / KERNEL_SIZE, *output]
        
    def kek_function(self):
        print('kek2!')
        
        
    def weak_area(self, debug=False):
        def opening(image, kernel=None):
            if kernel==None:
                kernel = np.ones((3, 3))
            out = cv2.erode(image.copy(), kernel)
            out = cv2.dilate(out.copy(), kernel)
            return out

#         def closing(image, kernel=None):
#             if kernel==None:
#                 kernel = np.ones((2, 2))
#             out = cv2.dilate(image.copy(), kernel)
#             out = cv2.erode(out.copy(), kernel)
#             return out
        def normalize_shift(image_orig):
            image = image_orig.copy()
            image = (image - np.min(image_orig))/(np.max(image_orig) - np.min(image_orig))
            return image
        
        def round_image(image):
            out = image.copy()
            out *= 255
            out = np.array(out, dtype=np.uint8)
            return out
        ihc_ahx = col.separate_stains(self.image[:, :, ::-1], col.ahx_from_rgb)
        if debug:
            image_show(ihc_ahx[:, :, 0])
        t1 = normalize_shift(ihc_ahx[:, :, 0])
    
        #image_show(t1)
        #print_stats(t1)
        # THRESHOLD_EOSIN = 0.68
        # t[np.where(t < THRESHOLD_EOSIN)] = 0
        # t[np.where(t >0)] = 255
        t1 = round_image(t1)
        THRESHOLD = 175 #175 doesn't work for 3pr.svs
        t1[np.where(t1 < THRESHOLD)] = 0
        t1[np.where(t1 > 0)] = 255
        #image_show(t1)
        t1 = opening(t1)
        if debug:
            image_show(t1)
        hweak = np.count_nonzero(t1)
        print(ihc_ahx.shape)
        return hweak, t1, ihc_ahx
    
    def strong_area(self, debug=False, kernel=None):
        def opening(image, kernel=None):
            if kernel==None:
                kernel = np.ones((2, 2))
            out = cv2.erode(image.copy(), kernel)
            out = cv2.dilate(out.copy(), kernel)
            return out

#         def closing(image, kernel=None):
#             if kernel==None:
#                 kernel = np.ones((2, 2))
#             out = cv2.dilate(image.copy(), kernel)
#             out = cv2.erode(out.copy(), kernel)
#             return out
        def normalize_shift(image_orig):
            image = image_orig.copy()
            image = (image - np.min(image_orig))/(np.max(image_orig) - np.min(image_orig))
            return image
        
        def round_image(image):
            out = image.copy()
            out *= 255
            out = np.array(out, dtype=np.uint8)
            return out
            ihc_hed = col.separate_stains(self.image[:, :, ::-1], col.hed_from_rgb)
            t1 = normalize_shift(ihc_hed[:, :, 2]).copy()
            THRESHOLD = 0.73
            t1[np.where(t1 < THRESHOLD)] = 0
            #image_show(t1)
            t1 = round_image(opening(t1, kernel))
            image_show(t1)
            hstrong = np.count_nonzero(t1)
            return hstrong, t1, ihc_hed
    
    def visualize_strong(self):
        image_hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
        indices = np.where(1.03*image_hsv[:, :, 0] + 0.47 * image_hsv[:, :, 2] < image_hsv[:, :, 1])
        t = self.image.copy()
        t[indices] = 0
        h.visualize(original_image=self.image, strong=t, strong_with_closing=h.closing(t, channel=True))
        strong_with_closing = h.closing(t, channel=True)
        area = np.count_nonzero(strong_with_closing == 0)
        return area, area/(np.size(self.image) // 3), indices

    def visualize_weak(self):
        image_hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
        indices = np.where(np.logical_and(image_hsv[:, :, 1] > 15 ,np.logical_and( image_hsv[:, :, 0] > 1.15*image_hsv[:, :, 1], image_hsv[:, :, 2] <= 215)))
        print(indices)
        t = self.image.copy()
        t[indices] = 0
        h.visualize(original_image=self.image, weak=t, weak_with_closing=h.closing(t, channel=True))
        weak_with_closing = h.closing(t, channel=True)
        area = np.count_nonzero(weak_with_closing == 0)
        return area, area/(np.size(self.image) // 3), indices
    
    def count_blue_kernels(self):
        t = self.image.copy()
        
        
        #image_show(t[:, :, ::-1])
        
        #image_show(t[11:20, 70:78, ::-1])
        # print('blue:')
        # print(t[11:20, 70:78, 0])
        # print('green:')
        # print(t[11:20, 70:78, 1])
        # print('red:')
        # print(t[11:20, 70:78, 2])
        
        t1 = (t[:, :, 0] < 130)
        t2 = (t[:, :, 0] > 88)
        t3 = t[:, :, 1] < 100
        t4 = t[:, :, 2] < 100
        
        inds = np.where(np.logical_and(np.logical_and(t1, t2), np.logical_and(t3, t4)))
        s = np.zeros(t.shape, dtype=np.uint8)
        s[inds] = t[inds].copy()
        
        #image_show(s[:, :, ::-1])
        
        
        s3 = cv2.blur(s, (8, 8))
        s2 = s3.copy()
        s2 = cv2.pyrMeanShiftFiltering(s3, 10, 10, s2, 2)
        s4 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
        inds = np.where(s4 > 13)
        s4[inds] = 255
        
        #image_show(s4)
        
        inds = np.where(s4 < 255)
        s4[inds] = 0
        
        #kernel = np.ones((3, 3), dtype='uint8')
        #s5 = cv2.erode(s4, kernel)
        #image_show(s5)
        
        output = cv2.connectedComponentsWithStats(s4, 8, cv2.CV_32S)
        
        #print(output[0])
#         new_labels = output[1]

#         f = np.vectorize(lambda x: (255 - 255//num_labels*x, 
#                                     255-255//num_labels*x,
#                                     min(255, np.round(np.sqrt(3*x)))))
#         t2 = np.array(f(new_labels), dtype='uint8')
#         t2 = t2.transpose(1, 2, 0)
#         image_show(t2)

        return output[0]

        
################################################################
        
    def get_image(self):
        return self.image
        
    def normalized(self, show=False):
        s = np.zeros((self.image.shape[0], self.image.shape[1]))
        t = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        s = cv2.normalize(t.copy(), s, 0, 255, cv2.NORM_MINMAX)
        if show:
            PieceOfSlide.__image_show(s)
            display(PieceOfSlide.__cv2_to_pil(s))
        return t
    
    def normalized2(self):
        s = np.copy(self.image)
        s = (s - np.mean(s)) / np.std(s)
        s -= np.min(s)
        s /= np.max(s)
        return s
    
    
################################################################

    def count_areas(self, labels):
        unique_values = np.unique(labels)
        t = []
        for i in unique_values:
            area = np.count_nonzero(labels == i)
            t.append((i, area))
        return t
    
    def get_valid_areas(self, labels, coef=1.0):
        t = []
        for n, a in self.count_areas(labels):
            if a > 45*coef:
                t.append(n)
        return t

    def remove_invalid_areas(self, labels, coef=1.0):
        new_labels = labels.copy()
        valid_areas = self.get_valid_areas(labels, coef)
        for l in np.unique(labels):
            if not (l in valid_areas):
                inds = np.where(labels == l)
                new_labels[inds] = 0
        return new_labels
    
    def extract_single_kernel(self, labels, coef=1.0):
        t = []
        for n, a in self.count_areas(labels):
            if a > 45*coef and a < 150*coef:
                t.append(a)

        return np.mean(t)
    
################################################################

    def __visualize(self, **images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()


    def __cv2_to_pil(image):
        out = None
        if len(image.shape) == 3:
            out = Image.fromarray(image[:, :, ::-1])
        if len(image.shape) < 3:
            out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            out = Image.fromarray(out)
        return out
    
    def __pil_to_cv2(image):
        out = image.convert("RGB")
        out = np.array(out, dtype='uint8')
        out = out[:, :, ::-1]
        return out
    
    def __image_show(image):
        plt.figure(figsize=(64, 10))
        plt.imshow(image)
        plt.show()

    def __print_stats(arr):
        print('median:', np.median(arr))
        print('mean:', np.mean(arr))
        print('max:', np.max(arr))
        print('min:', np.min(arr))
        return (np.median(arr), np.mean(arr), np.max(arr), np.min(arr)) 

    
def image_show(image):
    plt.figure(figsize=(64, 10))
    plt.imshow(image)
    plt.show()