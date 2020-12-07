import numpy as np
import os
import PIL
from PIL import Image, ImageColor, ImageEnhance, ImageFilter
import openslide
import cv2
import random
import matplotlib.pyplot as plt
import torch
import skimage
from skimage import io
from skimage import data
import skimage.color as col
import piece_of_slide
import segmentation_models_pytorch

class WholeSlide(openslide.OpenSlide):
    def __init__(
        self,
        filename='',
        model='./best_model.pth'
    ):
        #xself.filename = filename
        super().__init__(filename)
        self.model = None
        if type(model) == str:
            self.model = torch.load(model)
        elif type(model) == segmentation_models_pytorch.unet.model.Unet:
            self.model = model
            
    def predict_mask_region(self, top_left, ex, size, device='cuda'):
        region = self.read_region(top_left, ex, size)
        region = piece_of_slide.PieceOfSlide(region, model=self.model)
        return region.predict_mask(device=device)

    def weak_region(self, top_left, ex, size, debug=False):
        region = self.read_region(top_left, ex, size)
        region = piece_of_slide.PieceOfSlide(region, model=self.model)
        return region.weak_area(debug=debug)

    def strong_region(self, top_left, ex, size, debug=False):
        region = self.read_region(top_left, ex, size)
        region = piece_of_slide.PieceOfSlide(region, model=self.model)
        return region.strong_area(debug=debug)
    
    def get_masked_region(self, top_left, ex, size, device='cuda', show=False):
        region = self.read_region(top_left, ex, size)
        region = piece_of_slide.PieceOfSlide(region, model=self.model)
        if show:
            region.image_show()
        return region.get_masked(show=show, device=device)
    
    def predict_mask_whole(self, device='cuda'):
        region = piece_of_slide.PieceOfSlide(self.read_region((0, 0), 0, self.level_dimensions[0]), model=self.model)
        return region.predict_mask(device)
    
    def get_masked_whole(self, device='cuda'):
        l = 1024
        g1, g2 = np.ceil(np.array(self.level_dimensions[1]) / l)
        if self.level_dimensions[0][0] < 40000:
            l = 3849
            g1, g2 = np.ceil(np.array(self.level_dimensions[0]) / l)
        row = None
        for i in range(int(g1)):
            col = None
            for j in range(int(g2)):
                print(i, j)
                t = self.get_masked_region((i * l, j * l), 0, (l, l), device)
                if j == 0:
                    col = t #cv2.resize(t, (480, 480), interpolation=cv2.INTER_CUBIC)
                else:
                    col = np.vstack((col,  t)) #cv2.resize(t, (480, 480), interpolation=cv2.INTER_CUBIC)))
            if i == 0:
                row = col
            else:
                row = np.hstack((row, col))
        masked = cv2.resize(row, (1536, 1536), interpolation=cv2.INTER_LINEAR)
        return masked
