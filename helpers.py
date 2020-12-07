import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize(**images):
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
    
def opening(image, kernel=None, iterations=1, channel=False, debug=False):
    if kernel is None:
        kernel = np.ones((3, 3))
    out_1 = np.zeros(image.shape, dtype=np.uint16)
    out_2 = np.zeros(image.shape, dtype=np.uint16)
    for _ in range(iterations):
        if channel:
            if len(image.shape) < 3:
                return image
            else:        
                for i in range(3):
                    out_1[:, :, i] = cv2.erode(image[:, :, i], kernel)
                    out_2[:, :, i] = cv2.dilate(out_1[:, :, i], kernel)
                    if debug:
                        visualize(before=image, erode=out_1, opened=out_2)
        else:        
            out_1 = cv2.erode(image, kernel)
            out_2 = cv2.dilate(out_1, kernel)
        if debug:
            visualize(before=image, erode=out_1, opened=out_2)
    return out_2

def closing(image, kernel=None, iterations=1, channel=False, debug=False):
    if kernel is None:
        kernel = np.ones((2, 2))
        
    out_1 = np.zeros(image.shape, dtype=np.uint16)
    out_2 = np.zeros(image.shape, dtype=np.uint16)
    
    for _ in range(iterations):
        if channel:
            
            if len(image.shape) < 3:
                return image
            
            else:
                for i in range(3):
                    out_1[:, :, i] = cv2.dilate(image[:, :, i], kernel)
                    out_2[:, :, i] = cv2.erode(out_1[:, :, i], kernel)
                    if debug:
                        visualize(before=image, erode=out_1, closed=out_2)
        else:        
            out_1 = cv2.dilate(image, kernel)
            out_2 = cv2.erode(out_1, kernel)
            
        if debug:
            visualize(before=image, dilate=out_1, closed=out_2)
            
    return out_2


def normalize_shift(image_orig):
    image = image_orig.copy()
    image = (image - np.min(image_orig))/(np.max(image_orig) - np.min(image_orig))
    return image

def normalize_cut(image_orig):
    image = np.array(image_orig.copy(), dtype=np.uint16)
    image[image > 255] = 255
    image[image < 0] = 0
    return image

def round_image(image):
    out = image.copy()
    out *= 255
    out = np.array(np.round(out), dtype=np.uint8)
    return out

def image_show(image):
    plt.figure(figsize=(64, 10))
    plt.imshow(image, vmin=0, vmax=255)
    plt.show()
    
# def plot_tables(objects, titles=['No title']):
#     fig, axes = plt.subplots(len(objects) // 4 + 1, 4, figsize=(16, 16), sharex=True, sharey=True)
#     ax = axes.ravel()
#     if type(titles) != list:
#         titles = [titles]
#     div, rem = len(objects) // len(titles), len(objects) % len(titles)
#     titles = titles * div + titles[:rem]
#     for i, o in enumerate(objects):
#         o
        #ax[i].set_title = titles[i]
        
# t = piece_of_slide.PieceOfSlide(er10.read_region((24000, 6000), 1, (512, 512)), model=model)

# hls = cv2.cvtColor(t.get_image(), cv2.COLOR_BGR2HSV)
# fig, axes = plt.subplots(3, 4, figsize=(13, 10), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(t.get_image()[:, :, ::-1])
# ax[0].set_title("Original image")

# ax[1].imshow(hls[:, :, 0])
# ax[1].set_title("Hue")

# ax[2].imshow(hls[:, :, 1])
# ax[2].set_title("Saturation")

# ax[3].imshow(hls[:, :, 2])
# ax[3].set_title("Vaue")

# t1 = piece_of_slide.PieceOfSlide(er10.read_region((1024*35, 1024*26), 1, (512, 512)), model=model)
# hls_1 = cv2.cvtColor(t1.get_image(), cv2.COLOR_BGR2HSV)


# ax[4].imshow(t1.get_image()[:, :, ::-1])
# ax[4].set_title("Original image")

# ax[5].imshow(hls_1[:, :, 0])
# ax[5].set_title("Hue")

# ax[6].imshow(hls_1[:, :, 1])
# ax[6].set_title("Saturation")

# ax[7].imshow(hls_1[:, :, 2])
# ax[7].set_title("Vaue")

# t2 = piece_of_slide.PieceOfSlide(pr4.read_region((11000, 9000), 0, (512, 512)), model=model)
# hls_2 = cv2.cvtColor(t2.get_image(), cv2.COLOR_BGR2HSV)
# ax[8].imshow(t2.get_image()[:, :, ::-1])
# ax[8].set_title("Original image")

# ax[9].imshow(hls_2[:, :, 0])
# ax[9].set_title("Hue")

# ax[10].imshow(hls_2[:, :, 1])
# ax[10].set_title("Saturation")

# ax[11].imshow(hls_2[:, :, 2])
# ax[11].set_title("Vaue")