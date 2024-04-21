import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        # filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    height, width = img.shape

    filter_size //= 2
    
    padded_img = np.zeros((height + 2 * filter_size, width + 2 * filter_size), dtype=img.dtype)
    
    padded_img[filter_size:filter_size + height, filter_size:filter_size + width] = img

    # Padding top
    padded_img[:filter_size, filter_size:filter_size + width] = img[0]
    #Padding bottom
    padded_img[-filter_size:, filter_size:filter_size + width] = img[-1]

    for k in range (filter_size):
    # Padding left
      padded_img[:, k] = padded_img[:, filter_size]
    # Padding right
      padded_img[:, - k -1] = padded_img[:, - filter_size - 1]
      
    return padded_img 

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
  # Need to implement here

    height, width = img.shape

    padded_img = padding_img(img, filter_size)

    # Assume stride = 1
    
    kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size ** 2)

    filtered_img = np.zeros(img.shape, dtype = np.float32)

    for i in range (height):
      for j in range (width):
        neighbor = padded_img[i:i+filter_size, j:j+filter_size]
        filtered_img[i, j] = np.sum(neighbor * kernel)

    filtered_img = filtered_img.astype(img.dtype)
    return filtered_img


def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
  # Need to implement here
    height, width = img.shape

    padded_img = padding_img(img, filter_size)

    # Assume stride = 1
    
    kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size ** 2)

    filtered_img = np.zeros(img.shape, dtype = np.float32)

    for i in range (height):
      for j in range (width):
        neighbor = padded_img[i:i+filter_size, j:j+filter_size]

        filtered_img[i, j] = np.median(neighbor)

    filtered_img = filtered_img.astype(img.dtype)

    return filtered_img
  


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here
    MSE = np.mean((gt_img - smooth_img)**2)

    MAX = 255.0

    score = 10*(np.log10((MAX*MAX) / MSE))

    return score


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    img_o = read_img(img_gt)
    filter_size = 3

    # # Padding 
    # padded_img = padding_img(img, filter_size)
    # show_res(img, padded_img)
    # print(img.shape, padded_img.shape)
    
    # print('PSNR score of mean filter: ', psnr(img, img_o))

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))
    # print(img.shape, mean_smoothed_img.shape)

    

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

    # PSNR score of mean filter:  31.60920520374559
    # PSNR score of median filter:  37.119578300855245
    # -> For the provided image we should choose the median filter as the PNSR is higher


