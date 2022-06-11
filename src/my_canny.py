import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Img
from PIL import Image, ImageOps
import multiprocessing

def conv(X, filters, stride=1, pad=1):
    h, w = X.shape
    filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(pad, pad), (pad, pad)], 'constant')
    out = np.zeros((out_h, out_w))

    for h in range(out_h): # slide the filter vertically.
        h_start = h * stride
        h_end = h_start + filter_h
        for w in range(out_w): # slide the filter horizontally.
            w_start = w * stride
            w_end = w_start + filter_w
            # Element-wise multiplication.
            out[h, w] = np.sum(in_X[h_start:h_end, w_start:w_end] * filters)

    return out

def im2col(X, filters, stride=1, pad=0):
    n, c, h, w = X.shape
    n_f, _, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    out = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for h in range(filter_h):
        h_end = h + stride * out_h
        for w in range(filter_w):
            w_end = w + stride * out_w
            out[:, :, h, w, :, :] = in_X[:, :, h:h_end:stride, w:w_end:stride]

    out = out.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return out


def conv2d_np(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # XCorrel

    sub_matrices = np.lib.stride_tricks.as_strided(image,
                                                   shape=tuple(np.subtract(image.shape, kernel.shape)) + kernel.shape,
                                                   strides=image.strides * 2)

    return np.einsum('ij,klij->kl', kernel, sub_matrices)


def gaussian_kernel(k_size, sigma):
    size = k_size // 2
    y, x = np.ogrid[-size:size + 1, -size:size + 1]
    filter = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    sum = filter.sum()
    filter /= sum
    return filter


def padding(img, k_size):
    pad_size = k_size // 2
    rows, cols, ch = img.shape

    res = np.zeros((rows + (2 * pad_size), cols + (2 * pad_size), ch), dtype=np.float32)

    if pad_size == 0:
        res = img.copy()
    else:
        res[pad_size:-pad_size, pad_size:-pad_size] = img.copy()
    return res


def gaussian_filtering(img, k_size=3, sigma=1):
    img = np.expand_dims(img,axis=2)
    rows, cols, channels = img.shape
    filter = gaussian_kernel(k_size, sigma)
    print("filter = ",filter)
    pad_img = padding(img, k_size)
    filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)
    for ch in range(0, channels):
        for i in range(rows):
            for j in range(cols):
                filtered_img[i, j, ch] = np.sum(filter * pad_img[i:i + k_size, j:j + k_size, ch])

    return filtered_img.astype(np.uint8)


def non_max_suppression(img, D):
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    print("in function img max ",np.max(img))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    out[i, j] = img[i, j]
                else:
                    out[i, j] = 0
            except IndexError as e:
                pass
    return out


def non_maximum_surpression(edge_gradient, angle):
    eg_h, eg_w = edge_gradient.shape
    angle = (angle-1) // 4
    out = np.copy(edge_gradient)
    for i in range(1,eg_h-1):
        for j in range(1,eg_w-1):
            if angle[i,j] == 0:
                if np.max(edge_gradient[i-1:i+2,j])!=edge_gradient[i,j]:
                    out[i,j]=0
            if angle[i,j] == 1:
                if np.max([edge_gradient[i-1,j-1],edge_gradient[i,j],edge_gradient[i+1,j+1]])!=edge_gradient[i,j]:
                    out[i,j]=0
            if angle[i,j]==2:
                if np.max(edge_gradient[i,j-1:j+2])!=edge_gradient[i,j]:
                    out[i,j]=0
            if angle[i,j]==3:
                if np.max([edge_gradient[i-1,j+1],edge_gradient[i,j],edge_gradient[i+1,j-1]])!=edge_gradient[i,j]:
                    out[i,j]=0
    return out


def hysteresis(min_img, max_img, weak, strong=255):
    h, w = min_img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if (min_img[i,j] == weak):
                try:
                    if ((max_img[i+1, j-1] == strong) or (max_img[i+1, j] == strong) or (max_img[i+1, j+1] == strong)
                        or (max_img[i, j-1] == strong) or (max_img[i, j+1] == strong)
                        or (max_img[i-1, j-1] == strong) or (max_img[i-1, j] == strong) or (max_img[i-1, j+1] == strong)):
                        min_img[i, j] = strong
                    else:
                        min_img[i, j] = 0
                except IndexError as e:
                    pass
    return min_img

def my_canny(img, min_threshold, max_threshold):
    img = np.expand_dims(img,axis=(0,1))
    sobel_x = np.array([[[[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]]]])

    sobel_y = np.array([[[[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]]]])
    stride = 1

    starttime = time.time_ns()
    sobel_X_col = im2col(img, sobel_x, pad=1)
    sobel_Y_col = im2col(img, sobel_y, pad=1)
    n, c, h, w = img.shape
    n_f, _, filter_h, filter_w = sobel_x.shape
    out_h = (h + 2 * 1 - filter_h) // stride + 1
    out_w = (w + 2 * 1 - filter_w) // stride + 1
    
    out_x = np.matmul(sobel_X_col, sobel_x.reshape(n_f, -1).T)
    out_x = out_x.reshape(n, out_h, out_w, n_f)
    out_x = out_x.transpose(0, 3, 1, 2)
    out_x = out_x.squeeze(axis=(0,1))
    
    out_y = np.matmul(sobel_Y_col, sobel_y.reshape(n_f, -1).T)
    out_y = out_y.reshape(n, out_h, out_w, n_f)
    out_y = out_y.transpose(0, 3, 1, 2)
    out_y = out_y.squeeze(axis=(0,1))

    # out_x = conv2d_np(img.squeeze(axis=(0,1)),sobel_x.squeeze(axis=(0,1)))
    # out_y = conv2d_np(img.squeeze(axis=(0,1)),sobel_y.squeeze(axis=(0,1)))

    # out_x = conv(img.squeeze(axis=(0,1)),sobel_x.squeeze(axis=(0,1)))
    # out_y = conv(img.squeeze(axis=(0,1)),sobel_y.squeeze(axis=(0,1)))
    endtime = time.time_ns()
    convolution_time = endtime-starttime

    edge_gradient = out_x ** 2 + out_y ** 2
    edge_gradient = np.sqrt(edge_gradient)
    print(np.max(edge_gradient),np.min(edge_gradient))

    theta = np.arctan2(out_y, out_x)
    starttime = time.time_ns()
    nms = non_max_suppression(edge_gradient, theta)
    endtime = time.time_ns()
    max_suppression_time = endtime-starttime
    # print("nms",np.max(nms),np.min(nms))

    starttime = time.time_ns()
    max_thresholding = np.array([0, max_threshold])
    min_thresholding = np.array([0, min_threshold])
    max_thresholding_array = (np.digitize(nms, max_thresholding) - 1) * 255
    min_thresholding_array = (np.digitize(nms, min_thresholding) - 1) * 50
    # max_thresholding_array = max_thresholding_array.astype('uint8')
    # min_thresholding_array = min_thresholding_array.astype('uint8')
    endtime = time.time_ns()
    threshold_time = endtime-starttime

    starttime = time.time_ns()
    result = hysteresis(np.copy(min_thresholding_array),np.copy(max_thresholding_array),50)
    endtime = time.time_ns()
    hysteresis_time = endtime-starttime
    print("convolution process time         :",convolution_time,"ns")
    print("non_max_suppression process time :",max_suppression_time,"ns")
    # print("threshold_time",threshold_time)
    print("hysteresis thresholding time     :",hysteresis_time,"ns")

    return result.astype('uint8')

def gaussian_blur(img,k_size=3,sigma=1):
    img = np.expand_dims(img,axis=(0,1))
    filter = gaussian_kernel(k_size,sigma)
    filter = np.expand_dims(filter,axis=(0,1))
    pad = k_size // 2
    X_col = im2col(img, filter, pad=pad)

    n, c, h, w = img.shape
    n_f, _, filter_h, filter_w = filter.shape

    out_h = (h + 2 * pad - filter_h) // 1 + 1
    out_w = (w + 2 * pad - filter_w) // 1 + 1

    out = np.matmul(X_col, filter.reshape(n_f, -1).T)
    out = out.reshape(n, out_h, out_w, n_f)
    out = out.transpose(0, 3, 1, 2)
    out = out/np.max(out) * 255
    out = out.astype('uint8')

    return out.squeeze(axis=(0,1))



def gaussian_fitering_im2col(img,k_size=3,sigma=1):
    img = np.expand_dims(img,axis=(0,1))
    filter = gaussian_kernel(k_size,sigma)
    filter = np.expand_dims(filter,axis=(0,1))
    sobel_x = np.array([[[[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]]]])

    sobel_y = np.array([[[[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]]]])
    print(filter.shape)
    print(sobel_x.shape)

    stride = 1
    pad = k_size//2
    # img gaussian filtering
    X_col = im2col(img, filter,pad=pad)

    n, c, h, w = img.shape
    n_f, _, filter_h, filter_w = filter.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    out = np.matmul(X_col, filter.reshape(n_f, -1).T)
    out = out.reshape(n, out_h, out_w, n_f)
    out = out.transpose(0, 3, 1, 2)
##################
    # sobel gradient distance
    sobel_X_col = im2col(out,sobel_x,pad=1)
    sobel_Y_col = im2col(out,sobel_y,pad=1)
    n, c, h, w = out.shape
    n_f, _, filter_h, filter_w = sobel_x.shape

    out_h = (h + 2 * 1 - filter_h) // stride + 1
    out_w = (w + 2 * 1 - filter_w) // stride + 1

    out_x = np.matmul(sobel_X_col, sobel_x.reshape(n_f, -1).T)
    out_x = out_x.reshape(n, out_h, out_w, n_f)
    out_x = out_x.transpose(0, 3, 1, 2)

    out_y = np.matmul(sobel_Y_col, sobel_y.reshape(n_f, -1).T)
    out_y = out_y.reshape(n, out_h, out_w, n_f)
    out_y = out_y.transpose(0, 3, 1, 2)
    print("original img blurring shape", out.shape)
    print("sobel x gradient shape",out_x.shape)
    print("sobel y gradient shape",out_y.shape)

    edge_gradient = out_x**2 + out_y**2
    edge_gradient = np.sqrt(edge_gradient)
    # egmax = np.max(edge_gradient)
    # egmin = np.min(edge_gradient)
    # edge_gradient = (edge_gradient - egmin)/egmax

    # edge의 방향
    # 1 :   0, 5 : 180      위아래값 비교
    # 2 :  45, 6 : 225      왼쪽위, 오른쪽아래 비교
    # 3 :  90, 7 : 270      왼쪽,오른쪽 비교
    # 4 : 135, 8 : 315      왼쪽아래, 오른쪽 위 비교
    # gradient의 방향의 edge의 수직
    print("222 eg shape",edge_gradient.shape)
    print("edge gradient max ",np.max(edge_gradient))
    theta = np.arctan2(out_y,out_x)
    nms2 = non_max_suppression(edge_gradient.squeeze(axis=(0,1)), theta.squeeze(axis=(0,1)))
    nms2 = nms2 / np.max(nms2) * 255
    nms2 = nms2.astype('uint8')
    print(nms2.shape)
    print("nms2 max",np.max(nms2))

    max_threshold = 100
    min_threshold = 30
    max_thresholding = np.array([0,max_threshold])
    min_thresholding = np.array([0,min_threshold])
    max_thresholding_array = (np.digitize(nms2,max_thresholding)-1)*255
    # nms2 = max_thresholding_array + nms2
    # nms2 = nms2.astype('uint8')
    min_thresholding_array = (np.digitize(nms2,min_thresholding)-1)*50
    max_thresholding_array = max_thresholding_array.astype('uint8')
    min_thresholding_array = min_thresholding_array.astype('uint8')
    print("max threshold non zero",np.count_nonzero(max_thresholding_array))
    print("min threshold non zero",np.count_nonzero(min_thresholding_array))
    result = hysteresis(np.copy(min_thresholding_array),np.copy(max_thresholding_array),50)

    out = out.squeeze(axis=(0,1))
    out_x = out_x.squeeze(axis=(0,1))
    out_y = out_y.squeeze(axis=(0,1))
    edge_gradient = edge_gradient.squeeze(axis=(0,1))
    edge_gradient = edge_gradient / (np.max(edge_gradient) - np.min(edge_gradient)) * 255

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(out, cmap='gray')
    plt.subplot(3, 3, 2)
    plt.imshow(out_x, cmap='gray')
    plt.subplot(3, 3, 3)
    plt.imshow(out_y, cmap='gray')
    plt.subplot(3, 3, 4)
    plt.imshow(edge_gradient, cmap='gray')
    plt.subplot(3, 3, 5)
    plt.imshow(nms2, cmap='gray')
    plt.subplot(3, 3, 6)
    plt.imshow(max_thresholding_array, cmap='gray')
    plt.subplot(3, 3, 7)
    plt.imshow(min_thresholding_array, cmap='gray')
    plt.subplot(3, 3, 8)
    plt.imshow(result, cmap='gray')
    plt.show()
    # print(out)
    # return out
    # return out, out_x, out_y,
    # plt.imshow(out,cmap='gray')
    # plt.show()
