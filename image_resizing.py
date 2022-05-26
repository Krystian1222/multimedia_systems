import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
import random
###################################################################################################
def nearest_neighbour(image_in, scale):
    image_out = np.zeros((int(image_in.shape[0] * scale), int(image_in.shape[1] * scale), image_in.shape[2])).astype(np.uint8)
    for x in range(image_in.shape[0]):
        for y in range(image_in.shape[1]):
            x_in, y_in = floor(x / scale), floor(y / scale)
            for z in range(image_in.shape[2]):
                image_out[x][y][z] = image_in[x_in][y_in][z]
    return image_out
###################################################################################################
def bilinear_interpolation(image_in, scale):
    image_out = np.zeros((int(image_in.shape[0] * scale), int(image_in.shape[1] * scale), image_in.shape[2])).astype(np.uint8)
    height = np.linspace(0, image_in.shape[0] - 1, int(image_in.shape[0] * scale))
    width = np.linspace(0, image_in.shape[1] - 1, int(image_in.shape[1] * scale))
    for x in range(image_out.shape[0]):
        for y in range(image_out.shape[1]):
            x_b, y_b, x_t, y_t = floor(height[x]), floor(width[y]), ceil(height[x]), ceil(width[y])
            xn, yn = height[x] - x_b, width[y] - y_b
            for z in range(image_in.shape[2]):
                image_out[x][y][z] = image_in[x_b][y_b][z] * (1 - xn) * (1 - yn) + image_in[x_t][y_b][z] * xn * (1 - yn) + image_in[x_b][y_t][z] * (1 - xn) * yn + image_in[x_t][y_t][z] * xn * yn
    return image_out
###################################################################################################
def mean(image_in, scale):
    image_out = np.zeros((int(image_in.shape[0] * scale), int(image_in.shape[1] * scale), image_in.shape[2])).astype(np.uint8)
    for x in range(image_out.shape[0]):
        for y in range(image_out.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / scale - 1 / scale), int(x * 1 / scale + 1 / scale))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / scale - 1 / scale), int(y * 1 / scale + 1 / scale))) if a >= 0]
            tab = [image_in[i][j] for j in yn for i in xn]
            result = list()
            for k in range(len(tab)):
                sum = 0
                for l in range(len(tab)):
                    sum += tab[l][k]
                result.append(sum / len(tab))
            image_out[x][y] = result
    return image_out
###################################################################################################
def weighted_mean(image_in, scale):
    image_out = np.zeros((int(image_in.shape[0] * scale), int(image_in.shape[1] * scale), image_in.shape[2])).astype(np.uint8)
    for x in range(image_out.shape[0]):
        for y in range(image_out.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / scale - 1 / scale), int(x * 1 / scale + 1 / scale))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / scale - 1 / scale), int(y * 1 / scale + 1 / scale))) if a >= 0]
            tab = [image_in[i][j] for j in yn for i in xn]
            weights, result, sum_of_weights = list(), list(), 0
            for i in range(len(xn)):
                for j in range(len(yn)):
                    value = random.randint(1, 10) / 10
                    weights.append(value)
                    sum_of_weights += value
            for k in range(image_in.shape[2]):
                sum = 0
                for l in range(len(tab)):
                    sum += (tab[l][k] * weights[l])
                result.append(sum / sum_of_weights)
            image_out[x][y] = result
    return image_out
###################################################################################################
def median(image_in, scale):
    image_out = np.zeros((int(image_in.shape[0] * scale), int(image_in.shape[1] * scale), image_in.shape[2])).astype(np.uint8)
    for x in range(image_out.shape[0]):
        for y in range(image_out.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / scale - 1 / scale), int(x * 1 / scale + 1 / scale))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / scale - 1 / scale), int(y * 1 / scale + 1 / scale))) if a >= 0]
            tab = [image_in[i][j] for j in yn for i in xn]
            result, median_arg = list(), np.zeros((len(tab)))
            for k in range(image_in.shape[2]):
                for l in range(len(tab)):
                    median_arg = tab[l][k]
                result.append(np.median(median_arg))
            image_out[x][y] = result
    return image_out
###################################################################################################
def enlarged_images(image, scale, name):
    title_size = 7
    fig = plt.figure(figsize=(16, 7))
    images = [
        image,
        nearest_neighbour(image, scale),
        cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_NEAREST),
        bilinear_interpolation(image, scale),
        cv2.resize(image, (int(image.shape[1] * image), int(image.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    ]

    titles = [
        'Original image',
        'Nearest neighbour - scale = {}'.format(scale),
        'Nearest neighbour - embedded - scale = {}'.format(scale),
        'Bilinear interpolation - scale = {}'.format(scale),
        'Bilinear interpolation - embedded - scale = {}'.format(scale)
    ]

    for i in range(len(titles)):
        titles.append(titles[i])

    images += [cv2.Canny(images[i], 100, 200) for i in range(len(images))]
    for i in range(len(titles)):
        ax = fig.add_subplot(2, 5, i + 1)
        plt.imshow(images[i]) if i < 5 else plt.imshow(images[i], cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=title_size)
    plt.savefig("enlarged_2_scale_{}_{}.png".format(scale, name))
###################################################################################################
def reduced_images(image, scale, name):
    title_size = 7
    fig = plt.figure(figsize=(20, 7))
    images = [
        image,
        nearest_neighbour(image, scale),
        bilinear_interpolation(image, scale),
        mean(image, scale),
        weighted_mean(image, scale),
        median(image, scale)
    ]

    titles = [
        'Original image',
        'Nearest neighbour - scale = {}'.format(scale),
        'Bilinear interpolation - scale = {}'.format(scale),
        'Mean - scale = {}'.format(scale),
        'Weighted mean - scale = {}'.format(scale),
        'Median - scale = {}'.format(scale)
    ]

    for i in range(len(titles)):
        titles.append(titles[i])

    images += [cv2.Canny(images[i], 100, 200) for i in range(len(images))]
    for i in range(len(titles)):
        ax = fig.add_subplot(2, 6, i + 1)
        plt.tight_layout()
        plt.imshow(images[i]) if i < 6 else plt.imshow(images[i], cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=title_size)
    plt.savefig("reduced_2_scale_{}_{}.png".format(scale, name))
###################################################################################################
def main():
    images = [plt.imread("0001.jpg"), plt.imread("0002.jpg"), plt.imread("0003.jpg"), plt.imread("0004.jpg"),
              plt.imread("0005.jpg"), plt.imread("0006.jpg"), plt.imread("0007.jpg"), plt.imread("0008.tif")]
    names = ["000{}".format(i) for i in range(1, 9)]
    scales = [1.5, 2, 4, 8, 16, 0.05, 0.15, 0.25, 0.5, 0.75]
    for i in range(len(images)):
        for j in range(len(scales)):
            if i < 4 and j < 5:
                enlarged_images(images[i], scales[j], names[i])
            elif i >= 4 and j >= 5:
                reduced_images(images[i], scales[j], names[i])
###################################################################################################
if __name__ == '__main__':
    main()
