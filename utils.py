import numpy as np
from PIL import Image
from scipy import ndimage


def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def gaussian_kernel(size=5, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def gaussian_smooth(img, size, sigma):
    np_image = np.asarray(img)
    np_image = 255 - np_image
    smoothed_image = ndimage.filters.convolve(np_image, gaussian_kernel(size, sigma))
    out_image = 255 - smoothed_image
    out_image = Image.fromarray(out_image.astype('uint8'))
    return out_image


def add_pepper_noise(image, prop=0.05):
    np_image = np.asarray(image)
    row, column = np_image.shape
    pepper = np.random.randint(0, 256, (row, column))
    pepper = np.where(pepper < prop * 256, -255, 0)
    noise_image = np_image + pepper
    noise_image = np.where(noise_image < 0, 0, noise_image).astype('uint8')
    pil_image = Image.fromarray(noise_image)
    return pil_image


def add_gauss_noise(image):
    np_image = np.asarray(image)
    row, column = np_image.shape
    gauss_noise = np.random.normal(0, 50, (row, column))
    noise_image = np_image + gauss_noise
    noise_image = np.where(noise_image < 0, 0, np.where(noise_image > 255, 255, noise_image)).astype('uint8')
    pil_image = Image.fromarray(noise_image)
    return pil_image


# for testing
if __name__ == '__main__':
    image = Image.open('rcf-data/rcf_blur.png').convert('L')
    np_image = np.asarray(image)
    np_image = 255 - np_image
    smoothed_image = ndimage.filters.convolve(np_image, gaussian_kernel(size=5, sigma=1))
    out_image = 255 - smoothed_image
    out_image = Image.fromarray(out_image.astype('uint8'))
    out_image.show()
    out_image.save('rcf-data/rcf_g.png')
