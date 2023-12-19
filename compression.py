import cv2
import lbg
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_training(img, block):
    train_vec = []
    x = block[0]
    y = block[1]
    for i in range(0, img.shape[0], x):
        for j in range(0, img.shape[1], y):
            train_vec.append(img[i:i + x, j:j + y].reshape((x * y)))
    return (np.array(train_vec))
def distance(a, b):
    return np.mean((np.subtract(a, b) ** 2))
def closest_match(src, cb):
    c = np.zeros((cb.shape[0],))
    for i in range(0, cb.shape[0]):
        c[i] = distance(src, cb[i])
    minimum = np.argmin(c, axis=0)
    return minimum


def encode_image(img, cb, block):
    x = block[0]
    y = block[1]
    compressed = np.zeros((img.shape[0] // y, img.shape[1] // x))
    ix = 0
    for i in range(0, img.shape[0], x):
        iy = 0
        for j in range(0, img.shape[1], y):
            src = img[i:i + x, j:j + y].reshape((x * y)).copy()
            k = closest_match(src, cb)
            compressed[ix, iy] = k
            iy += 1
        ix += 1
    return compressed


def decode_image(cb, compressed, block):
    x = block[0]
    y = block[1]
    original = np.zeros((compressed.shape[0] * y, compressed.shape[1] * x))
    ix = 0
    for i in range(0, compressed.shape[0]):
        iy = 0
        for j in range(0, compressed.shape[1]):
            original[ix:ix + x, iy:iy + y] = cb[int(compressed[i, j])].reshape(block)
            iy += y
        ix += x
    return original


def save_weight(filename, cb):
    fd = open(filename, 'a')
    for i in range(0, cb.shape[0]):
        linecsv = str(cb[i]) + '\n'
        fd.write(linecsv)
    fd.close()


def save_codebook(filename, cb):
    fd = open(filename, 'a')
    for i in range(0, cb.shape[0]):
        linecsv = ''
        for j in range(0, cb.shape[1]):
            linecsv = linecsv + str(cb[i, j]) + ','
        linecsv = linecsv + '\n'
        fd.write(linecsv)
    fd.close()


def save_csv(root, csv, cb, cb_abs_w, cb_rel_w):
    numpy_cb = np.array(cb)
    numpy_abs_w = np.array(cb_abs_w)
    numpy_rel_w = np.array(cb_rel_w)
    save_codebook(root + 'CB_' + csv + '.csv', numpy_cb)
    save_weight(root + '3CB_abs_' + csv + '.csv', numpy_abs_w)
    save_weight(root + '3CB_rel_' + csv + '.csv', numpy_rel_w)


def sim_protocol(img, cb_size, epsilon, block, root, outpng):
    train_X = generate_training(img, block)
    cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(train_X, cb_size, epsilon)
    cb_n = np.array(cb)
    cb_abs_w_n = np.array(cb_abs_w)
    cb_rel_w_n = np.array(cb_rel_w)
    result = encode_image(img, cb_n, block)
    final_result = decode_image(cb_n, result, block)
    save_csv(root, outpng, cb_n, cb_abs_w_n, cb_rel_w_n)
    return final_result
