import math
import numpy as np
from functools import reduce
from collections import defaultdict
_size_data = 0
_dim = 0
def generate_codebook(data, size_codebook, epsilon):
    global _size_data, _dim
    _size_data = len(data)
    assert _size_data > 0
    _dim = len(data[0])
    assert _dim > 0
    codebook = []
    codebook_abs = [_size_data]
    codebook_rel = [1.0]
    c0 = avg_all_vectors(data, _dim, _size_data)
    codebook.append(c0)
    avg_dist = initial_avg_distortion(c0, data)
    while len(codebook) < size_codebook:
        codebook, codebook_abs, codebook_rel, avg_dist = split_codebook(data, codebook,epsilon, avg_dist)
    return codebook, codebook_abs, codebook_rel

def split_codebook(data, codebook, epsilon, initial_avg_dist):
    new_cv = []
    for c in codebook:
        c1 = new_codevector(c, epsilon)
        c2 = new_codevector(c, -epsilon)
        new_cv.extend((c1, c2))

    codebook = new_cv
    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook
    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        closest_c_list = [None] * _size_data
        vecs_near_c = defaultdict(list)
        vec_idxs_near_c = defaultdict(list)
        for i, vec in enumerate(data):
            min_dist = None
            closest_c_index = None
            for i_c, c in enumerate(codebook):
                d = get_mse(vec, c)
                if min_dist is None or d < min_dist:
                    min_dist = d
                    closest_c_list[i] = c
                    closest_c_index = i_c
            vecs_near_c[closest_c_index].append(vec)
            vec_idxs_near_c[closest_c_index].append(i)
        for i_c in range(len_codebook):
            vecs = vecs_near_c.get(i_c) or []
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                new_c = avg_all_vectors(vecs, _dim)
                codebook[i_c] = new_c
                for i in vec_idxs_near_c[i_c]:
                    closest_c_list[i] = new_c
                abs_weights[i_c] = num_vecs_near_c
                rel_weights[i_c] = num_vecs_near_c / _size_data
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        avg_dist = avg_codevector_dist(closest_c_list, data)
        err = (prev_avg_dist - avg_dist) / prev_avg_dist
        num_iter += 1

    return codebook, abs_weights, rel_weights, avg_dist

def avg_all_vectors(vecs, dim=None, size=None):
    size = size or len(vecs)
    nvec = np.array(vecs)
    nvec = nvec / size
    navg = np.sum(nvec, axis=0)
    return navg.tolist()

def new_codevector(c, e):
    nc = np.array(c)
    return (nc * (1.0 + e)).tolist()

def initial_avg_distortion(c0, data, size=None):
    size = size or _size_data
    nc = np.array(c0)
    nd = np.array(data)
    f = np.sum(((nc-nd)**2)/size)
    return f

def avg_codevector_dist(c_list, data, size=None):
    size = size or _size_data
    nc = np.array(c_list)
    nd = np.array(data)
    f = np.sum(((nc-nd)**2)/size)
    return f

def get_mse(a, b):
    na = np.array(a)
    nb = np.array(b)
    return np.sum((na-nb)**2)