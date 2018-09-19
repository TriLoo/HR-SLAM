# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.09.19'


import numpy as np
import graphSeg
import skimage.feature
import readRaw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _generate_segments(im_orig, scale, sigma, min_size):
    h, w, c = im_orig.shape
    im_mask = graphSeg.segment_hs(im_orig, sigma, scale, min_size)
    print('shape of im_mask: ', im_mask.shape)
    im_orig = np.transpose(im_orig, axes=(1, 2, 0))
    im_orig = np.append(im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, -1] = im_mask

    return im_orig


def _sim_texture(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_texture_gradient(img):
    ret = np.zeros(img.shape[0], img.shape[1], img.shape[2])

    for colour_channel in range(img.shape[2]):
        # local_binary_pattern: gray scale and rotation invariant LBP
        # LBP is an invariant descriptor that can be used for texture classification
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0
        )

    return ret


def _calc_texture_hist(img):
    BINS = 10
    hist = np.array([])
    for colour_channel in range(img.shape[2]):
        fd = img[:, colour_channel]

        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    hist = hist / len(img)

    return hist


def _extract_region(img):
    R = {}

    # pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, spectum in enumerate(i):
            #band = spectum[:, :-1]
            l = spectum[:, -1]

            if l not in R:
                R[l] = {"min_x":0xffff, "min_y":0xffff,
                        "max_x":0, "max_y":0, "labels":[l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)
    for k, v in R.items():
        masked_pixels = img[:, :, :-1][img[:, :, -1] == k]
        R[k]["size"] = len(masked_pixels / 4)
        # Only calculate texture histogram, No colour histogram is used here
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, -1] == k])

    return R


def _extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur+1 :]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_t": (
                          r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=500):
    img = _generate_segments(im_orig, scale, sigma, min_size)
    print('img shape: ', img.shape)
    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_region(img)

    neighbours = _extract_neighbours(R)

    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    while S != {}:
        i, j = sorted(S.items(), key=lambda i : i[1])[-1][0]
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        key_to_delete = []
        # mark similarities for regions to be removed
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions


def ss_test():
    im_orig = readRaw.read_raw_data()
    print('im_orig shape: ', im_orig.shape)
    img_lbl, regions = selective_search(im_orig)
    candidatas = set()
    for r in regions:
        if r['rect'] in candidatas:
            continue
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w/h > 1.2 or h/w > 1.2:
            continue

        candidatas.add(r['rect'])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = im_orig[0:3, :, :]
    img = np.transpose(img, axes=(1, 2, 0))
    ax.imshow(img)
    for x, y, w, h in candidatas:
        print(x, y, w, h)
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red',
                                  linewidth=1)
        ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    ss_test()


