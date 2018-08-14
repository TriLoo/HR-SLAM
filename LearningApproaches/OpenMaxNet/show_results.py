# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.20'

import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as scio
import joblib
import argparse


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--pred_file', default='result_file.joblib')
    parse.add_argument('--gt_file', default='Indian_pines_gt.mat')
    args = parse.parse_args()
    preds = joblib.load(args.pred_file)
    gts = scio.loadmat(args.gt_file)['indian_pines_gt']
    preds_img = np.ones((145, 145)) * 16

    for i in range(16):
        for j in range(16):
            idx = i * 16 + j
            preds_img[(i * 9):(i + 1) * 9, (j * 9):(j + 1) * 9] = np.argmax(preds[idx].asnumpy())
            print('class = ', np.argmax(preds[idx].asnumpy()))
            #print('label = ', gts[4][5])

    plt.imshow(gts)
    #plt.figure()
    #plt.imshow(preds_img)
    plt.show()


if __name__ == '__main__':
    main()

