# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.10'


import argparse
import joblib

import numpy as np

from OpenMaxLayer import fit_weibull, openmax


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mavs', default='mavs.joblib')
    parse.add_argument('--dists', default='dists.joblib')
    parse.add_argument('--scores', default='scores.joblib')
    parse.add_argument('--alpha', type=int, default=5)
    parse.add_argument('--tailsize', type=int, default=20)
    parse.add_argument('--distances_type', default='eucos')
    parse.add_argument('--euc_scale', type=float, default=5e-3)
    parse.add_argument('--threshold', type=float, default=0.0)
    parse.add_argument('--save', default='.', help='Directory to save the results.')
    args = parse.parse_args()

    categories = [i for i in range(9)]

    mavs = joblib.load(args.mavs)
    dists = joblib.load(args.dists)
    #scores, labels = joblib.load(args.scores)
    scores = joblib.load(args.scores)

    # 得到每一类的weibull model，
    weibull_model = fit_weibull(mavs, dists, categories, args.tailsize, args.distances_type)

    pred_y, pred_y_o = [], []

    #for score in scores[0]:
    for i, score in enumerate(scores[2]):    # for temp test
        #so, ss = openmax(weibull_model, categories, score, args.euc_scale, args.alpha, args.distances_type)
        so, ss = openmax(weibull_model, categories, score, args.euc_scale, 3, args.distances_type)
        pred_y.append(np.argmax(ss) if np.max(ss) >= args.threshold else 10)
        pred_y_o.append(np.argmax(so) if np.max(so) >= args.threshold else 10)
        print('value of softmax[0]: ', ss[0])
        print('value of softmax[2]: ', ss[2])
        print('value of openmax[0]: ', so[0])
        print('value of openmax[2]: ', so[2])
        print('value of openmax[100]: ', so[9])

        #print('lenght of pred_y: ', len(pred_y))
        #print('lenght of pred_y_o: ', len(pred_y_o))
        print('pred_y[-1]: ', pred_y[-1])
        print('pred_y_o[-1]: ', pred_y_o[-1])

        if i == 10:
            break

    print('Calculation Finished.')


if __name__ == '__main__':
    main()

