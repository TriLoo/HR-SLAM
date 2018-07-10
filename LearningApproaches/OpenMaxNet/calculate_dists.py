# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.10'

import argparse
import joblib

import numpy as np
import scipy.spatial.distance as spd


# mavs: shape = (1, class_num); features: shape = (N_c, 1, class_num)
def compute_channel_distance(mavs, features, eu_weight):
    eucos_dists, eu_dists, cos_dists = [], [], []

    for channel, mav in enumerate(mavs):
        # mav's shape : (class_num, )
        # feat's shape: (1, class_num)
        # feat[0]'s shape: (class_num,)
        # eu_dists' shape: (N_c)
        eu_dists.append([spd.euclidean(mav, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mav, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mav, feat[channel]) * eu_weight + spd.cosine(mav, feat[channel]) for feat in features])

    return {'eucos':np.array(eucos_dists), 'cosine':np.array(cos_dists), 'euclidean':np.array(eu_dists)}


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dists', default='dists.joblib')
    parse.add_argument('--eu_weight', type=float, default=5e-3)
    parse.add_argument('--scores', default='scores.joblib', help='the input scores.joblib file')
    parse.add_argument('--mavs', default='mavs.joblib', help='the input mavs.joblib file')
    args = parse.parse_args()

    # scores shape: (class_num, N_c, 1, class_num)
    # mavs shape  : (class_num, 1, class_num)
    scores = joblib.load(args.scores)
    mavs = joblib.load(args.mavs)

    # dists shape : (class_num, 3, 1, N_c), the 3 represent the 3 different type distances, verified
    dists = [compute_channel_distance(mav, score, args.eu_weight) for mav, score in zip(mavs, scores)]
    #print('lenght of dists: ', len(dists))
    #print('shape of dists elements: ', dists[0]['eucos'].shape)

    joblib.dump(dists, args.dists)

    print('Distance Saved Successful.')


if __name__ == '__main__':
    main()

