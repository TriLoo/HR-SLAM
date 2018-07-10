# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.10'

import numpy as np
import scipy.spatial.distance as spd
import libmr

# query_score shape: (1, class_num)
# mav shape        : (1, class_num)
def calc_distance(query_score, mav, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mav, query_score) * eu_weight + spd.cosine(mav, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mav, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mav, query_score)
    else:
        print('Distance type not known: only eucos, euclidean, cosine supported')

    return query_distance


# mavs'shape = (class_num, 1, class_num)
# dists'shape = (class_num, 3, N_c)
def fit_weibull(mavs, dists, categories, tailsize=20, distance_type='eucos'):
    weibull_model = {}    # a dictionary data structure

    # mav's shape (1, class_num)
    # dist's shape (3, 1, N_c)
    for mav, dist, category_name in zip(mavs, dists, categories):
        weibull_model[category_name] = {}  # one weibull model per category
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mav   # shape=(1, class_num)
        weibull_model[category_name]['weibull_model'] = []

        for channel in range(mav.shape[0]):
            mr = libmr.MR()
            # default sort axis = -1, the last axis
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fithigh(tailtofit, len(tailtofit))
            weibull_model[category_name][weibull_model].append(mr)   # one mr model per channel per category

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [
        weibull_model[category_name]['mean_vec'],
        weibull_model[category_name]['distances_{}'.format(distance_type)],
        weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    scores = np.means(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]

    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_scores, eu_weight, alpha=10, distance_type='eucos'):
    nb_classes = len(categories)

    ranked_list = input_scores.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []

    # input_score_channel shape: (class_nums, )
    for channel, input_score_channel in enumerate(input_scores):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)   # (channels=1, 100)
    scores_u = np.asarray(scores_u)  # (channels=1, 100)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_scores.ravel()))

    return openmax_prob, softmax_prob

