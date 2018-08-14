# -*- coding : utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.14'


import argparse
import readMat
import detection
import showDetectionResults
import joblib
import mxnet as mx


def parseAugs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_file', default='Indian_pines_corrected.mat')
    parse.add_argument('--cls_num', type=int, default=9)
    parse.add_argument('--net_params', default='ResNet3D.params')
    parse.add_argument('--ctx', default=mx.cpu())
    parse.add_argument('--width', type=int, default=9)
    parse.add_argument('--height', type=int, default=9)
    parse.add_argument('--mavs', default='mavs.joblib')
    parse.add_argument('--dists', default='dists.joblib')
    parse.add_argument('--show_om', type=bool, default=False)
    parse.add_argument('--show_sm', type=bool, default=False)
    parse.add_argument('--show_both', type=bool, default=True)
    args = parse.parse_args()
    return args


def detection_unknown_top(args):
    # Prepare data
    raw_data = readMat.readMatFile(args.img_file)
    mavs = joblib.load(args.mavs)
    dists = joblib.load(args.dists)
    splited_data = readMat.generate_child_window(raw_data, args.width, args.height)
    boxes = splited_data['Boxes']
    # Prediction
    om, sm = detection.detection_om_sm(args.cls_num, args.net_params, args.ctx, splited_data, mavs, dists, is_save=False)
    # Show Results
    img = raw_data[:, :, :1]
    showDetectionResults.showResults(img, om, sm, boxes, show_both=args.show_both, show_om=args.show_om, show_sm=args.show_sm)
    print('Done.')


if __name__ == '__main__':
    args = parseAugs()
    detection_unknown_top(args)
