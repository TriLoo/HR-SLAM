# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.12'

import cv2
import argparse


def center2corner(center_x, center_y, width, height):
    half_width = width >> 1
    half_height = height >> 1
    top_left_x = center_x - half_width
    top_left_y = center_y - half_height
    bottom_right_x = center_x + half_width
    bottom_right_y = center_y + half_height
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)


def drawResults(cls, img, clss, locs):
    for cls, loc in zip(clss, locs):
        pt1, pt2 = center2corner(loc[0], loc[1], 9, 9)
        cv2.rectangle(img, pt1, pt2)
        cv2.addText(img, cls)  # TODO: add text

    return img


def parseAugs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img', default='test.jpg')
    parse.add_argument('--clss')
    parse.add_argument('--locs')
    parse.add_argument('--show_om', type=bool, default=True)
    parse.add_argument('--show_sm', type=bool, default=True)
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parseAugs()
    img = drawResults(args.img, args.clss, args.locs)
    if args.show_om:
        cv2.imshow('Detection Result (openmax)', img)
    if args.show_sm:
        cv2.imshow('Detection Result (softmax)', img)
    cv2.waitKey()

