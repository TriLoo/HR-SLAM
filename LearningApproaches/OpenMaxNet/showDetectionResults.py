# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.12'

import cv2

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (128, 0, 128),    # order: BGR
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                (64, 128, 0), (0, 0, 255)]


def center2corner(loc, width, height):
    center_x, center_y = loc
    half_width = width >> 1
    half_height = height >> 1
    top_left_x = center_x - half_width
    if top_left_x < 0:
        top_left_x = 0
    top_left_y = center_y - half_height
    if top_left_y < 0:
        top_left_y = 0
    bottom_right_x = center_x + half_width
    bottom_right_y = center_y + half_height
    return (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y))


def drawResults(img, cls, loc, wid, hei):
    top_left, bottom_right = center2corner(loc, wid, hei)
    # add rectangle
    cv2.rectangle(img, top_left, bottom_right, color=class_colors[cls])
    # add text info
    bottom_left = (int(loc[0]-(wid >> 1)), int(loc[1]+(hei >> 1)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 0)
    cv2.putText(img, classes[cls], bottom_left, fontFace=font, fontScale=0.2, color=text_color, thickness=1, lineType=cv2.LINE_AA)
    return img


def showResults(img, om, sm, boxes, bg = 5,name_om='OpenMax', name_sm='SoftMax', show_both=True, show_om=False, show_sm=False):
    img_om = img.copy()
    img_sm = img.copy()
    for o, s, b in zip(om, sm, boxes):
        loc, wid, hei = b
        if s == bg:
            continue
        if show_both:
            drawResults(img_om, o, loc, wid, hei)
            drawResults(img_sm, s, loc, wid, hei)
        elif show_om:
            drawResults(img_om, o, loc, wid, hei)
        elif show_sm:
            drawResults(img_sm, s, loc, wid, hei)
        else:
            break
    if show_both:
        cv2.imshow(name_om, img_om)
        cv2.imshow(name_sm, img_sm)
        print('Press any key to close all images.')
    elif show_om:
        cv2.imshow(name_om, img_om)
        print('Press any key to close all images.')
    elif show_sm:
        cv2.imshow(name_sm, img_sm)
        print('Press any key to close all images.')
    else:
        print('No data will be shown.')

    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread('lena.jpg')
    drawResults(img, 9, (100, 100), 9, 9)
    drawResults(img, 9, (50, 50), 8, 8)
    cv2.imshow('result', img)
    cv2.waitKey()
