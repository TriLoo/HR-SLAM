import mxnet as mx
import matplotlib as mpl
mpl.rcParams['figure.dpi']=120
from matplotlib import pyplot as plt


def box_to_rect(box, color, linewidth=3):
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth
    )



'''
To process the data of VOC2012
'''

