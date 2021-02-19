#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from sklearn import manifold
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from model.utils import arg, Param


if __name__ == '__main__':
    args = arg()
    config_path = args.cfg
    params = Param(config_path)

    vecs = np.loadtxt(os.path.join(config_path, f"emb/{args.epoch}vecs.tsv"),
                      dtype=np.float,
                      delimiter='\t')
    tsne = manifold.TSNE(n_components=3,
                         learning_rate=100,
                         n_iter=350)
    out = tsne.fit_transform(vecs)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(18):
        ax.scatter(out[i*100:(i+1)*100, 0],
                   out[i*100:(i+1)*100, 1],
                   out[i*100:(i+1)*100, 2])
    plt.show()
