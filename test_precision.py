#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error

import cv2
import pathlib
import numpy as np
import pandas as pd
from model.utils import Param, arg


if __name__ == "__main__":
    args = arg()
    config_path = args.cfg
    params = Param(config_path)
    num = 1

    logdir = os.path.join(config_path, "emb/")
    vecs = np.loadtxt(os.path.join(logdir, f"cv-{args.epoch:02d}vecs.tsv"),
                      delimiter='\t')
    metas = np.loadtxt(os.path.join(logdir, f"cv-{args.epoch:02d}metas.tsv"),
                      delimiter='\t')
    np.savetxt(os.path.join(logdir, f"cv-{num}f-{args.epoch:02d}vecs.tsv"),
               vecs,
               fmt=f"%.{num}f",
               delimiter='\t')
    np.savetxt(os.path.join(logdir, f"cv-{num}f-{args.epoch:02d}metas.tsv"),
               metas,
               fmt='%i',
               delimiter='\t')
