#!/usr/bin/env python3
import os
import sys

import numpy as np
from idx import write_idx_file


def npz2idx(filename, model_dir):
    d = np.load(filename)

    for name in d:
        v = d[name]
        print('%24s :: %s%s' % (name, v.dtype, v.shape))
        if name.endswith(':0'):
            name = name[:-2]
        full_path = os.path.join(model_dir, name + '.idx')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        write_idx_file(full_path, v)


home = os.getenv('HOME')
model_dir = os.path.join(home, 'var/models/openpose')
fin = os.path.join(model_dir, 'hao28-pose600000.npz')
npz2idx(fin, model_dir)
