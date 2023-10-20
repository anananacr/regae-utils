#!/usr/bin/env python3.7

import h5py
import numpy as np

root='/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/darks'
f = h5py.File(f'{root}/pedal_d1_231020_mos_c3_ms_003_start.h5', "r")
g = h5py.File(f'{root}/pedal_d1_231020_mos_c3_ms_003_stop.h5', "r")
output=h5py.File(f'{root}/pedal_d1_231020_mos_c3_ms_003_average.h5', "w")
for key in f.keys():
    data=(np.array(f[key])+np.array(g[key]))//2
    output.create_dataset(key, data=data)
f.close()
g.close()
output.close()
