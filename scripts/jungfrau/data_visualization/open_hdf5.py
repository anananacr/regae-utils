import h5py
import numpy as np
import matplotlib.pyplot as plt
f = h5py.File('/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/231020_mos_c3_ms_001/ed_rot_step_003/231020_mos_c3_ms_001_003_master_9983.h5', "r")
data= np.array(f['data'])
plt.imshow(data[0], vmin=0,vmax=400)
plt.colorbar()
plt.show()
