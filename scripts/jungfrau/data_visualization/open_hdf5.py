import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File(
    "/asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/231020_mos_c3_ms_003_pad/231020_mos_c3_ms_003_pad_459.h5",
    "r",
)
data = np.array(f["data"])
plt.imshow(data[0], vmin=0, vmax=100)
plt.colorbar()
plt.show()
