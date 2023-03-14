"""
Function that plots intensity deviation in a ring for Au powder peaks according to the angle, takes the running average of 6 points.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

fh, ax = plt.subplots(1, 1, figsize=(8, 8))

# increase size to plot together different peaks
size = [0]
label = ["360deg"]
count = 0
peak = 0

for j in size:
    ## Set path to data
    file_path = f"path/to/file"
    hf = h5py.File(file_path, "r")
    data_name = f"ring_deviation_{j}"
    rad_0 = np.array(hf[data_name])
    x = np.array(hf[f"ring_deviation_deg_{j}"])
    hf.close()

    pair = list(zip(x, rad_0))
    pair = sorted(pair, key=lambda x: x[0])
    mov_average_int = []
    mov_average_ang = []
    acc_int = 0
    acc_ang = 0
    for idx, i in enumerate(pair):

        if idx % 6 == 0:
            if idx == 6:
                print(acc_int, acc_ang, pair[0:6])
            mov_average_int.append(acc_int / 6)
            mov_average_ang.append(acc_ang / 6)
            acc_int = 0
            acc_ang = 0
        else:
            acc_int += i[1]
            acc_ang += i[0]
    print(len(mov_average_ang), len(mov_average_int))
    plt.scatter(mov_average_ang[1:], mov_average_int[1:], label=f"Peak {peak}")
    mean = (np.mean(mov_average_int)) * np.ones(300)
    # print(mean)
    plt.plot(mean)

    count += 1

ax.set_xlabel("$\Theta$ [pixel]")
ax.set_ylabel("Intensity [keV]")
ax.grid()
# textstr=f'Mean:{round(np.mean(rad_0),2)}\nMedian:{round(np.median(rad_0),2)}\nStd:{round(np.std(rad_0),2)}\nVar:{round(np.var(rad_0),2)}'
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,verticalalignment='top')
# print(textstr)
ax.set_ylim(5e3, 3e4)
plt.legend()
# plt.savefig('../proc/plot/20220404_ni_0_scan_dev.png')
plt.show()
