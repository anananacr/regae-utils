import h5py
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2, figsize=(10, 5))

size=['no','com','cc']
data_name='intensity'
data_name_0='fwhm_radius'
color=['r','g','b']
count=0

for k in range(1):
    for i in size:
        x=[i,i,i]
        file_path=f'../proc/centered/20220506/20220506_Au_30nm_4_{i}.h5'
        hf = h5py.File(file_path, 'r')
        rad_0 = np.array(hf[data_name])
        rad_1 = np.array(hf[data_name_0])
        print(x,rad_0,rad_1)

        ax1=plt.subplot(121)
        for j in range(len(color)):
            if count==0:
                ax1.scatter(x[j],rad_0[j], color=color[j],label =f'peak_{j}',marker='.')
            else:
                ax1.scatter(x[j],rad_0[j], color=color[j],marker='.')
        ax2=plt.subplot(122)
        for j in range(len(color)):
            if count==0:
               ax2.scatter(x[j],rad_1[j], color=color[j],label=f'peak_{j}',marker='.')
            else:
               ax2.scatter(x[j],rad_1[j], color=color[j],marker='.')
        hf.close()
        count+=1

ax1.legend()
#ax2.legend()
ax1.set_xlabel('Sol67 current (A)')
ax1.set_ylabel('Intensity [keV]')
#ax1.set_xlim(2.4,4.0)
#ax2.set_xlim(2.4,4.0)
ax2.set_xlabel('Sol67 current (A)')
ax2.set_ylabel('FWHM/Radius')
ax1.grid()
ax2.grid()
plt.savefig('../proc/plot/20220506_Au_30nm_4_methods_compare.png')
plt.show()
