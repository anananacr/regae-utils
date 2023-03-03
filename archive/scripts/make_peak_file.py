import numpy as np

file_path='/asap3/fs-bmx/gpfs/regae/2022/data/11015323/processed/calib/20220524_Au30_peak_0_1.txt'
x=np.arange(0,164,1)

f=open(file_path,'w')
f.write('index min max')

for i in x:
    if i%2==0:
        f.write(f'\n{i} 35 45')
    else:
        f.write(f'\n{i} 60 70')
f.close()
