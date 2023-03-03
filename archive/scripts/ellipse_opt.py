import matplotlib.colors as color
import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
from skued import azimuthal_average
from ellipse import fit_ellipse, cart_to_pol,get_ellipse_pts
from ellipse_cut import polar_to_cart_lst
"""
    NOT SUPPORTED! ATTEMPT TO ELLIPSE CORRECTION!
"""

def cost(p):
    el_rat=p[0]
    el_ang=p[1]
    corr_data=utils.correct_ell(data.copy(),el_rat,el_ang)

    peak_positions=[]
    angle=[]
    intensity=[]
    for i in range(19,359,1):
        rad_signal_masked=azimuthal_average(corr_data, center=(515,532),angular_bounds=(i-19,i+1),trim=True)
        rad_signal_cut=[]
        rad_signal_cut.append(np.array(rad_signal_masked[0][10:]))
        rad_signal_cut.append(np.array(rad_signal_masked[1][10:]))
        baseline=utils.baseline_als(rad_signal_cut[1],1e3, 0.1)
        rad_sub=rad_signal_cut[1]-baseline
        rad_sub[np.where(rad_sub<0)]=0
        #plt.plot(rad_sub)
        #plt.show()

        peaks, half=utils.calc_fwhm(rad_sub,-1,threshold=500, height=1, width=3, distance=5)
        intensity=[]
        for j in range(len(peaks)):
            intensity.append(rad_sub[peaks[j]])
        #plt.plot(rad_sub)
        #plt.scatter(peaks,intensity,c='r')
        #plt.show()

        peak_px=list(peaks+rad_signal_cut[0][0])
        for j in peak_px:
            angle.append(i+1)
            peak_positions.append(j)

    rad_range = (25, 100)
    powder_polar = np.histogram2d(peak_positions, angle, bins=[np.linspace(*rad_range, 200), np.linspace(0, 360, 72)])
    return np.mean(powder_polar[0]**2)/np.mean(powder_polar[0] * np.roll(powder_polar[0], powder_polar[0].shape[1]//4, axis=1)) - 1
input='path_to_file'
file_path=f'{input}'
hf = h5py.File(file_path, 'r')
data_name='sum_frames_mask'
global data
data = np.array(hf[data_name])

#el_rat=np.arange(0.5,2,0.05)
el_rat=[1]
#el_ang=np.arange(30,180,5)
el_ang=0
for i in el_rat:
    print(f'el_ang:0 el_rat:{i}')
    angles,peak_rads=utils.check_ellipticity(data,i,el_ang)

'''
angles = np.arange(18, 28, 1)
ratio = np.arange(0.998, 1.008, 0.0005)

from concurrent.futures import ProcessPoolExecutor
X, Y = np.meshgrid(ratio, angles)
with ProcessPoolExecutor() as exc:
    foms = exc.map(cost, zip(X.ravel(), Y.ravel()))
foms = np.array(list(foms)).reshape(X.shape)

# show result
plt.figure()
plt.pcolormesh(ratio, angles, foms)
plt.colorbar()
plt.ylabel('Elliptical axis')
plt.xlabel('Ellipticity')
plt.title('Ellipticity correction')
plt.show()

'''


plt.close('all')
fh, ax = plt.subplots(1, 1, figsize=(5,5))
#print(powder_polar)
#ax.pcolormesh(powder_polar[2][:-1], powder_polar[1][:-1], powder_polar[0])
#ax.set_xlabel('Azimuth (deg)')
#ax.set_ylabel('Corrected radius')
#plt.show()

ang_0=[]
peak_0=[]
for i in range(len(angles)):
    if 55<peak_rads[i]<60:
        ang_0.append(angles[i])
        peak_0.append(peak_rads[i])

plt.scatter(ang_0,peak_0,marker='.',color='r')
ax.set_ylim(40,120)
plt.show()

x0,y0=polar_to_cart_lst(peak_0,ang_0)
x=np.array(x0)
y=np.array(y0)

coeffs = fit_ellipse(x, y)
print('Fitted parameters:')
print('a, b, c, d, e, f =', coeffs)
x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)

fig, ax = plt.subplots(1,1, figsize=(15, 10))
ax.imshow(data, cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,1e5))
x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
plt.plot(x, y,color='magenta')
plt.show()
