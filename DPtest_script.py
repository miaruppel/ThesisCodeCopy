## -- DP recon, python -- ##
## -- RPT, 9/19/2023, robby is really cool all the time
import os
os.chdir('C:/PIRL/Recon/')
parent_dir = os.getcwd()
import winsound
import time
import mapvbvd
import numpy as np
from scipy.interpolate import griddata
import scipy.io
import concurrent.futures
def alarm():
    winsound.Beep(784, 70);winsound.Beep(740, 70);winsound.Beep(622, 70);winsound.Beep(440, 70);winsound.Beep(415, 70);winsound.Beep(659, 70);winsound.Beep(830, 70);winsound.Beep(1046, 70);

## -- Set file directory as working directory and open the traj file -- ##
parent_dir = os.getcwd()
mat = scipy.io.loadmat(os.path.join(parent_dir,r'traj\traj_dis_afia.mat'))
traj = mat['data']

def split(a, n):
    k, m = divmod(a.shape[0], n)
    A = [a[i*k+min(i, m):(i+1)*k+min(i+1, m),:] for i in range(n)]
    return A

## -- For DP Recon -- ##
twix_obj = mapvbvd.mapVBVD(os.path.join(parent_dir,r'data\meas_MID00205_FID12765_7_xe_radial_Dixon_2201_DP.dat')) # Load a DP dataset (This one from JM sequence)
twix_obj.image.squeeze = True #-------Remove singleton dimensions from data array
twix_obj.image.flagIgnoreSeg = True # Ignore the 'Seg' Dimension which
twix_obj.image.flagRemoveOS = False # Keep the oversampled data (all 64 points)
raw_fids = np.transpose(twix_obj.image.unsorted().astype(np.cdouble)) # -- This is a 2030x64 array for JM sequence
data_gas = raw_fids[:-30][0::2, :] # -- The 'gas' data is every other line beginning at row 0
data_dis = raw_fids[:-30][1::2, :] # -- The 'dis' data is every other line beginning at row 1


N = 32 # defines the recon array size NxNxN
s = 0.8 # this scales the kSpace trajectory locations within kSpace
x = traj[:,:,0].flatten()
y = traj[:,:,1].flatten()
z = traj[:,:,2].flatten()
trajlist = np.column_stack((x,y,z))*N*s
mg = np.linspace(-N/2+1,N/2,N)
X,Y,Z = np.meshgrid(mg,mg,mg,indexing='ij')
castlist = np.column_stack((X.flatten(),Y.flatten(),Z.flatten()))


## -- MRIdata is a 2D array where columns are [0=kx, 1=ky, 2=kz, 3=gasdata, 4=DPdata]
MRIdata = np.column_stack((trajlist,data_gas.flatten(),data_dis.flatten()))

## -- Recon by Interpolation -- ##
starttime = time.time()
realgas = griddata(trajlist,np.real(MRIdata[:,3]),castlist)
imaggas = griddata(trajlist,np.imag(MRIdata[:,3]),castlist)
realdis = griddata(trajlist,np.real(MRIdata[:,4]),castlist)
imagdis = griddata(trajlist,np.imag(MRIdata[:,4]),castlist)

gas = [complex(realgas[i], imaggas[i]) for i in range(len(realgas))]
dis = [complex(realdis[i], imagdis[i]) for i in range(len(realdis))]

## -- interpData is a 2D array where columns are [0=kx, 1=ky, 2=kz, 3=gasdata, 4=DPdata] (interpolated to Cartesian grid)
interpData = np.column_stack((castlist,gas,dis))
#np.savez(os.path.join(parent_dir,r'reconData32.mat'), MRIdata=MRIdata, interpData=interpData)
np.save(os.path.join(parent_dir,r'reconData32'), interpData)
print(f'\n ## -- Interpolation time: {np.round((time.time() - starttime)/60,2)} min -- ##\n')
alarm()

## -- Recon by Interpolation (parallel real/imag) 
starttime = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    f1 = executor.submit(griddata, trajlist,np.real(MRIdata[:,3]),castlist)
    f2 = executor.submit(griddata, trajlist,np.imag(MRIdata[:,3]),castlist)
    f3 = executor.submit(griddata, trajlist,np.real(MRIdata[:,4]),castlist)
    f4 = executor.submit(griddata, trajlist,np.imag(MRIdata[:,4]),castlist)
    realgas = f1.result()
    imaggas = f2.result()
    realdis = f3.result()
    imagdis = f4.result()
gas = [complex(realgas[i], imaggas[i]) for i in range(len(realgas))]
dis = [complex(realdis[i], imagdis[i]) for i in range(len(realdis))]
interpData = np.column_stack((castlist,gas,dis))
#np.savez(os.path.join(parent_dir,r'reconData32_speed.mat'), MRIdata=MRIdata, interpData=interpData)
np.save(os.path.join(parent_dir,r'reconData32_speed'), interpData)
print(f'\n ## -- Parallel interpolation time: {np.round((time.time() - starttime)/60,2)} min -- ##\n')
alarm()





## -- Recont by Interpolation (parallel, multi-thread)
realz = [];imagz = [];
splitCastList = split(castlist, 8)
starttime = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    realResults = [executor.submit(griddata, trajlist, realgas, splitCastList[k]) for k in range(0,8)]
    for f1 in concurrent.futures.as_completed(realResults):
        realzgas.append(f1.result())

    imagResults = [executor.submit(griddata, trajlist,imagData,castlist[(0:65535)+65536*k,:]) for k in range(0,4)]
    for f2 in concurrent.futures.as_completed(imagResults):
        imagz.append(f2.result())

print(time.time() - starttime)

gas = [complex(realgas[i], imaggas[i]) for i in range(len(realgas))]
dis = [complex(realdis[i], imagdis[i]) for i in range(len(realdis))]
interpData = np.column_stack((castlist,gas,dis))
np.savez(os.path.join(parent_dir,r'reconData.mat'), MRIdata=MRIdata, interpData=interpData)




