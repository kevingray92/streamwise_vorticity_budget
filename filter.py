import numpy as np
import re
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator

fname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min_restarts/sklinv0_2sec.nc'
stormspd = []
stormspd = np.genfromtxt('/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/0_1minspdavg.txt', dtype=None)
timetag = '057'
t0 = 57
#thrhoprimepdf = 'Underupdraft_0-2km_CNTL_TLV1_'+timetag+'min_thrhoprimeavgtimeseries_forw.pdf'
forwtrajdir = '../offlinetrajectories_forward/FORWtrajs_0_2sec_0-2km_zFIX/'

forwxpos = np.load(forwtrajdir+timetag+'xpos.npy')
forwypos = np.load(forwtrajdir+timetag+'ypos.npy')
forwzpos = np.load(forwtrajdir+timetag+'zpos.npy')
forwzposhASL = np.load(forwtrajdir+timetag+'zpos_heightASL.npy')
forwupos = np.load(forwtrajdir+timetag+'uinterp.npy')
forwvpos = np.load(forwtrajdir+timetag+'vinterp.npy')
forwwpos = np.load(forwtrajdir+timetag+'winterp.npy')
forwxvortpos = np.load(forwtrajdir+timetag+'xvort.npy')
forwyvortpos = np.load(forwtrajdir+timetag+'yvort.npy')
forwzvortpos = np.load(forwtrajdir+timetag+'zvort.npy')

numxtraj = len(forwzposhASL[0,0,0,:])
numytraj = len(forwzposhASL[0,0,:,0])
numztraj = len(forwzposhASL[0,:,0,0])
numttraj = len(forwzposhASL[:,0,0,0])
print('numx: ',numxtraj)
print('numy: ',numytraj)
print('numz: ',numztraj)
print('numt: ',numttraj)

print('shape of forwzposhASL: ',forwzposhASL.shape)
ds = xr.open_dataset(fname)
xh = ds.xh
yh = ds.yh
zh = ds.z
w   = ds.winterp
xgrid = np.arange(0,len(xh))
ygrid = np.arange(0,len(yh))
filtertrajs = np.zeros((1, numztraj, numytraj, numxtraj))
howmanytimes = forwxpos[:,0,0,0]
timecount = np.arange(len(howmanytimes))
icount = np.arange(numxtraj)
jcount = np.arange(numytraj)
kcount = np.arange(numztraj)
for i in icount:
    print('i: ',i)
    for j in jcount:
        print('j: ',j)
        for k in kcount:
            zforchecking = forwzposhASL[:,k,j,i]
            x = forwxpos[:,(k),(j),(i)]
            y = forwypos[:,(k),(j),(i)]
            updraftflag = 0
            nanflag = 0
            for it in timecount:
                t = t0+(it+1)
                if (math.isnan(y[it])) or (math.isnan(x[it])):
                    w1kmtraj = 0.0
                else:
                    w1kmtraj = w[t,30,int(round(y[it])),int(round(x[it]))]
                if w1kmtraj > 15.0:
                    updraftflag = 1
            if updraftflag == 1:
                for kcheck in zforchecking:
                    if (kcheck < 12.5) or (math.isnan(kcheck)):
                        updraftflag = 0
                        if math.isnan(kcheck):
                            nanflag =1
                            print('i: ',i)
                            print('j: ',j)
                            print('k: ',k)
            filtertrajs[0,k,j,i] = updraftflag
np.save(forwtrajdir+timetag+'filtertrajs', filtertrajs)
