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

fname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min_restarts/sklinv0_2sec_108.nc'
longname = '0_1min'
shortname = 'CNTL'
tlvname = 'TLV2'
timetag = '108'
t0_1min = 108 #The end time in the 1min file to see if trajs end up with at least 0.1 s-1 of strmwise
trajdir = '/data/keeling/a/kevintg2/a/cm1/paper2_summer2021/offlinetrajectories_backward/BACKtrajs_0_2sec_0-2km_zFIX/'

umove  = 14.5
vmove  = 4.5
ifil   = '/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/'+longname+'i.txt'
jfil   = '/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/'+longname+'j.txt'
spdfil = '/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/'+longname+'spd.txt'
maxi  = np.loadtxt(ifil)
maxj  = np.loadtxt(jfil)
spds  = np.loadtxt(spdfil)
ci1   = spds[0]
cj1   = spds[1]
ci2   = spds[2]
cj2   = spds[3]
ci3   = spds[4]
cj3   = spds[5]
ci4   = spds[6]
cj4   = spds[7]

oldxpos = np.load(trajdir+timetag+'xpos.npy')
oldypos = np.load(trajdir+timetag+'ypos.npy')
oldzpos = np.load(trajdir+timetag+'zpos.npy')
oldzposhASL = np.load(trajdir+timetag+'zpos_heightASL.npy')
oldupos = np.load(trajdir+timetag+'uinterp.npy')
oldvpos = np.load(trajdir+timetag+'vinterp.npy')
oldwpos = np.load(trajdir+timetag+'winterp.npy')
oldxvortpos = np.load(trajdir+timetag+'xvort.npy')
oldyvortpos = np.load(trajdir+timetag+'yvort.npy')

ds = xr.open_dataset(fname)
time = ds.time
print('number of times in .nc file: ',len(time))

numxtraj = len(oldzposhASL[0,0,0,:])
numytraj = len(oldzposhASL[0,0,:,0])
numztraj = len(oldzposhASL[0,:,0,0])
numttraj = len(oldzposhASL[:,0,0,0])
print('numttraj: ',numttraj)

xpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
ypos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
zpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
zposhASL = np.zeros((numttraj, numztraj, numytraj, numxtraj))
upos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
vpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
wpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
xvortpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
yvortpos = np.zeros((numttraj, numztraj, numytraj, numxtraj))
icount = np.arange(numxtraj)
jcount = np.arange(numytraj)
kcount = np.arange(numztraj)
tcount = np.arange(numttraj)
numoftimes = len(tcount)
print('numoftimes, this should be same as numttraj: ',numoftimes)

for t in tcount:
    revt = numoftimes-t
    xpos[t,:,:,:] = oldxpos[revt-1,:,:,:]
    ypos[t,:,:,:] = oldypos[revt-1,:,:,:]
    zpos[t,:,:,:] = oldzpos[revt-1,:,:,:]
    zposhASL[t,:,:,:] = oldzposhASL[revt-1,:,:,:]
    upos[t,:,:,:] = oldupos[revt-1,:,:,:]
    vpos[t,:,:,:] = oldvpos[revt-1,:,:,:]
    wpos[t,:,:,:] = oldwpos[revt-1,:,:,:]
    xvortpos[t,:,:,:] = oldxvortpos[revt-1,:,:,:]
    yvortpos[t,:,:,:] = oldyvortpos[revt-1,:,:,:]
#Get only trajectories that stay above lowest 3 scalar levels
filtertrajs = np.zeros((1, numztraj, numytraj, numxtraj))
icount = np.arange(numxtraj)
jcount = np.arange(numytraj)
kcount = np.arange(numztraj)
totalcount = 0
belowcount = 0
strmwisecount = 0
for i in icount:
    for j in jcount:
        for k in kcount:
            totalcount = totalcount+1
            zforchecking = zposhASL[:,k,j,i]
            heightflag = 1
            for kcheck in zforchecking:
                if (kcheck < 12.5) or (math.isnan(kcheck)):
                    heightflag = 0
            if heightflag == 0:
                belowcount = belowcount+1
            filtertrajs[0,k,j,i] = heightflag
print('total trajectories: ',totalcount)
print('number of trajs above 12.5 m (z=2): ',totalcount-belowcount)
#Get only trajectories that are in the SVC, so strmwise > 0.1 s-1
for i in icount:
    for j in jcount:
        for k in kcount:
            heightflag = 0
            if filtertrajs[0,k,j,i] == 1:
                xvort = xvortpos[-1,k,j,i]
                yvort = yvortpos[-1,k,j,i]
                u = upos[-1,k,j,i]
                v = vpos[-1,k,j,i]
                if (t0_1min <= 60):
                    vsru = u-(ci1-umove)
                    vsrv = v-(cj1-vmove)
                if (t0_1min > 60 and t0_1min <= 120):
                    vsru = u-(ci2-umove)
                    vsrv = v-(cj2-vmove)
                if (t0_1min > 120 and t0_1min <= 180):
                    vsru = u-(ci3-umove)
                    vsrv = v-(cj3-vmove)
                if (t0_1min > 180 and t0_1min<=240):
                    vsru = u-(ci4-umove)
                    vsrv = v-(cj4-vmove)
                strmvort = ((xvort*vsru)+(yvort*vsrv))/(((vsru**2)+(vsrv**2))**0.5)
                if strmvort >= 0.1:
                    heightflag = 1
                    strmwisecount = strmwisecount+1
            filtertrajs[0,k,j,i] = heightflag
print('number of trajs meeting streamwise threshold: ',strmwisecount)

np.save(shortname+'_'+tlvname+'_'+timetag+'filtertrajs_lowest12.5', filtertrajs)
