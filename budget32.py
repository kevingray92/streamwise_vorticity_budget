print('test1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import re
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
print('test2')

###***This passes Kevin's check 07/28/21
#Set up file names and directory pathways
casename = '0_2sec_108'
shortname = 'CNTL_2sec_TLV2'
tlvname = 'TLV2'
timetag = '108'
timetagint = 108
fname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min_restarts/sklinv'+casename+'.nc'
bsfname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min/sklinv0_1min.nc'
stormspd = []
stormspd = np.genfromtxt('/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/0_1minspdavg.txt', dtype=None)
budgetpdf = shortname+'_'+tlvname+'_'+timetag+'min_strmwisevorttimeseries.pdf'
backtrajdir = '/data/keeling/a/kevintg2/a/cm1/paper2_summer2021/offlinetrajectories_backward/BACKtrajs_0_2sec_0-2km_zFIX/'
filterdir = '/data/keeling/a/kevintg2/a/cm1/paper2_summer2021/newfilter/filtertrajsfiles/'

t0 = 299 #Restart file starts at 47 min 2 sec (2822 s). So 57 min is time index 299 in the 2sec file ((3420s-2822s)/2 = 299).

#Load back trajectories, these will be the 'old' back trajectories since we have to make 'new' ones in chronological order
oldbackxpos = np.load(backtrajdir+timetag+'xpos.npy')
oldbackypos = np.load(backtrajdir+timetag+'ypos.npy')
oldbackzpos = np.load(backtrajdir+timetag+'zpos.npy')
oldbackzposhASL = np.load(backtrajdir+timetag+'zpos_heightASL.npy')
oldbackupos = np.load(backtrajdir+timetag+'uinterp.npy')
oldbackvpos = np.load(backtrajdir+timetag+'vinterp.npy')
oldbackwpos = np.load(backtrajdir+timetag+'winterp.npy')
oldbackxvortpos = np.load(backtrajdir+timetag+'xvort.npy')
oldbackyvortpos = np.load(backtrajdir+timetag+'yvort.npy')
oldbackzvortpos = np.load(backtrajdir+timetag+'zvort.npy')

###***This passes Kevin's check 07/28/21
#Open the 2 sec .nc file and get the times
ds = xr.open_dataset(fname)
timesinfile = ds.time

###***This passes Kevin's check 07/28/21
##Reverse the backward trajectory arrays

#Get the shape and length of each dimension
#print('shape of oldbackzposhASL: ',oldbackzposhASL.shape)
numxtraj = len(oldbackzposhASL[0,0,0,:])
numytraj = len(oldbackzposhASL[0,0,:,0])
numztraj = len(oldbackzposhASL[0,:,0,0])
numttraj = len(oldbackzposhASL[:,0,0,0])

#Set up empty arrays to hold the flipped back traj arrays
backxpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backypos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backzpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backzposhASL = np.empty((numttraj, numztraj, numytraj, numxtraj))
backupos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backvpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backwpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backxvortpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backyvortpos = np.empty((numttraj, numztraj, numytraj, numxtraj))
backzvortpos = np.empty((numttraj, numztraj, numytraj, numxtraj))

#Reverse the back traj arrays
tcount = np.arange(numttraj)
numoftimes = len(tcount)

for t in tcount:
    revt = numoftimes-t
    backxpos[t,:,:,:] = oldbackxpos[revt-1,:,:,:]
    backypos[t,:,:,:] = oldbackypos[revt-1,:,:,:]
    backzpos[t,:,:,:] = oldbackzpos[revt-1,:,:,:]
    backzposhASL[t,:,:,:] = oldbackzposhASL[revt-1,:,:,:]
    backupos[t,:,:,:] = oldbackupos[revt-1,:,:,:]
    backvpos[t,:,:,:] = oldbackvpos[revt-1,:,:,:]
    backwpos[t,:,:,:] = oldbackwpos[revt-1,:,:,:]
    backxvortpos[t,:,:,:] = oldbackxvortpos[revt-1,:,:,:]
    backyvortpos[t,:,:,:] = oldbackyvortpos[revt-1,:,:,:]
    backzvortpos[t,:,:,:] = oldbackzvortpos[revt-1,:,:,:]

###FILTERING OF THE TRAJECTORIES
#Set up an array that holds filtering flag information for each traj
filtertrajs = np.load(filterdir+shortname+'_'+timetag+'filtertrajs_lowest12.5.npy')

###***This passes Kevin's check 07/28/21
#The timestep is 2 sec
dt = 2.0
dx = 250.0
dy = 250.0

#Get umove and vmove and average storm speed
umove = 14.5 #Don't think we need to convert to g-r winds since the s-r winds in a translating domain are essentially g-r.
vmove = 4.5 #Tested this before. Using s-r winds makes physical sense
avgci = stormspd[0] #Still need storm speed for streamwise calculation
avgcj = stormspd[1]

#Open the .nc file and get grid dimensions and height values
ds = xr.open_dataset(fname)
dsbs = xr.open_dataset(bsfname)
xh = ds.xh
yh = ds.yh
zh = ds.z #1d array of scalar heights
xf = ds.xf
yf = ds.yf
zf = ds.zf
height = ds.zh #4d array of scalar heights
xgrid = np.arange(0,len(xh))
ygrid = np.arange(0,len(yh))
zgrid = np.arange(0,len(zh))
xgridp1 = np.arange(0,len(xf))
ygridp1 = np.arange(0,len(yf))
zgridp1 = np.arange(0,len(zf))

#Get some variables from the .nc file
u  = ds.uinterp
up1 = ds.u
uhidiff = ds.ub_hidiff
uvidiff = ds.ub_vidiff
uhturb  = ds.ub_hturb
uvturb  = ds.ub_vturb
v  = ds.vinterp
vp1 = ds.v
vhidiff = ds.vb_hidiff
vvidiff = ds.vb_vidiff
vhturb  = ds.vb_hturb
vvturb  = ds.vb_vturb
w  = ds.winterp
wp1 = ds.w
whidiff = ds.wb_hidiff
wvidiff = ds.wb_vidiff
whturb  = ds.wb_hturb
wvturb  = ds.wb_vturb
rho = ds.rho #dry air density
th0 = dsbs.th0[0,:,:,:]
qv0 = dsbs.qv[0,:,:,:]
qc0 = dsbs.qc[0,:,:,:]
qr0 = dsbs.qr[0,:,:,:]
qi0 = dsbs.qi[0,:,:,:]
qs0 = dsbs.qs[0,:,:,:]
qg0 = dsbs.qg[0,:,:,:]
qhl0 = dsbs.qhl[0,:,:,:]
thv0 = th0*(1.0+(0.61*qv0))
thrho0 = thv0*((1.0+qv0)/(1.0+qv0+qc0+qr0+qi0+qs0+qg0+qhl0)) #Get a 3D array of thrho0
th = ds.th
qv = ds.qv
qc = ds.qc
qi = ds.qi
qr = ds.qr
qs = ds.qs
qg = ds.qg
qhl = ds.qhl
prs = ds.prs
buoy = ds.buoyancy
xvort = ds.xvort
yvort = ds.yvort

###***This passes Kevin's check 08/11/21
#Define a function to get psi. The function takes u and v as input and returns a psi value.
def get_psi(uforpsi,vforpsi):
#No if statements
#     psi = math.atan(vforpsi/uforpsi)
#     quadrant = 0#use a bogus quadrant for now
#     return(psi,quadrant)
#     del(psi)
#If statements based on quadrant
    if uforpsi>0.0 and vforpsi>0.0:
        psi = math.atan(abs(vforpsi)/abs(uforpsi))
        quadrant = 1
    elif uforpsi<0.0 and vforpsi>0.0:
        psi = math.pi - math.atan(abs(vforpsi)/abs(uforpsi))
        quadrant = 2
    elif uforpsi<0.0 and vforpsi<0.0:
        psi = math.pi + math.atan(abs(vforpsi)/abs(uforpsi))
        quadrant = 3
    elif uforpsi>0.0 and vforpsi<0.0:
        psi = (2.0*math.pi) - math.atan(abs(vforpsi)/abs(uforpsi))
        quadrant = 4
    elif uforpsi>0.0 and vforpsi==0.0:
        psi = 0.0
        quadrant = math.nan
    elif uforpsi<0.0 and vforpsi==0.0:
        psi = math.pi
        quadrant = math.nan
    elif uforpsi==0.0 and vforpsi>0.0:
        psi = (math.pi)/2.0
        quadrant = math.nan
    elif uforpsi==0.0 and vforpsi<0.0:
        psi = (math.pi)*(3.0/2.0)
        quadrant = math.nan
    elif uforpsi==0.0 and vforpsi==0.0:
        psi = math.nan
        quadrant = math.nan
    return(psi,quadrant)
    del(psi)

###***This passes Kevin's check 08/11/21
#Define a function to get wind speed. The function takes u, v, and w as input and returns a wind speed value.
def get_V(uforV,vforV):
    V = ((uforV**2.0)+(vforV**2.0))**(0.5)#Horizontal speed only
    return(V)
    del(V)

###***
#Define a function to get the gridpoints left, right, ahead of, and behind the trajectory location.
def get_locs(traju,trajv):
    dirtheta = math.atan(abs(trajv)/abs(traju))
    if (traju)>0.0 and (trajv)>0.0:
        deltaiL = -(math.sin(dirtheta))
        deltajL = (math.cos(dirtheta))
        deltaiR = -deltaiL
        deltajR = -deltajL
        deltaiA = (math.cos(dirtheta))
        deltajA = (math.sin(dirtheta))
        deltaiB = -deltaiA
        deltajB = -deltajA
    if (traju)<0.0 and (trajv)>0.0:
        deltaiL = -(math.sin(dirtheta))
        deltajL = -(math.cos(dirtheta))
        deltaiR = -deltaiL
        deltajR = -deltajL
        deltaiA = -(math.cos(dirtheta))
        deltajA = (math.sin(dirtheta))
        deltaiB = -deltaiA
        deltajB = -deltajA
    if (traju)<0.0 and (trajv)<0.0:
        deltaiL = (math.sin(dirtheta))
        deltajL = -(math.cos(dirtheta))
        deltaiR = -deltaiL
        deltajR = -deltajL
        deltaiA = -(math.cos(dirtheta))
        deltajA = -(math.sin(dirtheta))
        deltaiB = -deltaiA
        deltajB = -deltajA
    if (traju)>0.0 and (trajv)<0.0:
        deltaiL = (math.sin(dirtheta))
        deltajL = (math.cos(dirtheta))
        deltaiR = -deltaiL
        deltajR = -deltajL
        deltaiA = (math.cos(dirtheta))
        deltajA = -(math.sin(dirtheta))
        deltaiB = -deltaiA
        deltajB = -deltajA
    return(deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB)
    del(deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB)

for k in np.arange(0,1,1):#(0,23,1):
    for j in np.arange(0,1,1):#(0,1,1):
        for i in np.arange(12,13,1):#(0,23,1):
            print('traj %d, %d, %d filter: ' %(k,j,i), filtertrajs[0,k,j,i])
            if filtertrajs[0,k,j,i] == 1:
                tk = k
                tj = j
                ti = i
                print('Working on trajectory [%d, %d, %d]' %(tk,tj,ti))

                #Combine the back and forw trajs into one array
                #Or just copy backtrajs to fullxxxx arrays
                fullxpos = np.copy(backxpos[:,tk,tj,ti])
                fullypos = np.copy(backypos[:,tk,tj,ti])
                fullzpos = np.copy(backzpos[:,tk,tj,ti])
                fullzposhASL = np.copy(backzposhASL[:,tk,tj,ti])
                fullupos = np.copy(backupos[:,tk,tj,ti])
                fullvpos = np.copy(backvpos[:,tk,tj,ti])
                fullwpos = np.copy(backwpos[:,tk,tj,ti])
                fullxvortpos = np.copy(backxvortpos[:,tk,tj,ti])
                fullyvortpos = np.copy(backyvortpos[:,tk,tj,ti])
                fullzvortpos = np.copy(backzvortpos[:,tk,tj,ti])

                fullupos[:] = fullupos[:]-(avgci-umove)
                fullvpos[:] = fullvpos[:]-(avgcj-vmove)

                ###--------------------------------------------------------------------------------------------------
                ###Get time that trajectory rises to 1 km after it's minimum height
                minheight = 9999.9
                time1 = 0
                time2 = 0
                for it in np.arange(0,len(fullxpos),1):
                    if fullzposhASL[it] < minheight:
                        minheight = fullzposhASL[it]
                        time1 = it
                for it in np.arange(time1,len(fullxpos),1):
                    if fullzposhASL[it] > 1000.0:
                        time2 = it
                        break
                if time2 == 0:
                    time2 = len(fullxpos)-1

                ###---------------------------------------------------------------------------------------------------
                ###***This passes Kevin's check 08/11/21
                ##This cell creates an array of real times
                realtimes = []

                timeindicestoloop = np.arange(0,time2+1,1)#Use this when ready for the full time loop
                #timeindicestoloop = np.arange(0,4,1)#For testing
                for it in timeindicestoloop:
                    realtimes.append((timetagint-10)+(2.0/60.0)+(it*(2.0/60.0)))

                heightlist = fullzposhASL[:-1]

                ###***This passes Kevin's check 08/11/21 FOR ONE TIME STEP
                #This cell creates an array of V at trajectory location
                #Set up arrays to hold variables and terms

                Vloc = []
                psiloc = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    Vloc.append(get_V(fullupos[it],fullvpos[it]))
                    psiloctemp,psiquad = get_psi(fullupos[it],fullvpos[it])
                    psiloc.append(psiloctemp)

                ##This cell creates an array of D(psi)/Dt using trajectory code output
                #Set up arrays to hold variables and terms
                DpsiDt = []
                psiquadissueDt = []
                psiquadissueDttimes = []
                psiafterarr = []
                psibeforearr = []
                
                #Get time indices to loop over, start at 1 since we will need the first t-1 to be zero for psi time tendency term
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)
                for it in timeindicestoloop:
                
                    #Get winds next to trajectory location using the interpolators
                    if it == 0:
                        ubefore = fullupos[it]
                        vbefore = fullvpos[it]
                    else:
                        ubefore = fullupos[it-1]#+umove
                        vbefore = fullvpos[it-1]#+umove
                    uafter = fullupos[it+1]#+umove
                    vafter = fullvpos[it+1]#+umove
                
                    #Call the psi function
                    psibefore,quadbefore = get_psi(ubefore,vbefore)
                    psiafter,quadafter = get_psi(uafter,vafter)
                #     print('time: ',realtimes[it])
                #     print('ubefore: ',ubefore)
                #     print('vbefore: ',vbefore)
                #     print('uafter: ',uafter)
                #     print('vafter: ',vafter)
                #     print('psibefore: ',psibefore)
                #     print('psiafter: ',psiafter)
                    if quadafter == 1 and quadbefore == 4:
                        psiafter = psiafter+(math.pi*2.0)
                        psiquadissueDt.append(it)
                        psiquadissueDttimes.append(realtimes[it])
                    if quadafter == 4 and quadbefore == 1:
                        psiquadissueDt.append(it)
                        psiquadissueDttimes.append(realtimes[it])
                    psiafterarr.append(psiafter)
                    psibeforearr.append(psibefore)
                    if it == 0:
                        DpsiDt.append((psiafter-psibefore)/(1.0*dt))
                    else:
                        DpsiDt.append((psiafter-psibefore)/(2.0*dt))

                ###--------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of dVds for streamwise stretching
                #and an array of dVdn tilting of crosswise to streamwise
                #Set up arrays to hold variables and terms
                dVds = []
                dVdn = []
                
                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:

                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), up1[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vp1[it,:,:,:])

                    ##############################################
                    #Call the get_locs function to get coords left, right, ahead of, and behind trajectory location
                    deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB = get_locs(fullupos[it],fullvpos[it])
                    #Get winds next to trajectory location using the interpolators
                    uR = uinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+0.5+deltaiR])-(avgci-umove)
                    vR = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajR, fullxpos[it]+deltaiR])-(avgcj-vmove)
                    #VsR = ((fullupos[it]*uR)+(fullvpos[it]*vR))/Vloc[it] #Dot product identity
                    VR = get_V(uR,vR)
                    uL = uinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+0.5+deltaiL])-(avgci-umove)
                    vL = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajL, fullxpos[it]+deltaiL])-(avgcj-vmove)
                    #VsL = ((fullupos[it]*uL)+(fullvpos[it]*vL))/Vloc[it]
                    VL = get_V(uL,vL)
                    uA = uinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+0.5+deltaiA])-(avgci-umove)
                    vA = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajA, fullxpos[it]+deltaiA])-(avgcj-vmove)
                    #VsA = ((fullupos[it]*uA)+(fullvpos[it]*vA))/Vloc[it]
                    VA = get_V(uA,vA)
                    uB = uinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+0.5+deltaiB])-(avgci-umove)
                    vB = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajB, fullxpos[it]+deltaiB])-(avgcj-vmove)
                    #VsB = ((fullupos[it]*uB)+(fullvpos[it]*vB))/Vloc[it]
                    VB = get_V(uB,vB)
                    
                    dVds.append((VA[0]-VB[0])/(2.0*dx))
                    dVdn.append((VL[0]-VR[0])/(2.0*dx))

                    del(uinterpolator)
                    del(vinterpolator)

                ###----------------------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of dVsdz for tilting of vertical to streamwise
                #Set up arrays to hold variables and terms
                dVdz = []
                countuponly = 0
                uponlylist = []
                
                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    uponlyflag = 0

                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), up1[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vp1[it,:,:,:])
                    heightinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])

                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])

                    #Get winds next to trajectory location using the interpolators
                    uup = uinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                    vup = vinterpolator([fullzpos[it]+1.0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                    #Vsup = ((fullupos[it]*uup)+(fullvpos[it]*vup))/Vloc[it]
                    Vup = get_V(uup,vup)
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        udown = uinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                        vdown = vinterpolator([fullzpos[it]-1.0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                        #Vsdown = ((fullupos[it]*udown)+(fullvpos[it]*vdown))/Vloc[it]
                        Vdown = get_V(udown,vdown)
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        udown = uinterpolator([0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                        vdown = vinterpolator([0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                        #Vsdown = ((fullupos[it]*udown)+(fullvpos[it]*vdown))/Vloc[it]
                        Vdown = get_V(udown,vdown)
                        zdown = heightinterpolator([0, fullypos[it], fullxpos[it]])
                        uponlyflag = 1
                        countuponly = countuponly+1
                        uponlylist.append(it)

                    dz = zup-zdown

                    dVdz.append((Vup-Vdown)/dz)
                #     ###Not sure why we need this if statement here but not in the vertical psi advection cell...
                #     if uponlyflag == 0:
                #         dVdz.append((Vup[0]-Vdown[0])/(dz[0]))
                #     else:
                #         dVdz.append((Vup[0]-Vdown)/(dz[0]))
                
                    del(uinterpolator)
                    del(vinterpolator)
                    del(heightinterpolator)

                ###-----------------------------------------------------------------------------------------------------
                ###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
                ###################Try just air density, no qv correction
                #This cell creates an array of the baroclinic terms for vertical vorticity equation
                #Set up arrays to hold variables and terms
                drhovdn = []
                dpdn = []
                drhovds = []
                dpds = []
                invrhov2 = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:

                    #Set up interpolators
                    rhointerpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), rho[it,:,:,:])
                    qvinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgrid), qv[it,:,:,:])
                    prsinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), prs[it,:,:,:])

                    #Get rhov for 1/rhov**2 in the baroclinic term
                    rholoc = rhointerpolator([fullzpos[it], fullypos[it], fullxpos[it]])
                    qvloc = qvinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])
                    #rhovloc = rholoc#+(rholoc*qvloc)#No qv correction
                    rhovloc = rholoc+(rholoc*qvloc)#Yes qv correction

                    invrhov2.append(1/(rhovloc[0]**2.0))

                    ##############################################
                    #Call the get_locs function to get coords left, right, ahead of, and behind trajectory location
                    deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB = get_locs(fullupos[it],fullvpos[it])
                    ##############################################
                    rhoR = rhointerpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qvR = qvinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    #rhovR = rhoR#+(rhoR*qvR)#No qv correction
                    rhovR = rhoR+(rhoR*qvR)#Yes qv correction
                    rhoL = rhointerpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qvL = qvinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    #rhovL = rhoL#+(rhoL*qvL)#No qv correction
                    rhovL = rhoL+(rhoL*qvL)#Yes qv correction
                    prsR = prsinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    prsL = prsinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])

                    drhovdn.append((rhovL[0]-rhovR[0])/(2.0*dx))
                    dpdn.append((prsL[0]-prsR[0])/(2.0*dx))

                    rhoA = rhointerpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qvA = qvinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    #rhovA = rhoA#+(rhoA*qvA)#No qv correction
                    rhovA = rhoA+(rhoA*qvA)#Yes qv correction
                    rhoB = rhointerpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qvB = qvinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    #rhovB = rhoB#+(rhoB*qvB)#No qv correction
                    rhovB = rhoB+(rhoB*qvB)#Yes qv correction
                    prsA = prsinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    prsB = prsinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])

                    drhovds.append((rhovA[0]-rhovB[0])/(2.0*dx))
                    dpds.append((prsA[0]-prsB[0])/(2.0*dx))

                    ##############################################

                    del(rhointerpolator)
                    del(qvinterpolator)
                    del(prsinterpolator)

                ###***This passes Kevin's check 08/16/21
                #This cell creates an array of the baroclinic terms (in terms of buoyancy, B)
                #Set up arrays to hold variables and terms
                dBdn = []
                dBds = []
                g = 9.81 #acceleration due to gravity

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:

                    #Set up interpolators
                    buoyinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), buoy[it,:,:,:])

                    ##############################################
                    #Call the get_locs function to get coords left, right, ahead of, and behind trajectory location
                    deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB = get_locs(fullupos[it],fullvpos[it])
                    ##############################################
                    BR = buoyinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    BL = buoyinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])

                    dBdn.append((BL[0]-BR[0])/(2.0*dx))

                    BA = buoyinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    BB = buoyinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])

                    dBds.append((BA[0]-BB[0])/(2.0*dx))

                    ##############################################
                    del(buoyinterpolator)

                ###---------------------------------------------------------------------------------------------------------
                ###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
                #This cell creates an array of dwdz for vertical vorticity stretching
                #Set up arrays to hold variables and terms
                dwdz = []
                countuponly = 0
                countuponlyarray = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    uponlyflag = 0

                    #Set up interpolators
                    winterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), wp1[it,:,:,:])
                    heightinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])

                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])

                    #Get winds next to trajectory location using the interpolators
                    wup = winterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        wdown = winterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        wdown = winterpolator([0.5, fullypos[it], fullxpos[it]])
                        zdown = heightinterpolator([0, fullypos[it], fullxpos[it]])
                        uponlyflag = 1
                        countuponly = countuponly+1
                        countuponlyarray.append(it)

                    dz = zup-zdown

                    dwdz.append((wup[0]-wdown[0])/dz[0])

                    ###Not sure why we need this if statement here but not in the vertical psi advection cell...
                    #if uponlyflag == 0:
                    #    dVdz.append((Vup[0]-Vdown[0])/(dz[0]))
                    #else:
                    #    dVdz.append((Vup[0]-Vdown)/(dz[0]))

                    del(winterpolator)
                    del(heightinterpolator)

                ###----------------------------------------------------------------------------------------------------------
                ###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
                #This cell creates an array of tilting terms for vertical vorticity
                #Set up arrays to hold variables and terms
                dwds = []
                dwdn = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:

                    #Set up interpolators
                    winterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), wp1[it,:,:,:])

                    ##############################################
                    #Call the get_locs function to get coords left, right, ahead of, and behind trajectory location
                    deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB = get_locs(fullupos[it],fullvpos[it])
                    ##############################################
                    wR = winterpolator([fullzpos[it]+0.5, fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    wL = winterpolator([fullzpos[it]+0.5, fullypos[it]+deltajL, fullxpos[it]+deltaiL])

                    dwdn.append((wL[0]-wR[0])/(2.0*dx))

                    wA = winterpolator([fullzpos[it]+0.5, fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    wB = winterpolator([fullzpos[it]+0.5, fullypos[it]+deltajB, fullxpos[it]+deltaiB])

                    dwds.append((wA[0]-wB[0])/(2.0*dx))
                    ##############################################
                    del(winterpolator)

                ###--------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of dpsidn for crosswise stretching
                #and an array of dpsids for tilting of streamwise to crosswise
                #Set up arrays to hold variables and terms
                dpsids = []
                dpsidn = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:

                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), up1[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vp1[it,:,:,:])

                    ##############################################
                    #Call the get_locs function to get coords left, right, ahead of, and behind trajectory location
                    deltaiL,deltajL,deltaiR,deltajR,deltaiA,deltajA,deltaiB,deltajB = get_locs(fullupos[it],fullvpos[it])
                    #Get winds next to trajectory location using the interpolators
                    uR = uinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+0.5+deltaiR])-(avgci-umove)
                    vR = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajR, fullxpos[it]+deltaiR])-(avgcj-vmove)
                    #VsR = ((fullupos[it]*uR)+(fullvpos[it]*vR))/Vloc[it] #Dot product identity
                    psiR,quadR = get_psi(uR,vR)
                    uL = uinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+0.5+deltaiL])-(avgci-umove)
                    vL = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajL, fullxpos[it]+deltaiL])-(avgcj-vmove)
                    #VsL = ((fullupos[it]*uL)+(fullvpos[it]*vL))/Vloc[it]
                    psiL,quadL = get_psi(uL,vL)
                    uA = uinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+0.5+deltaiA])-(avgci-umove)
                    vA = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajA, fullxpos[it]+deltaiA])-(avgcj-vmove)
                    #VsA = ((fullupos[it]*uA)+(fullvpos[it]*vA))/Vloc[it]
                    psiA,quadA = get_psi(uA,vA)
                    uB = uinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+0.5+deltaiB])-(avgci-umove)
                    vB = vinterpolator([fullzpos[it], fullypos[it]+0.5+deltajB, fullxpos[it]+deltaiB])-(avgcj-vmove)
                    #VsB = ((fullupos[it]*uB)+(fullvpos[it]*vB))/Vloc[it]
                    psiB,quadB = get_psi(uB,vB)
                    
                    if quadA == 1 and quadB == 4:
                        psiA = psiA+(math.pi*2.0)
                    if quadL == 1 and quadR == 4:
                        psiL = psiL+(math.pi*2.0)
                    
                    dpsids.append((psiA-psiB)/(2.0*dx))
                    dpsidn.append((psiL-psiR)/(2.0*dx))

                    del(uinterpolator)
                    del(vinterpolator)

                ###----------------------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of dpsidz for tilting of vertical to crosswise
                #Set up arrays to hold variables and terms
                dpsidz = []
                countuponly = 0
                uponlylist = []
                
                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    uponlyflag = 0

                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), up1[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vp1[it,:,:,:])
                    heightinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])

                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])

                    #Get winds next to trajectory location using the interpolators
                    uup = uinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                    vup = vinterpolator([fullzpos[it]+1.0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                    #Vsup = ((fullupos[it]*uup)+(fullvpos[it]*vup))/Vloc[it]
                    psiup,quadup = get_psi(uup,vup)
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        udown = uinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                        vdown = vinterpolator([fullzpos[it]-1.0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                        #Vsdown = ((fullupos[it]*udown)+(fullvpos[it]*vdown))/Vloc[it]
                        psidown,quaddown = get_psi(udown,vdown)
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        udown = uinterpolator([0, fullypos[it], fullxpos[it]+0.5])-(avgci-umove)
                        vdown = vinterpolator([0, fullypos[it]+0.5, fullxpos[it]])-(avgcj-vmove)
                        #Vsdown = ((fullupos[it]*udown)+(fullvpos[it]*vdown))/Vloc[it]
                        psidown,quaddown = get_psi(udown,vdown)
                        zdown = heightinterpolator([0, fullypos[it], fullxpos[it]])
                        uponlyflag = 1
                        countuponly = countuponly+1
                        uponlylist.append(it)

                    dz = zup-zdown

                    if quadup == 1 and quaddown == 4:
                        psiup = psiup+(math.pi*2.0)
                    
                    dpsidz.append((psiup-psidown)/dz)
                #     ###Not sure why we need this if statement here but not in the vertical psi advection cell...
                #     if uponlyflag == 0:
                #         dVdz.append((Vup[0]-Vdown[0])/(dz[0]))
                #     else:
                #         dVdz.append((Vup[0]-Vdown)/(dz[0]))

                    del(uinterpolator)
                    del(vinterpolator)
                    del(heightinterpolator)

                ###----------------------------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of implicit diffusion and subgrid-scale turbulence terms for xi
                #Set up arrays to hold variables and terms
                xiidiff = []
                xiturb  = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
    
                    #Set up interpolators
                    vhidiffinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vhidiff[it,:,:,:])
                    vvidiffinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vvidiff[it,:,:,:])
                    vhturbinterpolator        = RegularGridInterpolator((zgrid, ygridp1, xgrid), vhturb[it,:,:,:])
                    vvturbinterpolator        = RegularGridInterpolator((zgrid, ygridp1, xgrid), vvturb[it,:,:,:])
                    whidiffinterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), whidiff[it,:,:,:])
                    wvidiffinterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), wvidiff[it,:,:,:])
                    whturbinterpolator        = RegularGridInterpolator((zgridp1, ygrid, xgrid), whturb[it,:,:,:])
                    wvturbinterpolator        = RegularGridInterpolator((zgridp1, ygrid, xgrid), wvturb[it,:,:,:])
                    heightinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])
    
                    ###calculate idiff and turb xi in cartesian coordinates
                    #get what we need for dwdy
                    whidiffN    = whidiffinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    wvidiffN    = wvidiffinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    whturbN     = whturbinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    wvturbN     = wvturbinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    whidiffS    = whidiffinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    wvidiffS    = wvidiffinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    whturbS     = whturbinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    wvturbS     = wvturbinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    #add horizontal and vertical components of idiff and turb
                    widiffN = whidiffN+wvidiffN
                    wturbN  = whturbN+wvturbN
                    widiffS = whidiffS+wvidiffS
                    wturbS  = whturbS+wvturbS
                    #get dwdy
                    dwidiffdy = (widiffN-widiffS)/(2.0*dy)
                    dwturbdy  = (wturbN-wturbS)/(2.0*dy)
    
                    #get what we need for dvdz
                    vhidiffup   = vhidiffinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    vvidiffup   = vvidiffinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    vhturbup    = vhturbinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    vvturbup    = vvturbinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    zup         = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        vhidiffdown = vhidiffinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        vvidiffdown = vvidiffinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        vhturbdown  = vhturbinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        vvturbdown  = vvturbinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        zdown       = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        vhidiffdown = vhidiffinterpolator([0.5, fullypos[it], fullxpos[it]])
                        vvidiffdown = vvidiffinterpolator([0.5, fullypos[it], fullxpos[it]])
                        vhturbdown  = vhturbinterpolator([0.5, fullypos[it], fullxpos[it]])
                        vvturbdown  = vvturbinterpolator([0.5, fullypos[it], fullxpos[it]])
                        zdown       = heightinterpolator([0, fullypos[it], fullxpos[it]])
                    #add horizontal and vertical components of idiff and turb
                    vidiffup   = vhidiffup+vvidiffup
                    vturbup    = vhturbup+vvturbup
                    vidiffdown = vhidiffdown+vvidiffdown
                    vturbdown  = vhturbdown+vvturbdown
                    #get dvdz
                    dz = zup-zdown
                    dvidiffdz = (vidiffup-vidiffdown)/dz
                    dvturbdz  = (vturbup-vturbdown)/dz
    
                    #calculate and append xi due to idiff and turb
                    xiidiff.append(dwidiffdy-dvidiffdz)
                    xiturb.append(dwturbdy-dvturbdz)
    
                    del(vhidiffinterpolator)
                    del(vvidiffinterpolator)
                    del(vhturbinterpolator)
                    del(vvturbinterpolator)
                    del(whidiffinterpolator)
                    del(wvidiffinterpolator)
                    del(whturbinterpolator)
                    del(wvturbinterpolator)
                    del(heightinterpolator)

                ###----------------------------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of implicit diffusion and subgrid-scale turbulence terms for eta
                #Set up arrays to hold variables and terms
                etaidiff = []
                etaturb  = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    
                    #Set up interpolators
                    uhidiffinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), uhidiff[it,:,:,:])
                    uvidiffinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), uvidiff[it,:,:,:])
                    uhturbinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgridp1), uhturb[it,:,:,:])
                    uvturbinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgridp1), uvturb[it,:,:,:])
                    whidiffinterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), whidiff[it,:,:,:])
                    wvidiffinterpolator       = RegularGridInterpolator((zgridp1, ygrid, xgrid), wvidiff[it,:,:,:])
                    whturbinterpolator        = RegularGridInterpolator((zgridp1, ygrid, xgrid), whturb[it,:,:,:])
                    wvturbinterpolator        = RegularGridInterpolator((zgridp1, ygrid, xgrid), wvturb[it,:,:,:])
                    heightinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])
    
                    ###calculate idiff and turb eta in cartesian coordinates
                    #get what we need for dwdx
                    whidiffE    = whidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    wvidiffE    = wvidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    whturbE     = whturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    wvturbE     = wvturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    whidiffW    = whidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    wvidiffW    = wvidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    whturbW     = whturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    wvturbW     = wvturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    #add horizontal and vertical components of idiff and turb
                    widiffE = whidiffE+wvidiffE
                    wturbE  = whturbE+wvturbE
                    widiffW = whidiffW+wvidiffW
                    wturbW  = whturbW+wvturbW
                    #get dwdx
                    dwidiffdx = (widiffE-widiffW)/(2.0*dx)
                    dwturbdx  = (wturbE-wturbW)/(2.0*dx)
    
                    #get what we need for dvdz
                    uhidiffup   = uhidiffinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    uvidiffup   = uvidiffinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    uhturbup    = uhturbinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    uvturbup    = uvturbinterpolator([fullzpos[it]+1.5, fullypos[it], fullxpos[it]])
                    zup         = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        uhidiffdown = uhidiffinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        uvidiffdown = uvidiffinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        uhturbdown  = uhturbinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        uvturbdown  = uvturbinterpolator([fullzpos[it]-0.5, fullypos[it], fullxpos[it]])
                        zdown       = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        uhidiffdown = uhidiffinterpolator([0.5, fullypos[it], fullxpos[it]])
                        uvidiffdown = uvidiffinterpolator([0.5, fullypos[it], fullxpos[it]])
                        uhturbdown  = uhturbinterpolator([0.5, fullypos[it], fullxpos[it]])
                        uvturbdown  = uvturbinterpolator([0.5, fullypos[it], fullxpos[it]])
                        zdown       = heightinterpolator([0, fullypos[it], fullxpos[it]])
                    #add horizontal and vertical components of idiff and turb
                    uidiffup   = uhidiffup+uvidiffup
                    uturbup    = uhturbup+uvturbup
                    uidiffdown = uhidiffdown+uvidiffdown
                    uturbdown  = uhturbdown+uvturbdown
                    #get dvdz
                    dz = zup-zdown
                    duidiffdz = (uidiffup-uidiffdown)/dz
                    duturbdz  = (uturbup-uturbdown)/dz
    
                    #calculate and append xi due to idiff and turb
                    etaidiff.append(duidiffdz-dwidiffdx)
                    etaturb.append(duturbdz-dwturbdx)
    
                    del(uhidiffinterpolator)
                    del(uvidiffinterpolator)
                    del(uhturbinterpolator)
                    del(uvturbinterpolator)
                    del(whidiffinterpolator)
                    del(wvidiffinterpolator)
                    del(whturbinterpolator)
                    del(wvturbinterpolator)
                    del(heightinterpolator)

                ###----------------------------------------------------------------------------------------------------------
                ###***
                #This cell creates an array of implicit diffusion and subgrid-scale turbulence terms for zeta
                #Set up arrays to hold variables and terms
                zetaidiff = []
                zetaturb  = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
    
                    #Set up interpolators
                    uhidiffinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), uhidiff[it,:,:,:])
                    uvidiffinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgridp1), uvidiff[it,:,:,:])
                    uhturbinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgridp1), uhturb[it,:,:,:])
                    uvturbinterpolator        = RegularGridInterpolator((zgrid, ygrid, xgridp1), uvturb[it,:,:,:])
                    vhidiffinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vhidiff[it,:,:,:])
                    vvidiffinterpolator       = RegularGridInterpolator((zgrid, ygridp1, xgrid), vvidiff[it,:,:,:])
                    vhturbinterpolator        = RegularGridInterpolator((zgrid, ygridp1, xgrid), vhturb[it,:,:,:])
                    vvturbinterpolator        = RegularGridInterpolator((zgrid, ygridp1, xgrid), vvturb[it,:,:,:])
    
                    ###calculate idiff and turb zeta in cartesian coordinates
                    #get what we need for dvdx
                    vhidiffE    = vhidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    vvidiffE    = vvidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    vhturbE     = vhturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    vvturbE     = vvturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]+1.5])
                    vhidiffW    = vhidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    vvidiffW    = vvidiffinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    vhturbW     = vhturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    vvturbW     = vvturbinterpolator([fullzpos[it], fullypos[it], fullxpos[it]-0.5])
                    #add horizontal and vertical components of idiff and turb
                    vidiffE = vhidiffE+vvidiffE
                    vturbE  = vhturbE+vvturbE
                    vidiffW = vhidiffW+vvidiffW
                    vturbW  = vhturbW+vvturbW
                    #get dvdx
                    dvidiffdx = (vidiffE-vidiffW)/(2.0*dx)
                    dvturbdx  = (vturbE-vturbW)/(2.0*dx)
    
                    #get what we need for dudy
                    uhidiffN    = uhidiffinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    uvidiffN    = uvidiffinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    uhturbN     = uhturbinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    uvturbN     = uvturbinterpolator([fullzpos[it], fullypos[it]+1.5, fullxpos[it]])
                    uhidiffS    = uhidiffinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    uvidiffS    = uvidiffinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    uhturbS     = uhturbinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    uvturbS     = uvturbinterpolator([fullzpos[it], fullypos[it]-0.5, fullxpos[it]])
                    #add horizontal and vertical components of idiff and turb
                    uidiffN = uhidiffN+uvidiffN
                    uturbN  = uhturbN+uvturbN
                    uidiffS = uhidiffS+uvidiffS
                    uturbS  = uhturbS+uvturbS
                    #get dudy
                    duidiffdy = (uidiffN-uidiffS)/(2.0*dy)
                    duturbdy  = (uturbN-uturbS)/(2.0*dy)
    
                    #calculate and append xi due to idiff and turb
                    zetaidiff.append(dvidiffdx-duidiffdy)
                    zetaturb.append(dvturbdx-duturbdy)
    
                    del(uhidiffinterpolator)
                    del(uvidiffinterpolator)
                    del(uhturbinterpolator)
                    del(uvturbinterpolator)
                    del(vhidiffinterpolator)
                    del(vvidiffinterpolator)
                    del(vhturbinterpolator)
                    del(vvturbinterpolator)

                ###---------------------------------------------------------------------------------------------------
                ###***
                #This cell is to get implicit diffusion and subgrid-scale turbulence streamwise, and crosswise vorticity
                #Set up arrays to hold variables and terms
                idiffcrosswise  = []
                idiffstreamwise = []
                turbcrosswise   = []
                turbstreamwise  = []

                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    sru = fullupos[it]#these have all been converted to s-r
                    srv = fullvpos[it]

                    totvort = ((xiidiff[it]**2.0)+(etaidiff[it]**2.0))**(0.5)
                    idiffstreamwisetemp = (((xiidiff[it]*sru)+(etaidiff[it]*srv))/(((sru**2.0)+(srv**2.0))**(0.5)))
                    idiffcrosswisetemp = (((totvort**2.0)-(idiffstreamwisetemp**2.0))**(0.5))

                    #Find out if the crosswise vector is positive or negative
                    #Source: https://stackoverflow.com/questions/13221873/determining-if-one-2d-vector-is-to-the-right-or-left-of-another
                    if ((sru*(-etaidiff[it]))+(srv*xiidiff[it])) > 0.0:
                        idiffcrosswisetemp = -idiffcrosswisetemp
                    idiffstreamwise.append(idiffstreamwisetemp)
                    idiffcrosswise.append(idiffcrosswisetemp)
                    #idifftotal.append(totvort)
    
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    sru = fullupos[it]#these have all been converted to s-r
                    srv = fullvpos[it]

                    totvort = ((xiturb[it]**2.0)+(etaturb[it]**2.0))**(0.5)
                    turbstreamwisetemp = (((xiturb[it]*sru)+(etaturb[it]*srv))/(((sru**2.0)+(srv**2.0))**(0.5)))
                    turbcrosswisetemp = (((totvort**2.0)-(turbstreamwisetemp**2.0))**(0.5))

                    #Find out if the crosswise vector is positive or negative
                    #Source: https://stackoverflow.com/questions/13221873/determining-if-one-2d-vector-is-to-the-right-or-left-of-another
                    if ((sru*(-etaturb[it]))+(srv*xiturb[it])) > 0.0:
                        turbcrosswisetemp = -turbcrosswisetemp
                    turbstreamwise.append(turbstreamwisetemp)
                    turbcrosswise.append(turbcrosswisetemp)
                    #turbtotal.append(totvort)

                ###---------------------------------------------------------------------------------------------------
                ###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
                #This cell is to get model total horizontal, streamwise, and crosswise vorticity at the trajectory location
                #Set up arrays to hold variables and terms
                modcrosswise = []
                modstreamwise = []
                modtotal = []

                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    sru = fullupos[it]#these have all been converted to s-r
                    srv = fullvpos[it]

                    totvort = ((fullxvortpos[it]**2.0)+(fullyvortpos[it]**2.0))**(0.5)
                    modstreamwisetemp = (((fullxvortpos[it]*sru)+(fullyvortpos[it]*srv))/(((sru**2.0)+(srv**2.0))**(0.5)))
                    modcrosswisetemp = (((totvort**2.0)-(modstreamwisetemp**2.0))**(0.5))

                    #Find out if the crosswise vector is positive or negative
                    #Source: https://stackoverflow.com/questions/13221873/determining-if-one-2d-vector-is-to-the-right-or-left-of-another
                    if ((sru*(-fullyvortpos[it]))+(srv*fullxvortpos[it])) > 0.0:
                        modcrosswisetemp = -modcrosswisetemp
                    modstreamwise.append(modstreamwisetemp)
                    modcrosswise.append(modcrosswisetemp)
                    modtotal.append(totvort)

                ###----------------------------------------------------------------------------------------------------------
                #This cell integrates streamwise, crosswise, and vertical vorticity terms we calculated above
                #Set up arrays to hold variables and terms
                #NO RESIDUAL

                intstreamwise = []
                intcrosswise = []
                intvert = []

                intexchangestream = []
                inttiltstretchstream = []
                inttiltcross2stream = []
                inttiltvert2stream = []
                intstretchstream = []
                intbaroclinicstream = []
                intmixingstream = []

                intexchangecross = []
                inttiltstretchcross = []
                inttiltstream2cross = []
                inttiltvert2cross = []
                intstretchcross = []
                intbarocliniccross = []
                intmixingcross = []

                intstretchvert = []
                inttiltstream2vert = []
                inttiltcross2vert = []
                intbaroclinicvert = []
                intmixingvert = []

                instantexchangestream = []
                instanttiltstretchstream = []
                instanttiltcross2stream = []
                instanttiltvert2stream = []
                instantstretchstream = []
                instantbaroclinicstream = []
                instantmixingstream = []

                instantexchangecross = []
                instanttiltstretchcross = []
                instanttiltstream2cross = []
                instanttiltvert2cross = []
                instantstretchcross = []
                instantbarocliniccross = []
                instantmixingcross = []

                instantstretchvert = []
                instanttiltstream2vert = []
                instanttiltcross2vert = []
                instantbaroclinicvert = []
                instantmixingvert = []
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    #print('it: ',it)
                    #print('working on time ',realtimes[it])

                    if it <= 0: #A higher number delays the integration, say after entering outflow surge
                        intstreamwise.append(modstreamwise[it])
                        intexchangestream.append(modstreamwise[it])
                        inttiltstretchstream.append(modstreamwise[it])
                        inttiltcross2stream.append(modstreamwise[it])
                        inttiltvert2stream.append(modstreamwise[it])
                        intstretchstream.append(modstreamwise[it])
                        intbaroclinicstream.append(modstreamwise[it])
                        intmixingstream.append(modstreamwise[it])

                        intcrosswise.append(modcrosswise[it])
                        intexchangecross.append(modcrosswise[it])
                        inttiltstretchcross.append(modcrosswise[it])
                        inttiltstream2cross.append(modcrosswise[it])
                        inttiltvert2cross.append(modcrosswise[it])
                        intstretchcross.append(modcrosswise[it])
                        intbarocliniccross.append(modcrosswise[it])
                        intmixingcross.append(modcrosswise[it])

                        intvert.append(fullzvortpos[it])
                        intstretchvert.append(fullzvortpos[it])
                        inttiltstream2vert.append(fullzvortpos[it])
                        inttiltcross2vert.append(fullzvortpos[it])
                        intbaroclinicvert.append(fullzvortpos[it])
                        intmixingvert.append(fullzvortpos[it])

                        instantexchangestream.append(intcrosswise[it]*DpsiDt[it])
                        instanttiltstretchstream.append((intstreamwise[it]*dVds[it])+(intcrosswise[it]*dVdn[it])+(intvert[it]*dVdz[it]))
                        instanttiltcross2stream.append(intcrosswise[it]*dVdn[it])
                        instanttiltvert2stream.append(intvert[it]*dVdz[it])
                        instantstretchstream.append(intstreamwise[it]*dVds[it])
                        instantbaroclinicstream.append(dBdn[it])
                        instantmixingstream.append(idiffstreamwise[it]+turbstreamwise[it])

                        instantexchangecross.append((-intstreamwise[it])*DpsiDt[it])
                        instanttiltstretchcross.append((intstreamwise[it]*Vloc[it]*dpsids[it])+(intcrosswise[it]*Vloc[it]*dpsidn[it])+(intvert[it]*Vloc[it]*dpsidz[it]))
                        instanttiltstream2cross.append(intstreamwise[it]*Vloc[it]*dpsids[it])
                        instanttiltvert2cross.append(intvert[it]*Vloc[it]*dpsidz[it])
                        instantstretchcross.append(intcrosswise[it]*Vloc[it]*dpsidn[it])
                        instantbarocliniccross.append(-dBds[it])
                        instantmixingcross.append(idiffcrosswise[it]+turbcrosswise[it])

                        instantstretchvert.append(intvert[it]*dwdz[it])
                        instanttiltstream2vert.append(intstreamwise[it]*dwds[it])
                        instanttiltcross2vert.append(intcrosswise[it]*dwdn[it])
                        instantbaroclinicvert.append(invrhov2[it]*((drhovdn[it]*dpds[it])-(drhovds[it]*dpdn[it])))
                        instantmixingvert.append(zetaidiff[it]+zetaturb[it])
                    if it > 0:
                        #Hold terms in temp variables to make code cleaner
                        exchangestreamtemp = (intcrosswise[it-1]*DpsiDt[it-1])*dt
                        tiltstretchstreamtemp = ((intstreamwise[it-1]*dVds[it-1])+(intcrosswise[it-1]*dVdn[it-1])+(intvert[it-1]*dVdz[it-1]))*dt
                        tiltcross2streamtemp = (intcrosswise[it-1]*dVdn[it-1])*dt
                        tiltvert2streamtemp = (intvert[it-1]*dVdz[it-1])*dt
                        stretchstreamtemp = (intstreamwise[it-1]*dVds[it-1])*dt
                        baroclinicstreamtemp = (dBdn[it-1])*dt
                        mixingstreamtemp = (idiffstreamwise[it-1]+turbstreamwise[it-1])*dt

                        #Add terms to the previous vorticity values for the integration
                        intexchangestream.append(intexchangestream[it-1]+exchangestreamtemp)
                        inttiltstretchstream.append(inttiltstretchstream[it-1]+tiltstretchstreamtemp)
                        inttiltcross2stream.append(inttiltcross2stream[it-1]+tiltcross2streamtemp)
                        inttiltvert2stream.append(inttiltvert2stream[it-1]+tiltvert2streamtemp)
                        intstretchstream.append(intstretchstream[it-1]+stretchstreamtemp)
                        intbaroclinicstream.append(intbaroclinicstream[it-1]+baroclinicstreamtemp)
                        intmixingstream.append(intmixingstream[it-1]+mixingstreamtemp)
                        intstreamwise.append(intstreamwise[it-1]+exchangestreamtemp+tiltstretchstreamtemp+baroclinicstreamtemp+mixingstreamtemp)
                        instantexchangestream.append(exchangestreamtemp/dt)
                        instanttiltstretchstream.append(tiltstretchstreamtemp/dt)
                        instanttiltcross2stream.append(tiltcross2streamtemp/dt)
                        instanttiltvert2stream.append(tiltvert2streamtemp/dt)
                        instantstretchstream.append(stretchstreamtemp/dt)
                        instantbaroclinicstream.append(baroclinicstreamtemp/dt)
                        instantmixingstream.append(mixingstreamtemp/dt)

                        #Hold terms in temp variables to make code cleaner
                        exchangecrosstemp = ((-intstreamwise[it-1])*DpsiDt[it-1])*dt
                        tiltstretchcrosstemp = ((intstreamwise[it-1]*Vloc[it-1]*dpsids[it-1])+(intcrosswise[it-1]*Vloc[it-1]*dpsidn[it-1])+(intvert[it-1]*Vloc[it-1]*dpsidz[it-1]))*dt
                        tiltstream2crosstemp = (intstreamwise[it-1]*Vloc[it-1]*dpsids[it-1])*dt
                        tiltvert2crosstemp = (intvert[it-1]*Vloc[it-1]*dpsidz[it-1])*dt
                        stretchcrosstemp = (intcrosswise[it-1]*Vloc[it-1]*dpsidn[it-1])*dt
                        barocliniccrosstemp = (-dBds[it-1])*dt
                        mixingcrosstemp = (idiffcrosswise[it-1]+turbcrosswise[it-1])*dt

                        #Add terms to the previous vorticity values for the integration
                        intexchangecross.append(intexchangecross[it-1]+exchangecrosstemp)
                        inttiltstretchcross.append(inttiltstretchcross[it-1]+tiltstretchcrosstemp)
                        inttiltstream2cross.append(inttiltstream2cross[it-1]+tiltstream2crosstemp)
                        inttiltvert2cross.append(inttiltvert2cross[it-1]+tiltvert2crosstemp)
                        intstretchcross.append(intstretchcross[it-1]+stretchcrosstemp)
                        intbarocliniccross.append(intbarocliniccross[it-1]+barocliniccrosstemp)
                        intmixingcross.append(intmixingcross[it-1]+mixingcrosstemp)
                        intcrosswise.append(intcrosswise[it-1]+exchangecrosstemp+tiltstretchcrosstemp+barocliniccrosstemp+mixingcrosstemp)
                        instantexchangecross.append(exchangecrosstemp/dt)
                        instanttiltstretchcross.append(tiltstretchcrosstemp/dt)
                        instanttiltstream2cross.append(tiltstream2crosstemp/dt)
                        instanttiltvert2cross.append(tiltvert2crosstemp/dt)
                        instantstretchcross.append(stretchcrosstemp/dt)
                        instantbarocliniccross.append(barocliniccrosstemp/dt)
                        instantmixingcross.append(mixingcrosstemp/dt)

                        #Hold terms in temp variables to make code cleaner
                        stretchverttemp = (intvert[it-1]*dwdz[it-1])*dt
                        tiltverttemp = ((intstreamwise[it-1]*dwds[it-1])+(intcrosswise[it-1]*dwdn[it-1]))*dt
                        tiltstream2verttemp = (intstreamwise[it-1]*dwds[it-1])*dt
                        tiltcross2verttemp = (intcrosswise[it-1]*dwdn[it-1])*dt
                        baroclinicverttemp = (invrhov2[it-1]*((drhovdn[it-1]*dpds[it-1])-(drhovds[it-1]*dpdn[it-1])))*dt
                        mixingverttemp = (zetaidiff[it-1]+zetaturb[it-1])*dt

                        #Add terms to the previous vorticity values for the integration
                        intstretchvert.append(intstretchvert[it-1]+stretchverttemp)
                        inttiltstream2vert.append(inttiltstream2vert[it-1]+tiltstream2verttemp)
                        inttiltcross2vert.append(inttiltcross2vert[it-1]+tiltcross2verttemp)
                        intbaroclinicvert.append(intbaroclinicvert[it-1]+baroclinicverttemp)
                        intmixingvert.append(intmixingvert[it-1]+mixingverttemp)
                        intvert.append(intvert[it-1]+stretchverttemp+tiltstream2verttemp+tiltcross2verttemp+baroclinicverttemp+mixingverttemp)
                        instantstretchvert.append(stretchverttemp/dt)
                        instanttiltstream2vert.append(tiltstream2verttemp/dt)
                        instanttiltcross2vert.append(tiltcross2verttemp/dt)
                        instantbaroclinicvert.append(baroclinicverttemp/dt)
                        instantmixingvert.append(mixingverttemp/dt)

                ###-------------------------------------------------------------------------------------------------------
                samesignstream = []
                samesigncross = []
                samesignvert = []
                for it in timeindicestoloop:
                    if (intstreamwise[it]>0.0 and modstreamwise[it]>0.0) or (intstreamwise[it]<0.0 and modstreamwise[it]<0.0):
                        samesignstream.append(1)
                    else:
                        samesignstream.append(0)
                for it in timeindicestoloop:
                    if (intcrosswise[it]>0.0 and modcrosswise[it]>0.0) or (intcrosswise[it]<0.0 and modcrosswise[it]<0.0):
                        samesigncross.append(1)
                    else:
                        samesigncross.append(0)
                for it in timeindicestoloop:
                    if (intvert[it]>0.0 and fullzvortpos[it]>0.0) or (intvert[it]<0.0 and fullzvortpos[it]<0.0):
                        samesignvert.append(1)
                    else:
                        samesignvert.append(0)

                ###------------------------------------------------------------------------------------------------------
                budgetpdf = '32_'+shortname+'_'+timetag+'min_series_[%d,%d,%d].pdf' %(tk,tj,ti)
                pdf = PdfPages(budgetpdf)

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                fig.set_figheight(10)
                fig.set_figwidth(16)
                ax1.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax1.plot(realtimes[:-1],intstreamwise,color='blue',label='Integrated streamwise')
                ax1.plot(realtimes[:-1],intexchangestream,color='red',label='Exchange')
                ax1.plot(realtimes[:-1],inttiltcross2stream,color='lightgreen',label='Tilt cross2stream')
                ax1.plot(realtimes[:-1],inttiltvert2stream,color='green',label='Tilt vert2stream')
                ax1.plot(realtimes[:-1],intstretchstream,color='gray',label='Stretch stream')
                ax1.plot(realtimes[:-1],intbaroclinicstream,color='purple',label='Baroclinic')
                ax1.plot(realtimes[:-1],intmixingstream,color='magenta',label='Mixing')
                ax1.plot(realtimes[:-1],modstreamwise,color='orange',label='Model streamwise')
                #ax1.plot(realtimes[:-1],residualstreamwise,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesignstream[it] == 0:
                        ax1.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax1.legend(loc='lower left')
                ax1.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nStreamwise vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax1.set_xlabel('Time (min)')
                ax1.set_ylabel('Vorticity (s-1)')
                ax1.set_ylim(-0.1,0.1)

                ax2.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax2.plot(realtimes[:-1],intcrosswise,color='blue',label='Integrated crosswise')
                ax2.plot(realtimes[:-1],intexchangecross,color='red',label='Exchange')
                ax2.plot(realtimes[:-1],inttiltstream2cross,color='lightgreen',label='Tilt stream2cross')
                ax2.plot(realtimes[:-1],inttiltvert2cross,color='green',label='Tilt vert2cross')
                ax2.plot(realtimes[:-1],intstretchcross,color='gray',label='Stretch cross')
                ax2.plot(realtimes[:-1],intbarocliniccross,color='purple',label='Baroclinic')
                ax2.plot(realtimes[:-1],intmixingcross,color='magenta',label='Mixing')
                ax2.plot(realtimes[:-1],modcrosswise,color='orange',label='Model crosswise')
                #ax2.plot(realtimes[:-1],residualcrosswise,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesigncross[it] == 0:
                        ax2.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax2.legend(loc='lower left')
                ax2.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nCrosswise vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('Vorticity (s-1)')
                ax2.set_ylim(-0.1,0.1)

                ax3.plot(realtimes[:-1],heightlist,color='gray',label='Trajectory height')
                ax3.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax3.legend()
                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Height (m)')

                ax4.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax4.plot(realtimes[:-1],intvert,color='blue',label='Integrated zeta')
                ax4.plot(realtimes[:-1],inttiltstream2vert,color='green',label='Tilt stream2zeta')
                ax4.plot(realtimes[:-1],inttiltcross2vert,color='red',label='Tilt cross2zeta')
                ax4.plot(realtimes[:-1],intstretchvert,color='gray',label='Stretch zeta')
                ax4.plot(realtimes[:-1],intbaroclinicvert,color='purple',label='Baroclinic zeta')
                ax4.plot(realtimes[:-1],intmixingvert,color='magenta',label='Mixing')
                ax4.plot(realtimes[:-1],fullzvortpos[0:len(realtimes)-1],color='orange',label='Model zeta')
                #ax4.plot(realtimes[:-1],residualvertical,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesignvert[it] == 0:
                        ax4.plot([realtimes[it],realtimes[it]],[-0.95,0.095],color='gray',alpha=0.25)
                ax4.legend(loc='lower left')
                ax4.set_title('Vertical vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax4.set_xlabel('Time (min)')
                ax4.set_ylabel('Vorticity (s-1)')
                ax4.set_ylim(-0.015,0.015)
                pdf.savefig(fig)
                pdf.close()

                ###-------------------------------------------------------------------------------------------------------
                budgetpdf = '32_'+shortname+'_'+timetag+'min_series_[%d,%d,%d]_instant.pdf' %(tk,tj,ti)
                pdf = PdfPages(budgetpdf)

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                fig.set_figheight(10)
                fig.set_figwidth(16)
                ax1.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax1.plot(realtimes[:-1],instantexchangestream,color='red',label='Exchange')
                ax1.plot(realtimes[:-1],instanttiltcross2stream,color='lightgreen',label='Tilt cross2stream')
                ax1.plot(realtimes[:-1],instanttiltvert2stream,color='green',label='Tilt vert2stream')
                ax1.plot(realtimes[:-1],instantstretchstream,color='gray',label='Stretch stream')
                ax1.plot(realtimes[:-1],instantbaroclinicstream,color='purple',label='Baroclinic')
                ax1.plot(realtimes[:-1],instantmixingstream,color='magenta',label='Mixing')
                #ax1.plot(realtimes[:-1],instantresidualstreamwise,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesignstream[it] == 0:
                        ax1.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax1.legend()
                ax1.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nInstantaneaous streamwise terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax1.set_xlabel('Time (min)')
                ax1.set_ylabel('dstreamwise/dt (s-2)')
                ax1.set_ylim(-0.0005,0.0005)

                ax2.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax2.plot(realtimes[:-1],instantexchangecross,color='red',label='Exchange')
                ax2.plot(realtimes[:-1],instanttiltstream2cross,color='lightgreen',label='Tilt stream2cross')
                ax2.plot(realtimes[:-1],instanttiltvert2cross,color='green',label='Tilt vert2cross')
                ax2.plot(realtimes[:-1],instantstretchcross,color='gray',label='Stretch cross')
                ax2.plot(realtimes[:-1],instantbarocliniccross,color='purple',label='Baroclinic')
                ax2.plot(realtimes[:-1],instantmixingcross,color='magenta',label='Mixing')
                #ax2.plot(realtimes[:-1],instantresidualcrosswise,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesigncross[it] == 0:
                        ax2.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax2.legend()
                ax2.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nInstantaneaous crosswise terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('dcrosswise/dt (s-2)')
                ax2.set_ylim(-0.0005,0.0005)

                ax3.plot(realtimes[:-1],heightlist,color='black',label='Trajectory height')
                ax3.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax3.legend()
                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Height (m)')

                ax4.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax4.plot(realtimes[:-1],instanttiltstream2vert,color='green',label='Tilt stream2zeta')
                ax4.plot(realtimes[:-1],instanttiltcross2vert,color='red',label='Tilt cross2zeta')
                ax4.plot(realtimes[:-1],instantstretchvert,color='gray',label='Stretch zeta')
                ax4.plot(realtimes[:-1],instantbaroclinicvert,color='purple',label='Baroclinic zeta')
                ax4.plot(realtimes[:-1],instantmixingvert,color='magenta',label='Mixing')
                #ax4.plot(realtimes[:-1],instantresidualvertical,color='magenta',linestyle='--',label='Residual')
                for it in timeindicestoloop:
                    if samesignvert[it] == 0:
                        ax4.plot([realtimes[it],realtimes[it]],[-0.95,0.095],color='gray',alpha=0.25)
                ax4.legend()
                ax4.set_title('Instantaneous vertical vorticity terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax4.set_xlabel('Time (min)')
                ax4.set_ylabel('dzeta/dt (s-2)')
                ax4.set_ylim(-0.0005,0.0005)
                pdf.savefig(fig)
                pdf.close()

                ###-----------------------------------------------------------------------------------------------------------
                ###Get error statistics
                ###Error magnitude (absolute value of the difference between model and integrated)
                errorstream = []
                errorcross = []
                errorvert = []
                for i in np.arange(len(intstreamwise)):
                    errorstream.append((abs(modstreamwise[i]-intstreamwise[i])))
                    errorcross.append((abs(modcrosswise[i]-intcrosswise[i])))
                    errorvert.append((abs(fullzvortpos[i]-intvert[i])))

                ###-----------------------------------------------------------------------------------------------------------
                np.save('32[%d,%d,%d]_modstream' %(tk,tj,ti),modstreamwise)
                np.save('32[%d,%d,%d]_modcross' %(tk,tj,ti),modcrosswise)

                np.save('32[%d,%d,%d]_instantexchangestream' %(tk,tj,ti),instantexchangestream)
                np.save('32[%d,%d,%d]_instanttiltcross2stream' %(tk,tj,ti),instanttiltcross2stream)
                np.save('32[%d,%d,%d]_instanttiltvert2stream' %(tk,tj,ti),instanttiltvert2stream)
                np.save('32[%d,%d,%d]_instantstretchstream' %(tk,tj,ti),instantstretchstream)
                np.save('32[%d,%d,%d]_instantbaroclinicstream' %(tk,tj,ti),instantbaroclinicstream)
                np.save('32[%d,%d,%d]_instantmixingstream' %(tk,tj,ti),instantmixingstream)
                np.save('32[%d,%d,%d]_intstreamwise' %(tk,tj,ti),intstreamwise)
                #np.save('30[%d,%d,%d]_residualstreamwise' %(tk,tj,ti),residualstreamwise)

                np.save('32[%d,%d,%d]_instantexchangecross' %(tk,tj,ti),instantexchangecross)
                np.save('32[%d,%d,%d]_instanttiltstream2cross' %(tk,tj,ti),instanttiltstream2cross)
                np.save('32[%d,%d,%d]_instanttiltvert2cross' %(tk,tj,ti),instanttiltvert2cross)
                np.save('32[%d,%d,%d]_instantstretchcross' %(tk,tj,ti),instantstretchcross)
                np.save('32[%d,%d,%d]_instantbarocliniccross' %(tk,tj,ti),instantbarocliniccross)
                np.save('32[%d,%d,%d]_instantmixingcross' %(tk,tj,ti),instantmixingcross)
                np.save('32[%d,%d,%d]_intcrosswise' %(tk,tj,ti),intcrosswise)
                #np.save('30[%d,%d,%d]_residualcrosswise' %(tk,tj,ti),residualcrosswise)

                np.save('32[%d,%d,%d]_instantstretchvert' %(tk,tj,ti),instantstretchvert)
                np.save('32[%d,%d,%d]_instanttiltstream2vert' %(tk,tj,ti),instanttiltstream2vert)
                np.save('32[%d,%d,%d]_instanttiltcross2vert' %(tk,tj,ti),instanttiltcross2vert)
                np.save('32[%d,%d,%d]_instantbaroclinicvert' %(tk,tj,ti),instantbaroclinicvert)
                np.save('32[%d,%d,%d]_instantmixingvert' %(tk,tj,ti),instantmixingvert)
                np.save('32[%d,%d,%d]_intvert' %(tk,tj,ti),intvert)
                #np.save('30[%d,%d,%d]_residualvertical' %(tk,tj,ti),residualvertical)

                np.save('32[%d,%d,%d]_errorstream' %(tk,tj,ti),errorstream)
                np.save('32[%d,%d,%d]_errorcross' %(tk,tj,ti),errorcross)
                np.save('32[%d,%d,%d]_errorvert' %(tk,tj,ti),errorvert)
