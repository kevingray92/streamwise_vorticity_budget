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
casename = '0_2sec'
shortname = 'CNTL_2sec'
tlvname = 'TLV1'
timetag = '057'
fname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min_restarts/sklinv'+casename+'.nc'
stormspd = []
stormspd = np.genfromtxt('/data/frame/a/kevintg2/cm1/fall2019/locationtxtfiles/0_1minspdavg.txt', dtype=None)
budgetpdf = shortname+'_'+tlvname+'_'+timetag+'min_strmwisevorttimeseries.pdf'
backtrajdir = '/data/keeling/a/kevintg2/a/cm1/paper2_summer2021/offlinetrajectories_backward/BACKtrajs_0_2sec_0-2km_zFIX/'
forwtrajdir = '/data/keeling/a/kevintg2/a/cm1/paper2_summer2021/offlinetrajectories_forward/FORWtrajs_0_2sec_0-2km_zFIX/'

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
#print('shape of oldbackxpos: ',oldbackxpos.shape)

#Load forward trajectories
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
#print('shape of forwxpos: ',forwxpos.shape)

###***This passes Kevin's check 07/28/21
#Open the 2 sec .nc file and get the times
ds = xr.open_dataset(fname)
timesinfile = ds.time
timesinfile

###***This passes Kevin's check 07/28/21
##Reverse the backward trajectory arrays

#Get the shape and length of each dimension
#print('shape of oldbackzposhASL: ',oldbackzposhASL.shape)
numxtraj = len(oldbackzposhASL[0,0,0,:])
numytraj = len(oldbackzposhASL[0,0,:,0])
numztraj = len(oldbackzposhASL[0,:,0,0])
numttraj = len(oldbackzposhASL[:,0,0,0])
#print('numx: ',numxtraj)
#print('numy: ',numytraj)
#print('numz: ',numztraj)
#print('numt: ',numttraj)

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
#print('t0: ',t0)
#print('shape of backzposhASL: ',backzposhASL.shape)
#print('oldbackxpos in time: ',oldbackxpos[:,0,0,0])
#print('newbackxpos in time: ',backxpos[:,0,0,0])

###***This passes Kevin's check 07/28/21
###New filtering cell
#print('shape of forwzposhASL: ',forwzposhASL.shape)
#print('example poshASL: ',forwzposhASL[:,0,0,0])

###FILTERING OF THE TRAJECTORIES
#Set up an array that holds filtering flag information for each traj
#filtertrajs = np.zeros((1, numztraj, numytraj, numxtraj)) #use np.zeros to assume no trajectories meet criteria first
#icount = np.arange(numxtraj)
#jcount = np.arange(numytraj)
#kcount = np.arange(numztraj)
#timecount = np.arange(len(forwxpos[:,0,0,0]))
##Set up variables that will hold information about traj with the max strmvort
#strmmaxi = 0
#strmmaxj = 0
#strmmaxk = 0
#strmmax = 0.0
##Count how many trajs are usable after filtering and if any nans show up (this shouldn't happen anymore after ZFIX in traj code)
#count = 0
#howmanynans = 0
#for i in icount:
#    for j in jcount:
#        for k in kcount:
#            zforchecking = forwzposhASL[:,k,j,i]
#            wforchecking = forwwpos[:,k,j,i]
#            uforchecking = forwupos[:,k,j,i]
#            vforchecking = forwvpos[:,k,j,i]
#            xvortforchecking = forwxvortpos[:,k,j,i]
#            yvortforchecking = forwyvortpos[:,k,j,i]
#            heightflag = 0 #Assume none of the trajectories meet criteria to begin with
#            nanflag = 0
#            for wcheck in wforchecking:
#                if wcheck > 15:
#                    heightflag = 1 #If the trajectory ever experiences a 30m/s updraft, consider it for analysis
#            for kcheck in zforchecking:
#                if (kcheck < 12.5) or (math.isnan(kcheck)):
#                    heightflag = 0 #Do not consider any trajectory that goes below 12.5 m (lowest scalar model level)
#                    if math.isnan(kcheck):
#                        nanflag =1 #Check for nans (this shouldn't happen any more)
#                        #print('i: ',i)
#                        #print('j: ',j)
#                        #print('k: ',k)
#            heightflag250 = 0 #A separate flag to make sure trajectory at least goes below 250 m.
#            if heightflag == 1:
#                for kcheck in zforchecking:
#                    if (kcheck < 250.0):
#                        heightflag250 = 1 #If traj met previous criteria, check if it at least goes below 250 m
#            if heightflag == 1 and heightflag250 == 1:
#                count = count+1
#                filtertrajs[0,k,j,i] = heightflag #Save filter information to filter array
#                for it in timecount:
#                    #calculate streamwise vort if the traj meets criteria
#                    strmwise = ((xvortforchecking[it]*uforchecking[it])+(yvortforchecking[it]*vforchecking[it]))/(((uforchecking[it]**2)+(vforchecking[it]**2))**(0.5))
#                    if strmwise > strmmax:
#                        strmmaxi = i
#                        strmmaxj = j
#                        strmmaxk = k
#                        strmmax = strmwise #Find the traj with max strmvort
#            howmanynans = howmanynans+nanflag
#            
#            #Delete old data to get ready for the new trajectory
#            del(zforchecking)
#            del(wforchecking)
#            del(uforchecking)
#            del(vforchecking)
#            del(xvortforchecking)
#            del(yvortforchecking)
#print('any nans?: ',howmanynans)
#print('Forward traj x location: ',forwxpos[0,0,0,0])
#print('Forward traj y location: ',forwypos[0,0,0,0])
#print('Forward traj z location: ',forwzpos[0,0,0,0])
#print('Backward traj x location: ',backxpos[298,0,0,0])
#print('Backward traj y location: ',backypos[298,0,0,0])
#print('Backward traj z location: ',backzpos[298,0,0,0])
#print('traj with max strmwise has strmwise of: ',strmmax)
#print('k: ',strmmaxk)
#print('j: ',strmmaxj)
#print('i: ',strmmaxi)
#print('is it ok? ',filtertrajs[0,5,4,11])#[0,8,6,9])#[0,strmmaxk,strmmaxj,strmmaxi])
#print('count: ',count)
###OR JUST LOAD A FILTERED TRAJ ARRAY
filtertrajs = np.load(forwtrajdir+timetag+'filtertrajs.npy')

###***This passes Kevin's check 07/28/21
#The timestep is 2 sec
dt = 2.0
dx = 250.0
dy = 250.0
#trajindices = [5,4,11]#[8,6,9]#[strmmaxk,strmmaxj,strmmaxi] #Choose your trajectory by inputting indices from the traj array

###***This passes Kevin's check 07/28/21
#Confirm we are using the right file
#print(fname)
#print(fullxpos.shape)

#Get umove and vmove and average storm speed
#umove = 14.5 #Don't think we need to convert to g-r winds since the s-r winds in a translating domain are essentially g-r.
#vmove =  4.5 #Tested this before. Using s-r winds makes physical sense
avgci = stormspd[0] #Still need storm speed for streamwise calculation
avgcj = stormspd[1]
#print('avgci: ', avgci)
#print('avgcj: ', avgcj)

#Open the .nc file and get grid dimensions and height values
ds = xr.open_dataset(fname)
xh = ds.xh
yh = ds.yh
zh = ds.z #1d array of scalar heights
height = ds.zh #4d array of scalar heights
xgrid = np.arange(0,len(xh))
ygrid = np.arange(0,len(yh))
zgrid = np.arange(0,len(zh))
#print('len of xgrid: ',len(xgrid))
#print('len of ygrid: ',len(ygrid))
#print('len of zgrid: ',len(zgrid))

#Get some variables from the .nc file
u  = ds.uinterp
v  = ds.vinterp
w  = ds.winterp
rho = ds.rho #dry air density
th0 = ds.th0[0,:,:,:]
qv0 = ds.qv[0,:,:,:]
qc0 = ds.qc[0,:,:,:]
qr0 = ds.qr[0,:,:,:]
qi0 = ds.qi[0,:,:,:]
qs0 = ds.qs[0,:,:,:]
qg0 = ds.qg[0,:,:,:]
qhl0 = ds.qhl[0,:,:,:]
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
xvort = ds.xvort
yvort = ds.yvort

###***This passes Kevin's check 08/11/21
#Define a function to get psi. The function takes u and v as input and returns a psi value.
#This cell has if statements based on quadrant
def get_psi(uforpsi,vforpsi):
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

count = 0
###The full array of initialized trajectories is [41,13,13]
for k in np.arange(15,21,1):
    for j in np.arange(0,13,1):
        for i in np.arange(0,13,1):
            print('traj %d, %d, %d filter: ' %(k,j,i), filtertrajs[0,k,j,i])
            if filtertrajs[0,k,j,i] == 1:

                tk = k
                tj = j
                ti = i
                print('Working on trajectory [%d, %d, %d]' %(tk,tj,ti))

                #Combine the back and forw trajs into one array
                fullxpos = np.append(backxpos[:,tk,tj,ti],forwxpos[1:,tk,tj,ti])
                fullypos = np.append(backypos[:,tk,tj,ti],forwypos[1:,tk,tj,ti])
                fullzpos = np.append(backzpos[:,tk,tj,ti],forwzpos[1:,tk,tj,ti])
                fullzposhASL = np.append(backzposhASL[:,tk,tj,ti],forwzposhASL[1:,tk,tj,ti])
                fullupos = np.append(backupos[:,tk,tj,ti],forwupos[1:,tk,tj,ti])
                fullvpos = np.append(backvpos[:,tk,tj,ti],forwvpos[1:,tk,tj,ti])
                fullwpos = np.append(backwpos[:,tk,tj,ti],forwwpos[1:,tk,tj,ti])
                fullxvortpos = np.append(backxvortpos[:,tk,tj,ti],forwxvortpos[1:,tk,tj,ti])
                fullyvortpos = np.append(backyvortpos[:,tk,tj,ti],forwyvortpos[1:,tk,tj,ti])
                fullzvortpos = np.append(backzvortpos[:,tk,tj,ti],forwzvortpos[1:,tk,tj,ti])
                #print('fullxpos: ',fullxpos)
                count = count+1

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
                print('time1: ', time1)
                print('time2: ', time2)
###---------------------------------------------------------------------------------------------------
###***This passes Kevin's check 08/11/21
##This cell creates an array of real times
                realtimes = []

                timeindicestoloop = np.arange(0,time2+1,1)#Use this when ready for the full time loop
                #timeindicestoloop = np.arange(0,4,1)#For testing
                for it in timeindicestoloop:
                    realtimes.append(47+(2.0/60.0)+(it*(2.0/60.0)))

###--------------------------------------------------------------------------------------------------
###***
##This cell creates an array of D(psi)/Dt using trajectory code output
#Set up arrays to hold variables and terms
                DpsiDt = [0]
                psiquadissueDt = []
                psiquadissueDttimes = []

                #Get time indices to loop over, start at 1 since we will need the first t-1 to be zero for psi time tendency term
                timeindicestoloop = np.arange(1,len(realtimes)-1,1)
                for it in timeindicestoloop:
    
                    #Get winds next to trajectory location using the interpolators
                    ubefore = fullupos[it-1]#+umove
                    uafter = fullupos[it+1]#+umove
                    vbefore = fullvpos[it-1]#+umove
                    vafter = fullvpos[it+1]#+umove
    
                    #Call the psi function
                    psibefore,quadbefore = get_psi(ubefore,vbefore)
                    psiafter,quadafter = get_psi(uafter,vafter)
                    if quadafter == 1 and quadbefore == 4:
                        psiafter = psiafter+(math.pi*2.0)
                        psiquadissueDt.append(it)
                        psiquadissueDttimes.append(realtimes[it])
                    if quadafter == 4 and quadbefore == 1:
                        psiquadissueDt.append(it)
                        psiquadissueDttimes.append(realtimes[it])
                    DpsiDt.append((psiafter-psibefore)/(2.0*dt))#Check this if we get some weird values when psi crosses due east (eg, 2-358 degrees).
    
#print('realtimes: ',realtimes)
#print('D(psi)/Dt: ',DpsiDt)

###-------------------------------------------------------------------------------------------
###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
#This cell creates an array of VERTICAL psi advection for the total psi derivative
#Set up arrays to hold variables and terms
                dpsidz = [0]
                verticalpsiadv = [0]
                heightlist = [fullzposhASL[0]]
                countuponly = 0
                uponlylist = []
                psiquadissuedz = []
                psiquadissuedztimes = []

                #Get time indices to loop over, start at 1 since we will need the first t-1 to be zero for psi time tendency term
                timeindicestoloop = np.arange(1,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
    
                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), u[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), v[it,:,:,:])
                    heightinterpolator  = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])
    
                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])
                    heightlist.append(zloc[0])
    
                    #Get winds next to trajectory location using the interpolators
                    wloc = fullwpos[it]
                    uup = uinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])#+umove
                    vup = vinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])#+vmove
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        udown = uinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])#+umove
                        vdown = vinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])#+vmove
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        udown = uinterpolator([0, fullypos[it], fullxpos[it]])#+umove
                        vdown = vinterpolator([0, fullypos[it], fullxpos[it]])#+vmove
                        zdown = heightinterpolator([0, fullypos[it], fullxpos[it]])
                        countuponly = countuponly+1
                        uponlylist.append(it)
    
                    dz = zup-zdown
    
                    #Call the psi function
                    psiup,quadup = get_psi(uup,vup)
                    psidown,quaddown = get_psi(udown,vdown)
                    if quadup == 1 and quaddown == 4:
                        psiquadissuedz.append(it)
                        psiquadissuedztimes.append(realtimes[it])
                        psiup = psiup+(math.pi*2.0)
                    if quadup == 4 and quaddown == 1:
                        psiquadissuedz.append(it)
                        psiquadissuedztimes.append(realtimes[it])
    
                    deltapsi = psiup-psidown
    
                    dpsidz.append(deltapsi/(dz[0]))#Check this if we get some weird values when psi crosses due east (eg, 2-358 degrees).
                    verticalpsiadv.append(wloc*dpsidz[it])
    
                    del(uinterpolator)
                    del(vinterpolator)
                    del(heightinterpolator)
                #print('realtimes: ',realtimes)
                #print('vertical psi adv: ',verticalpsiadv)

###--------------------------------------------------------------------------------------
###***This passes Kevin's check 08/11/21 FOR ONE TIME STEP
#This cell creates an array of V at trajectory location
#Set up arrays to hold variables and terms

                Vloc = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    Vloc.append(get_V(fullupos[it],fullvpos[it]))
#print('realtimes: ',realtimes)
#print('Vloc: ',Vloc)

###--------------------------------------------------------------------------------------
###***This passes Kevin's check 08/10/21 FOR ONE TIME STEP
#This cell creates an array of dVds and dVdn
#Set up arrays to hold variables and terms
                dVds = []
                dVdn = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
    
                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), u[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), v[it,:,:,:])
    
    ##############################################
                    dirtheta = math.atan(abs(fullvpos[it])/abs(fullupos[it]))
                    if (fullupos[it])>0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])>0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
        
                    #Get winds next to trajectory location using the interpolators
                    uR = uinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    vR = vinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    VR = get_V(uR,vR)
                    uL = uinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    vL = vinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    VL = get_V(uL,vL)
                    uA = uinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    vA = vinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    VA = get_V(uA,vA)
                    uB = uinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    vB = vinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    VB = get_V(uB,vB)
    
                    dVds.append((VA[0]-VB[0])/(2.0*dx))
                    dVdn.append((VL[0]-VR[0])/(2.0*dx))
    
                    del(uinterpolator)
                    del(vinterpolator)
#print('realtimes: ',realtimes)

###------------------------------------------------------------------------------------------------
###***This passes Kevin's check 08/10/21 FOR ONE TIME STEP
#This cell creates an array of d(psi)/ds and d(psi)/dn
#Set up arrays to hold variables and terms
                dpsids = []
                dpsidn = []
                psiquadissueds = []
                psiquadissuedn = []
                psiquadissuedstimes = []
                psiquadissuedntimes = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
    
                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), u[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), v[it,:,:,:])
    
                    ##############################################
                    dirtheta = math.atan(abs(fullvpos[it])/abs(fullupos[it]))
                    if (fullupos[it])>0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])>0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    #Get winds next to trajectory location using the interpolators
                    uR = uinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    vR = vinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    psiR,quadR = get_psi(uR,vR)
                    uL = uinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    vL = vinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    psiL,quadL = get_psi(uL,vL)
                    uA = uinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    vA = vinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    psiA,quadA = get_psi(uA,vA)
                    uB = uinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    vB = vinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    psiB,quadB = get_psi(uB,vB)
    
                    if (quadA == 1 and quadB == 4) or (quadA == 4 and quadB == 1):
                        psiquadissueds.append(it)
                        psiquadissuedstimes.append(realtimes[it])
                    if (quadL == 1 and quadR == 4) or (quadL == 4 and quadR == 1):
                        psiquadissuedn.append(it)
                        psiquadissuedntimes.append(realtimes[it])
                    if quadA == 1 and quadB == 4:
                        psiA = psiA+(math.pi*2.0)
                    if quadL == 1 and quadR == 4:
                        psiL = psiL+(math.pi*2.0)
    
                    dpsids.append((psiA-psiB)/(2.0*dx))
                    dpsidn.append((psiL-psiR)/(2.0*dx))
    
                    del(uinterpolator)
                    del(vinterpolator)
#print('psiquadissueds: ',psiquadissueds)
#print('psiquadissuedstimes: ',psiquadissuedstimes)
#print('psiquadissuedn: ',psiquadissuedn)
#print('psiquadissuedntimes: ',psiquadissuedntimes)

###----------------------------------------------------------------------------------------------------
###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
#This cell creates an array of dVdz
#Set up arrays to hold variables and terms
                dVdz = []
                countuponly = 0
                uponlylist = []

                #Get time indices to loop over, start at 0 since we don't need any t-1 terms
                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    uponlyflag = 0
    
                    #Set up interpolators
                    uinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), u[it,:,:,:])
                    vinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), v[it,:,:,:])
                    heightinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])
    
                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])
    
                    #Get winds next to trajectory location using the interpolators
                    uup = uinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])#+umove
                    vup = vinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])#+vmove
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        udown = uinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])#+umove
                        vdown = vinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])#+vmove
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        udown = uinterpolator([0, fullypos[it], fullxpos[it]])#+umove
                        vdown = vinterpolator([0, fullypos[it], fullxpos[it]])#+vmove
                        zdown = heightinterpolator([0, fullypos[it], fullxpos[it]])
                        uponlyflag = 1
                        countuponly = countuponly+1
                        uponlylist.append(it)
    
                    dz = zup-zdown
    
                    #Call the speed function
                    Vup = get_V(uup,vup)
                    Vdown = get_V(udown,vdown)
                    ###Not sure why we need this if statement here but not in the vertical psi advection cell...
                    if uponlyflag == 0:
                        dVdz.append((Vup[0]-Vdown[0])/(dz[0]))
                    else:
                        dVdz.append((Vup[0]-Vdown)/(dz[0]))
    
                    del(uinterpolator)
                    del(vinterpolator)
                    del(heightinterpolator)
#print('realtimes: ',realtimes)
#print('dVdz: ',dVdz)
#print('number of times we did an upward only derivative: ',countuponly)
#print('list of times we did an upward only derivative: ',uponlylist)

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
                    rhovloc = rholoc#+(rholoc*qvloc)#No qv correction
                    #rhovloc = rholoc+(rholoc*qvloc)#Yes qv correction
    
                    invrhov2.append(1/(rhovloc[0]**2.0))

                    ##############################################
                    dirtheta = math.atan(abs(fullvpos[it])/abs(fullupos[it]))
                    if (fullupos[it])>0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])>0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    rhoR = rhointerpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qvR = qvinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    rhovR = rhoR#+(rhoR*qvR)#No qv correction
                    #rhovR = rhoR+(rhoR*qvR)#Yes qv correction
                    rhoL = rhointerpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qvL = qvinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    rhovL = rhoL#+(rhoL*qvL)#No qv correction
                    #rhovL = rhoL+(rhoL*qvL)#Yes qv correction
                    prsR = prsinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    prsL = prsinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])

                    drhovdn.append((rhovL[0]-rhovR[0])/(2.0*dx))
                    dpdn.append((prsL[0]-prsR[0])/(2.0*dx))
    
                    rhoA = rhointerpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qvA = qvinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    rhovA = rhoA#+(rhoA*qvA)#No qv correction
                    #rhovA = rhoA+(rhoA*qvA)#Yes qv correction
                    rhoB = rhointerpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qvB = qvinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    rhovB = rhoB#+(rhoB*qvB)#No qv correction
                    #rhovB = rhoB+(rhoB*qvB)#Yes qv correction
                    prsA = prsinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    prsB = prsinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
    
                    drhovds.append((rhovA[0]-rhovB[0])/(2.0*dx))
                    dpds.append((prsA[0]-prsB[0])/(2.0*dx))
    
                    ##############################################
    
                    del(rhointerpolator)
                    del(qvinterpolator)
                    del(prsinterpolator)
#print('realtimes: ',realtimes)

###----------------------------------------------------------------------------------------------------
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
                    thinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), th[it,:,:,:])
                    qvinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qv[it,:,:,:])
                    qcinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qc[it,:,:,:])
                    qiinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qi[it,:,:,:])
                    qrinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qr[it,:,:,:])
                    qsinterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qs[it,:,:,:])
                    qginterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), qg[it,:,:,:])
                    qhlinterpolator      = RegularGridInterpolator((zgrid, ygrid, xgrid), qhl[it,:,:,:])
    
                    thrho0interpolator   = RegularGridInterpolator((zgrid, ygrid, xgrid), thrho0[:,:,:])
    
                    ##############################################
                    dirtheta = math.atan(abs(fullvpos[it])/abs(fullupos[it]))
                    if (fullupos[it])>0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])>0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    thR = thinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qvR = qvinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qcR = qcinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qiR = qiinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qrR = qrinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qsR = qsinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qgR = qginterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    qhlR = qhlinterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    thrho0R = thrho0interpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    thvR = thR*(1.0+(0.61*qvR))
                    thrhoR = thvR*((1.0+qvR)/(1.0+qvR+qcR+qrR+qiR+qsR+qgR+qhlR))
                    BR = g*((thrhoR-thrho0R)/thrho0R)
    
                    thL = thinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qvL = qvinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qcL = qcinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qiL = qiinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qrL = qrinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qsL = qsinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qgL = qginterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    qhlL = qhlinterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    thrho0L = thrho0interpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])
                    thvL = thL*(1.0+(0.61*qvL))
                    thrhoL = thvL*((1.0+qvL)/(1.0+qvL+qcL+qrL+qiL+qsL+qgL+qhlL))
                    BL = g*((thrhoL-thrho0L)/thrho0L)

                    dBdn.append((BL[0]-BR[0])/(2.0*dx))
    
                    thA = thinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qvA = qvinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qcA = qcinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qiA = qiinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qrA = qrinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qsA = qsinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qgA = qginterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    qhlA = qhlinterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    thrho0A = thrho0interpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    thvA = thA*(1.0+(0.61*qvA))
                    thrhoA = thvA*((1.0+qvA)/(1.0+qvA+qcA+qrA+qiA+qsA+qgA+qhlA))
                    BA = g*((thrhoA-thrho0A)/thrho0A)
    
                    thB = thinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qvB = qvinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qcB = qcinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qiB = qiinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qrB = qrinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qsB = qsinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qgB = qginterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    qhlB = qhlinterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    thrho0B = thrho0interpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
                    thvB = thB*(1.0+(0.61*qvB))
                    thrhoB = thvB*((1.0+qvB)/(1.0+qvB+qcB+qrB+qiB+qsB+qgB+qhlB))
                    BB = g*((thrhoB-thrho0B)/thrho0B)
    
                    dBds.append((BA[0]-BB[0])/(2.0*dx))
    
                    ##############################################
    
                    del(thinterpolator)
                    del(qvinterpolator)
                    del(qcinterpolator)
                    del(qiinterpolator)
                    del(qrinterpolator)
                    del(qsinterpolator)
                    del(qginterpolator)
                    del(qhlinterpolator)
                    del(thrho0interpolator)
#print('realtimes: ',realtimes)

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
                    winterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), w[it,:,:,:])
                    heightinterpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), height[it,:,:,:])
    
                    zloc = heightinterpolator([fullzpos[it], fullypos[it], fullxpos[it]])
    
                    #Get winds next to trajectory location using the interpolators
                    wup = winterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    zup = heightinterpolator([fullzpos[it]+1.0, fullypos[it], fullxpos[it]])
                    if fullzpos[it]>1.0:
                        wdown = winterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                        zdown = heightinterpolator([fullzpos[it]-1.0, fullypos[it], fullxpos[it]])
                    else:
                        wdown = winterpolator([0, fullypos[it], fullxpos[it]])
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

#print('number of times we did an upward only derivative: ',countuponly)
#print('array of times we did an upward only derivative: ',countuponlyarray)

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
                    winterpolator       = RegularGridInterpolator((zgrid, ygrid, xgrid), w[it,:,:,:])
    
                    ##############################################
                    dirtheta = math.atan(abs(fullvpos[it])/abs(fullupos[it]))
                    if (fullupos[it])>0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])>0.0:
                        deltaiL = -(math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = (math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])<0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = -(math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = -(math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    if (fullupos[it])>0.0 and (fullvpos[it])<0.0:
                        deltaiL = (math.sin(dirtheta))
                        deltajL = (math.cos(dirtheta))
                        deltaiR = -deltaiL
                        deltajR = -deltajL
                        deltaiA = (math.cos(dirtheta))
                        deltajA = -(math.sin(dirtheta))
                        deltaiB = -deltaiA
                        deltajB = -deltajA
                    wR = winterpolator([fullzpos[it], fullypos[it]+deltajR, fullxpos[it]+deltaiR])
                    wL = winterpolator([fullzpos[it], fullypos[it]+deltajL, fullxpos[it]+deltaiL])

                    dwdn.append((wL[0]-wR[0])/(2.0*dx))
    
                    wA = winterpolator([fullzpos[it], fullypos[it]+deltajA, fullxpos[it]+deltaiA])
                    wB = winterpolator([fullzpos[it], fullypos[it]+deltajB, fullxpos[it]+deltaiB])
    
                    dwds.append((wA[0]-wB[0])/(2.0*dx))
                    ##############################################
                    del(winterpolator)
#print('realtimes: ',realtimes)

###---------------------------------------------------------------------------------------------------
###***This passes Kevin's check 07/28/21 FOR ONE TIME STEP
#This cell is to get model total horizontal, streamwise, and crosswise vorticity at the trajectory location
#Set up arrays to hold variables and terms
                modcrosswise = []
                modstreamwise = []
                modtotal = []

                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    gru = fullupos[it]#+umove
                    grv = fullvpos[it]#+vmove
    
                    totvort = ((fullxvortpos[it]**2.0)+(fullyvortpos[it]**2.0))**(0.5)
                    modstreamwisetemp = (((fullxvortpos[it]*gru)+(fullyvortpos[it]*grv))/(((gru**2.0)+(grv**2.0))**(0.5)))
                    modcrosswisetemp = (((totvort**2.0)-(modstreamwisetemp**2.0))**(0.5))
    
                    #Find out if the crosswise vector is positive or negative
                    #Source: https://stackoverflow.com/questions/13221873/determining-if-one-2d-vector-is-to-the-right-or-left-of-another
                    if ((gru*(-fullyvortpos[it]))+(grv*fullxvortpos[it])) > 0.0:
                        modcrosswisetemp = -modcrosswisetemp
                    modstreamwise.append(modstreamwisetemp)
                    modcrosswise.append(modcrosswisetemp)
                    modtotal.append(totvort)
#print('modstreamwise: ',modstreamwise)
#print('modcrosswise: ',modcrosswise)
#print('total vorticity: ',modtotal)

###----------------------------------------------------------------------------------------------------------
#This cell integrates streamwise, crosswise, and vertical vorticity terms we calculated above
#Set up arrays to hold variables and terms

                intstreamwise = []
                intcrosswise = []
                intvert = []

                intexchangestream = []
                inttiltstretchstream = []
                inttiltcross2stream = []
                inttiltvert2stream = []
                intstretchstream = []
                intbaroclinicstream = []

                intexchangecross = []
                inttiltstretchcross = []
                inttiltstream2cross = []
                inttiltvert2cross = []
                intstretchcross = []
                intbarocliniccross = []

                intstretchvert = []
                inttiltstream2vert = []
                inttiltcross2vert = []
                intbaroclinicvert = []

                instantexchangestream = []
                instanttiltstretchstream = []
                instanttiltcross2stream = []
                instanttiltvert2stream = []
                instantstretchstream = []
                instantbaroclinicstream = []

                instantexchangecross = []
                instanttiltstretchcross = []
                instanttiltstream2cross = []
                instanttiltvert2cross = []
                instantstretchcross = []
                instantbarocliniccross = []

                instantstretchvert = []
                instanttiltstream2vert = []
                instanttiltcross2vert = []
                instantbaroclinicvert = []

                timeindicestoloop = np.arange(0,len(realtimes)-1,1)#Use this when ready for the full time loop
                for it in timeindicestoloop:
                    print('it: ',it)
                    print('working on time ',realtimes[it])

                    if it <= 1: #A higher number delays the integration, say after entering outflow surge
                        intstreamwise.append(modstreamwise[it])
                        intexchangestream.append(modstreamwise[it])
                        inttiltstretchstream.append(modstreamwise[it])
                        inttiltcross2stream.append(modstreamwise[it])
                        inttiltvert2stream.append(modstreamwise[it])
                        intstretchstream.append(modstreamwise[it])
                        intbaroclinicstream.append(modstreamwise[it])
        
                        intcrosswise.append(modcrosswise[it])
                        intexchangecross.append(modcrosswise[it])
                        inttiltstretchcross.append(modcrosswise[it])
                        inttiltstream2cross.append(modcrosswise[it])
                        inttiltvert2cross.append(modcrosswise[it])
                        intstretchcross.append(modcrosswise[it])
                        intbarocliniccross.append(modcrosswise[it])
        
                        intvert.append(fullzvortpos[it])
                        intstretchvert.append(fullzvortpos[it])
                        inttiltstream2vert.append(fullzvortpos[it])
                        inttiltcross2vert.append(fullzvortpos[it])
                        intbaroclinicvert.append(fullzvortpos[it])
        
                        instantexchangestream.append(0.0)
                        instanttiltstretchstream.append(0.0)
                        instanttiltcross2stream.append(0.0)
                        instanttiltvert2stream.append(0.0)
                        instantstretchstream.append(0.0)
                        instantbaroclinicstream.append(0.0)
        
                        instantexchangecross.append(0.0)
                        instanttiltstretchcross.append(0.0)
                        instanttiltstream2cross.append(0.0)
                        instanttiltvert2cross.append(0.0)
                        instantstretchcross.append(0.0)
                        instantbarocliniccross.append(0.0)
        
                        instantstretchvert.append(0.0)
                        instanttiltstream2vert.append(0.0)
                        instanttiltcross2vert.append(0.0)
                        instantbaroclinicvert.append(0.0)
                    if it > 1:
                        #Hold terms in temp variables to make code cleaner
                        exchangestreamtemp = (intcrosswise[it-1]*DpsiDt[it-1])*dt
                        tiltstretchstreamtemp = ((intstreamwise[it-1]*dVds[it-1])+(intcrosswise[it-1]*dVdn[it-1])+(intvert[it-1]*dVdz[it-1]))*dt
                        tiltcross2streamtemp = (intcrosswise[it-1]*dVdn[it-1])*dt
                        tiltvert2streamtemp = (intvert[it-1]*dVdz[it-1])*dt
                        stretchstreamtemp = (intstreamwise[it-1]*dVds[it-1])*dt
                        baroclinicstreamtemp = (dBdn[it-1])*dt
        
                        #Add terms to the previous vorticity values for the integration
                        intexchangestream.append(intexchangestream[it-1]+exchangestreamtemp)
                        inttiltstretchstream.append(inttiltstretchstream[it-1]+tiltstretchstreamtemp)
                        inttiltcross2stream.append(inttiltcross2stream[it-1]+tiltcross2streamtemp)
                        inttiltvert2stream.append(inttiltvert2stream[it-1]+tiltvert2streamtemp)
                        intstretchstream.append(intstretchstream[it-1]+stretchstreamtemp)
                        intbaroclinicstream.append(intbaroclinicstream[it-1]+baroclinicstreamtemp)
                        intstreamwise.append(intstreamwise[it-1]+exchangestreamtemp+tiltstretchstreamtemp+baroclinicstreamtemp)
                        instantexchangestream.append(exchangestreamtemp/dt)
                        instanttiltstretchstream.append(tiltstretchstreamtemp/dt)
                        instanttiltcross2stream.append(tiltcross2streamtemp/dt)
                        instanttiltvert2stream.append(tiltvert2streamtemp/dt)
                        instantstretchstream.append(stretchstreamtemp/dt)
                        instantbaroclinicstream.append(baroclinicstreamtemp/dt)
        
                        #Hold terms in temp variables to make code cleaner
                        exchangecrosstemp = ((-intstreamwise[it-1])*DpsiDt[it-1])*dt
                        tiltstretchcrosstemp = ((intstreamwise[it-1]*Vloc[it-1]*dpsids[it-1])+(intcrosswise[it-1]*Vloc[it-1]*dpsidn[it-1])+(intvert[it-1]*Vloc[it-1]*dpsidz[it-1]))*dt
                        tiltstream2crosstemp = (intstreamwise[it-1]*Vloc[it-1]*dpsids[it-1])*dt
                        tiltvert2crosstemp = (intvert[it-1]*Vloc[it-1]*dpsidz[it-1])*dt
                        stretchcrosstemp = (intcrosswise[it-1]*Vloc[it-1]*dpsidn[it-1])*dt
                        barocliniccrosstemp = (-dBds[it-1])*dt
        
                        #Add terms to the previous vorticity values for the integration
                        intexchangecross.append(intexchangecross[it-1]+exchangecrosstemp)
                        inttiltstretchcross.append(inttiltstretchcross[it-1]+tiltstretchcrosstemp)
                        inttiltstream2cross.append(inttiltstream2cross[it-1]+tiltstream2crosstemp)
                        inttiltvert2cross.append(inttiltvert2cross[it-1]+tiltvert2crosstemp)
                        intstretchcross.append(intstretchcross[it-1]+stretchcrosstemp)
                        intbarocliniccross.append(intbarocliniccross[it-1]+barocliniccrosstemp)
                        intcrosswise.append(intcrosswise[it-1]+exchangecrosstemp+tiltstretchcrosstemp+barocliniccrosstemp)
                        instantexchangecross.append(exchangecrosstemp/dt)
                        instanttiltstretchcross.append(tiltstretchcrosstemp/dt)
                        instanttiltstream2cross.append(tiltstream2crosstemp/dt)
                        instanttiltvert2cross.append(tiltvert2crosstemp/dt)
                        instantstretchcross.append(stretchcrosstemp/dt)
                        instantbarocliniccross.append(barocliniccrosstemp/dt)

                        #Hold terms in temp variables to make code cleaner
                        stretchverttemp = (intvert[it-1]*dwdz[it-1])*dt
                        tiltverttemp = ((intstreamwise[it-1]*dwds[it-1])+(intcrosswise[it-1]*dwdn[it-1]))*dt
                        tiltstream2verttemp = (intstreamwise[it-1]*dwds[it-1])*dt
                        tiltcross2verttemp = (intcrosswise[it-1]*dwdn[it-1])*dt
                        baroclinicverttemp = (invrhov2[it-1]*((drhovdn[it-1]*dpds[it-1])-(drhovds[it-1]*dpdn[it-1])))*dt

                        #Add terms to the previous vorticity values for the integration
                        intstretchvert.append(intstretchvert[it-1]+stretchverttemp)
                        inttiltstream2vert.append(inttiltstream2vert[it-1]+tiltstream2verttemp)
                        inttiltcross2vert.append(inttiltcross2vert[it-1]+tiltcross2verttemp)
                        intbaroclinicvert.append(intbaroclinicvert[it-1]+baroclinicverttemp)
                        intvert.append(intvert[it-1]+stretchverttemp+tiltstream2verttemp+tiltcross2verttemp+baroclinicverttemp)
                        instantstretchvert.append(stretchverttemp/dt)
                        instanttiltstream2vert.append(tiltstream2verttemp/dt)
                        instanttiltcross2vert.append(tiltcross2verttemp/dt)
                        instantbaroclinicvert.append(baroclinicverttemp/dt)

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
                budgetpdf = shortname+'_'+tlvname+'_'+timetag+'min_series_[%d,%d,%d].pdf' %(tk,tj,ti)
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
                ax1.plot(realtimes[:-1],modstreamwise,color='orange',label='Model streamwise')
                for it in timeindicestoloop:
                    if samesignstream[it] == 0:
                        ax1.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax1.legend()
                ax1.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nStreamwise vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax1.set_xlabel('Time (min)')
                ax1.set_ylabel('Vorticity (s-1)')
                ax1.set_ylim(-0.4,0.4)

                ax2.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax2.plot(realtimes[:-1],intcrosswise,color='blue',label='Integrated crosswise')
                ax2.plot(realtimes[:-1],intexchangecross,color='red',label='Exchange')
                ax2.plot(realtimes[:-1],inttiltstream2cross,color='lightgreen',label='Tilt stream2cross')
                ax2.plot(realtimes[:-1],inttiltvert2cross,color='green',label='Tilt vert2cross')
                ax2.plot(realtimes[:-1],intstretchcross,color='gray',label='Stretch cross')
                ax2.plot(realtimes[:-1],intbarocliniccross,color='purple',label='Baroclinic')
                ax2.plot(realtimes[:-1],modcrosswise,color='orange',label='Model crosswise')
                for it in timeindicestoloop:
                    if samesigncross[it] == 0:
                        ax2.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax2.legend()
                ax2.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nCrosswise vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('Vorticity (s-1)')
                ax2.set_ylim(-0.4,0.4)

                ax3.plot(realtimes[:-1],heightlist,color='gray',label='Trajectory height')
                ax3.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax3.legend()
                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Height (m)')

                ax4.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax4.plot(realtimes[:-1],intvert,color='blue',label='Integrated zeta')
                ax4.plot(realtimes[:-1],inttiltstream2vert,color='red',label='Tilt stream2zeta')
                ax4.plot(realtimes[:-1],inttiltcross2vert,color='magenta',label='Tilt cross2zeta')
                ax4.plot(realtimes[:-1],intstretchvert,color='green',label='Stretch zeta')
                ax4.plot(realtimes[:-1],intbaroclinicvert,color='purple',label='Baroclinic zeta')
                ax4.plot(realtimes[:-1],fullzvortpos[0:len(realtimes)-1],color='orange',label='Model zeta')
                for it in timeindicestoloop:
                    if samesignvert[it] == 0:
                        ax4.plot([realtimes[it],realtimes[it]],[-0.95,0.095],color='gray',alpha=0.25)
                ax4.legend()
                ax4.set_title('Vertical vorticity budget for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax4.set_xlabel('Time (min)')
                ax4.set_ylabel('Vorticity (s-1)')
                ax4.set_ylim(-0.1,0.1)
                pdf.savefig(fig)
                pdf.close()
###-------------------------------------------------------------------------------------------------------
                budgetpdf = shortname+'_'+tlvname+'_'+timetag+'min_series_[%d,%d,%d]_instant.pdf' %(tk,tj,ti)
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
                for it in timeindicestoloop:
                    if samesignstream[it] == 0:
                        ax1.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax1.legend()
                ax1.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nInstantaneaous streamwise terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax1.set_xlabel('Time (min)')
                ax1.set_ylabel('dstreamwise/dt (s-2)')
                ax1.set_ylim(-0.004,0.004)

                ax2.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax2.plot(realtimes[:-1],instantexchangecross,color='red',label='Exchange')
                ax2.plot(realtimes[:-1],instanttiltstream2cross,color='lightgreen',label='Tilt stream2cross')
                ax2.plot(realtimes[:-1],instanttiltvert2cross,color='green',label='Tilt vert2cross')
                ax2.plot(realtimes[:-1],instantstretchcross,color='gray',label='Stretch cross')
                ax2.plot(realtimes[:-1],instantbarocliniccross,color='purple',label='Baroclinic')
                for it in timeindicestoloop:
                    if samesigncross[it] == 0:
                        ax2.plot([realtimes[it],realtimes[it]],[-0.35,0.35],color='gray',alpha=0.25)
                ax2.legend()
                ax2.set_title(shortname+' '+tlvname+' surge at time '+timetag+' min\nInstantaneaous crosswise terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('dcrosswise/dt (s-2)')
                ax2.set_ylim(-0.004,0.004)

                ax3.plot(realtimes[:-1],heightlist,color='black',label='Trajectory height')
                ax3.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax3.legend()
                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Height (m)')

                ax4.plot([realtimes[0],realtimes[-1]],[0.0,0.0],linestyle='--',color='black')
                ax4.plot(realtimes[:-1],instanttiltstream2vert,color='red',label='Tilt stream2zeta')
                ax4.plot(realtimes[:-1],instanttiltcross2vert,color='magenta',label='Tilt cross2zeta')
                ax4.plot(realtimes[:-1],instantstretchvert,color='green',label='Stretch zeta')
                ax4.plot(realtimes[:-1],instantbaroclinicvert,color='purple',label='Baroclinic zeta')
                for it in timeindicestoloop:
                    if samesignvert[it] == 0:
                        ax4.plot([realtimes[it],realtimes[it]],[-0.95,0.095],color='gray',alpha=0.25)
                ax4.legend()
                ax4.set_title('Instantaneous vertical vorticity terms for trajectory [%d,%d,%d]' %(tk,tj,ti))
                ax4.set_xlabel('Time (min)')
                ax4.set_ylabel('dzeta/dt (s-2)')
                ax4.set_ylim(-0.004,0.004)
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
                np.save('[%d,%d,%d]_instantexchangestream' %(tk,tj,ti),instantexchangestream)
                np.save('[%d,%d,%d]_instanttiltstretchstream' %(tk,tj,ti),instanttiltstretchstream)
                np.save('[%d,%d,%d]_instanttiltcross2stream' %(tk,tj,ti),instanttiltcross2stream)
                np.save('[%d,%d,%d]_instanttiltvert2stream' %(tk,tj,ti),instanttiltvert2stream)
                np.save('[%d,%d,%d]_instantstretchstream' %(tk,tj,ti),instantstretchstream)
                np.save('[%d,%d,%d]_instantbaroclinicstream' %(tk,tj,ti),instantbaroclinicstream)
                np.save('[%d,%d,%d]_intstreamwise' %(tk,tj,ti),intstreamwise)

                np.save('[%d,%d,%d]_instantexchangecross' %(tk,tj,ti),instantexchangecross)
                np.save('[%d,%d,%d]_instanttiltstretchcross' %(tk,tj,ti),instanttiltstretchcross)
                np.save('[%d,%d,%d]_instanttiltstream2cross' %(tk,tj,ti),instanttiltstream2cross)
                np.save('[%d,%d,%d]_instanttiltvert2cross' %(tk,tj,ti),instanttiltvert2cross)
                np.save('[%d,%d,%d]_instantstretchcross' %(tk,tj,ti),instantstretchcross)
                np.save('[%d,%d,%d]_instantbarocliniccross' %(tk,tj,ti),instantbarocliniccross)
                np.save('[%d,%d,%d]_intcrosswise' %(tk,tj,ti),intcrosswise)

                np.save('[%d,%d,%d]_intstretchvert' %(tk,tj,ti),instantstretchvert)
                np.save('[%d,%d,%d]_inttiltstream2vert' %(tk,tj,ti),instanttiltstream2vert)
                np.save('[%d,%d,%d]_inttiltcross2vert' %(tk,tj,ti),instanttiltcross2vert)
                np.save('[%d,%d,%d]_intbaroclinicvert' %(tk,tj,ti),instantbaroclinicvert)
                np.save('[%d,%d,%d]_intvert' %(tk,tj,ti),intvert)

                np.save('[%d,%d,%d]_errorstream' %(tk,tj,ti),errorstream)
                np.save('[%d,%d,%d]_errorcross' %(tk,tj,ti),errorcross)
                np.save('[%d,%d,%d]_errorvert' %(tk,tj,ti),errorvert)
