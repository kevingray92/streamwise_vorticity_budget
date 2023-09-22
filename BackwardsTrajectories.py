import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
import time

#Use xarray to open model output and specify chunking if data set is large (set by user)
fname = '/data/frame/a/kevintg2/cm1/output/sickle/_0_1min_restarts/sklinv0_2sec_108.nc'
print(fname)
#ds = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/12ms_2000m_tug.nc', chunks={'nk': 4})
ds = xr.open_dataset(fname)

#Get model output dimensions
num_x = ds.nx #Number of gridpoints in x
num_y = ds.ny #Number of gridpoints in y
num_z = ds.nz #Number of gridpoints in z

x = np.arange(0,num_x,1)
y = np.arange(0,num_y,1)
z = np.arange(0,num_z,1)

#Number of parcels in vertical (can be more than number of vertical levels; set by user) 
num_seeds_z = 23

#Number of parcels in y (set by user) 
num_seeds_y = 1

#Number of parcels in x (set by user)
num_seeds_x = 23

#Number of time steps to run trajectories back (set by user) 
time_steps = 300

#Time step to start backward trajectories at (set by user) 
start_time_step = 299
timetag = '108'
#Variable to record at each parcel's location throughout trajectory (code can be easily modified to add more; set by user) 
var_name1 = 'uinterp'
var_name2 = 'vinterp'
var_name3 = 'winterp'
var_name4 = 'xvort'
var_name5 = 'yvort'
var_name6 = 'zvort'

#Set as 'Y' or 'N' for 'yes' or 'no' if the u, v, and w model output is on the staggered grid 
#(unless you have interpolated u, v, and w to the scalar grid, they are most likely on the staggered grid (set by user)
staggered = 'N'
#Enter umove and vmove values so that ground-relative winds are used
#umove = 14.5
#vmove =  4.5

#Horizontal resolution of model output (meters)
hor_resolution = (ds.xf[1].values-ds.xf[0].values)*1000

#Vertical resolution of model output (meters). Changes in x and y, if there is terrain, and z, if grid is stretched.
vert_resolution = ds.zh[0,1:,:,:].values-ds.zh[0,:-1,:,:].values 
                  
#Model output time step length (seconds)
time_step_length = (ds.time[1].values - ds.time[0].values)/np.timedelta64(1, 's')

xpos = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #x-location (grid points on staggered grid)
ypos = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #y-location (grid points on staggered grid)
zpos = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #z-location (grid points on staggered grid)
zpos_heightASL = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #Height above sea level (meters)
zpos_vert_res = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #Vertical grid spacing at parcel location (meters)
variable1 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable1 to track
variable2 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable2 to track
variable3 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable3 to track
variable4 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable4 to track
variable5 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable5 to track
variable6 = np.zeros((time_steps, num_seeds_z, num_seeds_y, num_seeds_x)) #User specified variable6 to track

#x-position
for i in range(num_seeds_x):
    xpos[0,:,:,i] = 257+(0.5*i) #This example initializes all seeds at same x-position (1000th x-grpt, set by user)

#y-position   
for i in range(num_seeds_y):
    ypos[0,:,i,:] = 316+(i) #This example initializes seeds evenly in y-dimension (0th, 4th, 8th, etc. y-grpt; set by user)

#z-position
for i in range(num_seeds_z):
    zpos[0,i,:,:] = 3+(0.5*i) #This example initializes seeds evenly starting in z-dimension (0th, 1st, 2nd, etc., z-grpt; set by user)

#Get height of surface
try:
    zs = ds.zs[0,:,:].values
    print('Output has terrain')
except: 
    zs = np.zeros((ds.ny,ds.nx))
    print('Output does not have terrain')

#Get height of vertical coordinates (scalar grid)
try:
    zh = ds.zh[0,:,:,:].values
    print('Output has terrain')
except:
    zh1d = (ds.z[:].values)*1000
    zh2d = np.repeat(zh1d,ds.ny, axis = 0).reshape(ds.nz, ds.ny)
    zh = np.repeat(zh2d,ds.nx, axis = 0).reshape(ds.nz, ds.ny, ds.nx)
    print('Output does not have terrain')

#Create list of initial coordinates to get height
xloc = (xpos[0,:,:,:]).flatten()
yloc = (ypos[0,:,:,:]).flatten()
zloc = (zpos[0,:,:,:]).flatten()
coord_height = []
for i in range(len(xloc)):
    coord_height.append((zloc[i], yloc[i], xloc[i]))

#Get the actual inital height of the parcels in meters above sea level
zpos_heightASL[0,:,:,:] = np.reshape(interpolate.interpn((z,y,x), zh, coord_height, method='linear', bounds_error=False, fill_value= 0), (num_seeds_z, num_seeds_y, num_seeds_x))

#Loop over all time steps and compute trajectory
height1D = ds.z
height1D = height1D*1000.0
for t in range(time_steps-1):
    
    start = time.time() #Timer
    
    ##########################################################################################################
    ##########################################################################################################   
    ##################### Get data for 'first guess' step of integration scheme ##############################
    ##########################################################################################################
    ##########################################################################################################
    
    #Get model data (set by user)
    u = ds.uinterp[start_time_step-t,:,:,:].values
    v = ds.vinterp[start_time_step-t,:,:,:].values
    #Make u and v ground-relative by adding umove and vmove
    #u = u+umove
    #v = v+vmove
    w = ds.winterp[start_time_step-t,:,:,:].values
    var1 = getattr(ds,var_name1)[start_time_step-t,:,:,:].values
    var2 = getattr(ds,var_name2)[start_time_step-t,:,:,:].values
    var3 = getattr(ds,var_name3)[start_time_step-t,:,:,:].values
    var4 = getattr(ds,var_name4)[start_time_step-t,:,:,:].values
    var5 = getattr(ds,var_name5)[start_time_step-t,:,:,:].values
    var6 = getattr(ds,var_name6)[start_time_step-t,:,:,:].values
        
    ############## Generate coordinates for interpolations ###############

    #x, y, and z on staggered and scalar grids
    xloc = np.copy(xpos[t,:,:,:]).flatten()
    xloc_stag = np.copy(xpos[t,:,:,:]+0.5).flatten()
    yloc = np.copy(ypos[t,:,:,:]).flatten()
    yloc_stag = np.copy(ypos[t,:,:,:]+0.5).flatten()
    zloc = np.copy(zpos[t,:,:,:]).flatten()
    zloc_stag = np.copy(zpos[t,:,:,:]+0.5).flatten()

    #If u, v, and w are staggered, generate three staggered sets of coordinates:
    #    1) u-grid (staggered in x)
    #    2) v-grid (staggered in y)
    #    3) w-grid (staggered in z)
    
    if staggered == 'Y':
        coord_u = []
        coord_v = []
        coord_w = []
        for i in range(len(xloc)):
            coord_u.append((zloc[i], yloc[i], xloc_stag[i])) 
            coord_v.append((zloc[i], yloc_stag[i], xloc[i])) 
            coord_w.append((zloc_stag[i], yloc[i], xloc[i])) 
    
    #If not, generate scalar coordinates
    else: 
        coord_u = []
        coord_v = []
        coord_w = []
        for i in range(len(xloc)):
            coord_u.append((zloc[i], yloc[i], xloc[i])) 
            coord_v.append((zloc[i], yloc[i], xloc[i])) 
            coord_w.append((zloc[i], yloc[i], xloc[i])) 
    
    #Scalar coordinates for all other variables
    coord = []
    for i in range(len(xloc)):
        coord.append((zloc[i], yloc[i], xloc[i])) 
    
    ##########################################################################################################   
    ########################## Integrate 'first guess' of parcel's new location ##############################
    ##########################################################################################################   

    
    #########################   Calc 'first guess' new xpos in grdpts   #######################################
    dx_0 = np.reshape(interpolate.interpn((z,y,x), u, coord_u, method='linear', bounds_error=False, fill_value=np.nan)*time_step_length/hor_resolution, (num_seeds_z, num_seeds_y, num_seeds_x))
    xpos_1 = xpos[t,:,:,:] - dx_0

    #########################   Calc 'first guess' new ypos in grdpts   #######################################
    dy_0 = np.reshape(interpolate.interpn((z,y,x), v, coord_v, method='linear', bounds_error=False, fill_value=np.nan)*time_step_length/hor_resolution, (num_seeds_z, num_seeds_y, num_seeds_x))
    ypos_1 = ypos[t,:,:,:] - dy_0

    #########################   Calc 'first guess' new zpos in meters above sea level ######################################
    dz_0 = np.reshape(interpolate.interpn((z,y,x), w, coord_w, method='linear', bounds_error=False, fill_value= 0)*time_step_length, (num_seeds_z, num_seeds_y, num_seeds_x))
    zpos_heightASL_1 = zpos_heightASL[t,:,:,:] - dz_0
    
    ############# Convert zpos from meters above sea level to gridpts abve surface for interpolation #########
    #Get vertical grid spacing at each parcel's location
    zpos_vert_res[t,:,:,:] = np.reshape(interpolate.interpn((z[:-1],y,x), vert_resolution, coord, method='linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))


    #Calculate change in surface height and change in parcel height
    xloc = np.copy(xpos[t,:,:,:]).flatten()
    yloc = np.copy(ypos[t,:,:,:]).flatten()
    coord_zs1 = []
    for i in range(len(xloc)):
        coord_zs1.append((yloc[i], xloc[i]))
        
    xloc = np.copy(xpos_1).flatten()
    yloc = np.copy(ypos_1).flatten()
    coord_zs2 = []
    for i in range(len(xloc)):
        coord_zs2.append((yloc[i], xloc[i]))
    
    #Change in surface height over last timestep
    zs1 = interpolate.interpn((y,x), zs, coord_zs1, method='linear', bounds_error=False, fill_value= np.nan)
    zs2 = interpolate.interpn((y,x), zs, coord_zs2, method='linear', bounds_error=False, fill_value= np.nan)
    zs_change = zs2-zs1
    
    #Change in parcel height over last times step
    zpos_heightASL_change = zpos_heightASL_1.flatten()-zpos_heightASL[t,:,:,:].flatten()
    
    #Calculate zpos in grdpts above surface
    zpos_1 = zpos[t,:,:,:] + np.reshape((zpos_heightASL_change - zs_change)/zpos_vert_res[t,:,:,:].flatten(), (num_seeds_z, num_seeds_y, num_seeds_x))
    
    #Prevent parcels from going into the ground
    zpos_heightASL_1 = zpos_heightASL_1.clip(min=0)
    zpos_1 = zpos_1.clip(min=0)
    
    
    #Calculate value of variable at each parcel's location
    variable1[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var1, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x)) 
    variable2[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var2, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
    variable3[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var3, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
    variable4[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var4, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
    variable5[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var5, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
    variable6[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var6, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))

    ##########################################################################################################
    ##########################################################################################################   
    ##################### Get data for 'correction' step of integration scheme ###############################
    ##########################################################################################################
    ##########################################################################################################
    
    
    
    #Get model data for next time step (set by user)
    t = t + 1
    u = ds.uinterp[start_time_step-t,:,:,:].values
    v = ds.vinterp[start_time_step-t,:,:,:].values
    #Make u and v ground-relative by adding umove and vmove
    #u = u+umove
    #v = v+vmove
    
    w = ds.winterp[start_time_step-t,:,:,:].values
    t = t - 1
        
        
    ############## Generate coordinates for interpolations ###############

    #x, y, and z on staggered and scalar grids
    xloc = np.copy(xpos_1).flatten()
    xloc_stag = np.copy(xpos_1+0.5).flatten()
    yloc = np.copy(ypos_1).flatten()
    yloc_stag = np.copy(ypos_1+0.5).flatten()
    zloc = np.copy(zpos_1).flatten()
    zloc_stag = np.copy(zpos_1+0.5).flatten()

    #If u, v, and w are staggered, generate three staggered sets of coordinates:
    #    1) u-grid (staggered in x)
    #    2) v-grid (staggered in y)
    #    3) w-grid (staggered in z)
    
    if staggered == 'Y':
        coord_u = []
        coord_v = []
        coord_w = []
        for i in range(len(xloc)):
            coord_u.append((zloc[i], yloc[i], xloc_stag[i])) 
            coord_v.append((zloc[i], yloc_stag[i], xloc[i])) 
            coord_w.append((zloc_stag[i], yloc[i], xloc[i])) 
    
    #If not, generate scalar coordinates
    else: 
        coord_u = []
        coord_v = []
        coord_w = []
        for i in range(len(xloc)):
            coord_u.append((zloc[i], yloc[i], xloc[i])) 
            coord_v.append((zloc[i], yloc[i], xloc[i])) 
            coord_w.append((zloc[i], yloc[i], xloc[i])) 
    
    #Scalar coordinates for all other variables
    coord = []
    for i in range(len(xloc)):
        coord.append((zloc[i], yloc[i], xloc[i])) 
        
    
    ##########################################################################################################   
    ########################## Integrate 'correction' of parcel's new location ###############################
    ##########################################################################################################   

    
    #########################   Calc 'correction' new xpos in grdpts   #######################################
    dx_1 = np.reshape(interpolate.interpn((z,y,x), u, coord_u, method='linear', bounds_error=False, fill_value=np.nan)*time_step_length/hor_resolution, (num_seeds_z, num_seeds_y, num_seeds_x))
    xpos[t+1,:,:,:] = xpos[t,:,:,:] - (dx_0 + dx_1)/2

    #########################   Calc 'correction' new ypos in grdpts   #######################################
    dy_1 = np.reshape(interpolate.interpn((z,y,x), v, coord_v, method='linear', bounds_error=False, fill_value=np.nan)*time_step_length/hor_resolution, (num_seeds_z, num_seeds_y, num_seeds_x))
    ypos[t+1,:,:,:] = ypos[t,:,:,:] - (dy_0 + dy_1)/2

    #########################   Calc 'correction' new zpos in meters above sea level ######################################
    dz_1 = np.reshape(interpolate.interpn((z,y,x), w, coord_w, method='linear', bounds_error=False, fill_value= 0)*time_step_length, (num_seeds_z, num_seeds_y, num_seeds_x))
    zpos_heightASL[t+1,:,:,:] = zpos_heightASL[t,:,:,:] - (dz_0 + dz_1)/2
    
    
    
    ############# Convert zpos from meters above sea level to gridpts abve surface for interpolation #########
    #Get vertical grid spacing at each parcel's location
    zpos_vert_res[t,:,:,:] = np.reshape(interpolate.interpn((z[:-1],y,x), vert_resolution, coord, method='linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))

    
    #Calculate change in surface height and change in parcel height
    xloc = np.copy(xpos[t,:,:,:]).flatten()
    yloc = np.copy(ypos[t,:,:,:]).flatten()
    coord_zs1 = []
    for i in range(len(xloc)):
        coord_zs1.append((yloc[i], xloc[i]))
        
    xloc = np.copy(xpos[t+1,:,:,:]).flatten()
    yloc = np.copy(ypos[t+1,:,:,:]).flatten()
    coord_zs2 = []
    for i in range(len(xloc)):
        coord_zs2.append((yloc[i], xloc[i]))
    
    #Change in surface height over last timestep
    zs1 = interpolate.interpn((y,x), zs, coord_zs1, method='linear', bounds_error=False, fill_value= np.nan)
    zs2 = interpolate.interpn((y,x), zs, coord_zs2, method='linear', bounds_error=False, fill_value= np.nan)
    zs_change = zs2-zs1
    
    #Change in parcel height over last timestep
    zpos_heightASL_change = zpos_heightASL[t+1,:,:,:].flatten()-zpos_heightASL[t,:,:,:].flatten()
    
    #Calculate zpos in grdpts above surface
    zpos[t+1,:,:,:] = zpos[t,:,:,:] + np.reshape((zpos_heightASL_change - zs_change)/zpos_vert_res[t,:,:,:].flatten(), (num_seeds_z, num_seeds_y, num_seeds_x))
    ##########################################################################################################
    indexbelow = 0
    print('before many loops')
    for i in np.arange(len(zpos_heightASL[t+1,0,0,:])):
        for j in np.arange(len(zpos_heightASL[t+1,0,:,0])):
            for k in np.arange(len(zpos_heightASL[t+1,:,0,0])):
                #if height1D[0] > zpos_heightASL[t+1,k,j,i]:
                #    indexbelow = 0
                for heightindex in np.arange(len(height1D)):
                    if height1D[heightindex] < zpos_heightASL[t+1,k,j,i]:
                        indexbelow = heightindex
                indexabove = indexbelow+1
                #print('indexbelow: ',indexbelow)
                #print('indexabove: ',indexabove)
                between = (zpos_heightASL[t+1,k,j,i]-height1D[indexbelow])/(height1D[indexabove]-height1D[indexbelow])
                zpos[t+1,k,j,i] = indexbelow+between
    
    #Prevent parcels from going into the ground
    zpos = zpos.clip(min=0)
    zpos_heightASL = zpos_heightASL.clip(min=0)    
        

    #Timer
    stop = time.time()
    print("Integration {:01d} took {:.2f} seconds".format(t, stop-start))
    

#Load variable data
t = time_steps-1
var1 = getattr(ds,var_name1)[start_time_step-t,:,:,:].values
var2 = getattr(ds,var_name2)[start_time_step-t,:,:,:].values
var3 = getattr(ds,var_name3)[start_time_step-t,:,:,:].values
var4 = getattr(ds,var_name4)[start_time_step-t,:,:,:].values
var5 = getattr(ds,var_name5)[start_time_step-t,:,:,:].values
var6 = getattr(ds,var_name6)[start_time_step-t,:,:,:].values

#Get get x, y, and z positions from scalar grid
xloc = np.copy(xpos[t,:,:,:]).flatten()
yloc = np.copy(ypos[t,:,:,:]).flatten()
zloc = np.copy(zpos[t,:,:,:]).flatten()
coord = []
for i in range(len(xloc)):
    coord.append((zloc[i], yloc[i], xloc[i])) 

#Variables
variable1[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var1, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
variable2[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var2, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
variable3[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var3, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
variable4[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var4, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
variable5[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var5, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))
variable6[t,:,:,:] = np.reshape(interpolate.interpn((z,y,x), var6, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (num_seeds_z, num_seeds_y, num_seeds_x))

#timetag = '057'
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'xpos', xpos)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'ypos', ypos)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'zpos', zpos)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'zpos_heightASL', zpos_heightASL)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name1, variable1)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name2, variable2)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name3, variable3)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name4, variable4)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name5, variable5)
np.save('BACKtrajs_0_2sec_0-2km_zFIX/'+timetag+'%s' %var_name6, variable6)
