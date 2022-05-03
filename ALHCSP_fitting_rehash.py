#!/usr/bin/env python

import math
import matplotlib.pyplot as plot
import numpy as np
from lmfit import Parameters, Model
import os
from astropy.table import Table
from astropy.io import ascii
import shutil
import time

### Defining the parameters that are to be varied in the model ###

start_i = 73.0
start_theta = 77.0
start_phi = 60.0
start_pcos = 2.0
start_psin = 1.0
start_prat = 0.9

params = Parameters()
params.add('latLOS',   min=start_i - 5.0,  max=start_i + 6.0,  brute_step=2.0)
params.add('latHS',    min=start_theta - 5.0,  max=start_theta + 6.0,  brute_step=2.0)
params.add('longHS',   min=start_phi - 10.0,   max=start_phi + 11.0, brute_step=5.0)
params.add('cospower', min=start_pcos - 0.5,   max=start_pcos + 0.6,   brute_step=0.5)
params.add('sinpower', min=start_psin - 0.5,   max=start_psin + 0.6,   brute_step=0.5)
params.add('prat',     min=start_prat - 0.1,  max=start_prat + 0.2,   brute_step=0.1)

#Initializing all lists

All_latLOS = []
All_latHS = []
All_longHS = []
All_latHS2 = []
All_longHS2 = []
All_cospower = []
All_sinpower = []
All_prat = []
Residuals_Sum = []
All_chisqr = []
stats = []
Avg_error = []
ID = []
PF = []
LX = []
WATCH = []

#Defining constants:

title = 'HMXB_23'
source_period = ('1323')
products = ('032022')
name = 'HMXB_23'
period = 'HMXB_23'

mass = 1.4
radius = 10.0
G = 0.004302 
c = 2.99792458e5
rgpc = (2.0 * G * mass)/(c*c)
rg = rgpc * 3.08567758e13

#Defining initial parameters:
longLOS =        0.0
#prat =           1.0
mrat =           0.5
#sinpower =       1.0
per =            1.0        
pulse_fraction = 1.0

#calculating angular velocity
omega = (2.0 * math.pi)/per

### Functions ###

def pf(counts):
	
	PC = []
	C = []
	
	counts = np.array(counts)
	f_min = min(counts)
	pulsed_counts = [(x - f_min) for x in counts]
	pulsed_counts = np.array(pulsed_counts)
	
	for i in range(0, len(counts)):
		
		PC.append(pulsed_counts[i])
		C.append(counts[i])
		
	sum_pc = np.sum(PC)
	sum_c = np.sum(C)
	
	pulsed_fraction = sum_pc/sum_c
	pulsed_fraction = np.round(pulsed_fraction, 2)
	
	print ('Pulsed Fraction:    ' + str(pulsed_fraction))
	
	return pulsed_fraction

#Splitting a list:
def split_list(a_list):
            half = int(len(a_list)/2)
            return a_list[0:(half)]

#Converting to MJD:
def date_to_mjd(year, month, day):
	
	if month == 1 or month == 2:
		yearp = year - 1
		monthp = month + 12
	
	else:
		yearp = year
		monthp = month
	
	A = np.trunc(yearp/100.0)
	B = 2 - A + np.trunc(A/4.0)
	
	if yearp < 0:
		C = np.trunc((365.25 * yearp)-0.75)
	else:
		C = np.trunc(365.25 * yearp)
	
	D = np.trunc(30.6001 * (monthp + 1))
	
	jd = B + C + D + day + 1720994.5
	
	mjd = jd - 2400000.5
	
	return mjd

### getting info from file names ### 
def polestar_name(profile):
	
	obsid = profile.rsplit('_', 4)[0]
	energy_band = profile.rsplit('_', 4)[1]
	camera = profile.rsplit('_', 4)[2]
	
	return (obsid, energy_band, camera)
	
### loading the data ###    
def polestar_lc(each_file):
    
        P = []
        C = []
        E = []
        #PT = []
        #Cnorm = []
        #Enorm = []
        
        f = open(each_file, 'r')
        header = f.readline()
        
        for line in f:
            line = line.strip()
            columns = line.split()
            phase = float(columns[0])
            counts = float(columns[2])
            error = float(columns[3])
            C.append(counts)
            E.append(error)
            P.append(phase)
            
        f.close()
        
        #taking just the first period
        Pfinal = split_list(P)
        Cfinal = split_list(C)
        Efinal = split_list(E)
        
        # turn the strings in list into floats
        dataphase = np.array(Pfinal) + 0
        datacounts = np.array(Cfinal) + 0
        dataerror = np.array(Efinal) + 0
        
        data_points = len(dataphase)
        data_max = max(datacounts)
        data_mean = np.mean(datacounts)
        
        print ('# of data points:    ' + str(data_points))
        
        #normalizing the data
        datanorm = datacounts/data_max
        datacounts[datacounts==0] = 0.00001
        
        #normalizing the error
        percent_error = np.divide(dataerror, datacounts)
        percent_error = np.nan_to_num(percent_error)
        errornorm = np.multiply(percent_error, datanorm)
        
        #determining uncertainty for red chisqr
        uncertainty = np.mean(errornorm)
        errornorm[errornorm==0] = uncertainty
        error_norm_mean = len(datanorm) * [uncertainty]
        error_norm_mean = np.array(error_norm_mean)
        errornorm = len(datanorm) * [uncertainty]
        
        #calculating the variance for weighting purposes:
        var = np.divide(1.0, error_norm_mean)
        
        return(dataphase, datacounts, errornorm, datanorm, data_points, data_max, data_mean, var)
   
### Polestar for fitting ###    
def polestar_model(dataphase, latLOS, latHS, longHS, cospower, sinpower, prat):
           
     latHS2 = 180.0 - latHS
     longHS2 = 180.0 + longHS
     
     frat = 1.0 - prat
     
     #calculating the angles in the xy plane between the Hot Spot vectors and the Line Of Sight vector:
     thetadeg = longHS - longLOS
     theta = thetadeg*math.pi/180
     #longHSrad = longHS*math.pi/180

     theta2deg = longHS2 - longLOS
     theta2 = theta2deg*math.pi/180
     #longHS2rad = longHS2*math.pi/180

     #calculating the inclination angle from the xy plane of the LOS vector:
     phideg = 90 - latLOS
     phi = phideg*math.pi/180
     #latLOSrad = latLOS*math.pi/180

     #calculating the inclination angles from the xy plane of the HS vectors:
     gammadeg = 90 - latHS
     gamma = gammadeg*math.pi/180
     #latHSrad = latHS*math.pi/180

     gamma2deg = 90 - latHS2
     gamma2 = gamma2deg*math.pi/180
     #latHS2rad = latHS2*math.pi/180

     #creating the unit column vector for the Line Of Sight:
     LOS = np.array([math.cos(phi),0,math.sin(phi)]).T

     # creating the unit column vectors for the Hot Spots:
     mu = np.array([math.cos(theta)*math.cos(gamma),math.sin(theta)*math.cos(gamma),math.sin(gamma)]).T
     mu2 = np.array([math.cos(theta2)*math.cos(gamma2),math.sin(theta2)*math.cos(gamma2),math.sin(gamma2)]).T
     
     T = [] 
     #F = [] 
     FP = [] 
     FS = []
     
     t = 0.0001
     per = 1.0
     steps = len(dataphase)
     
     dtheta = omega*per/steps
     
     #creating the rotation matrix:
     x = [math.cos(dtheta), math.sin(dtheta),0]
     y = [-math.sin(dtheta), math.cos(dtheta),0]	
     z = [0,0,1]

     R = np.array((x,y,z), dtype=float)
     
     rat = rg/radius
     
     #redshift = 1.0/np.sqrt(1.0 - rat) - 1.0
     
     #loop that rotates NS in dtheta increments through one rotation
     
     while t < per:
         
         t += per/steps
         
         T.append(t/per)
         
         #rotating (and updating) the LOS unit vector (through angle dtheta):
         LOS = np.dot(R,LOS)
         
         #calculating the pencil beam component:
         
         #taking the dot products of LOS vector and HS vectors:
         coseta = np.dot(LOS,mu)
         coseta2 = np.dot(LOS,mu2)

         #calculating the angles (in degrees) between the LOS vector and the HS vectors:
         #eta = np.arccos(coseta)
         #etadeg = eta*180/math.pi
         
         #eta2 = np.arccos(coseta2)
         #eta2deg = eta2*180/math.pi
         
         #adding lightbending with Belodorov's cosine approximation:
         cosalpha = rat + coseta - rat*coseta
         cosalpha2 = rat + coseta2 - rat*coseta2
         
         #setting the products equal to zero if they are negative (i.e. hidden by the star):
         cosalpha = max(0.0, cosalpha)
         cosalpha2 = max(0.0, cosalpha2)
         
         #calculating the fan beam component:
         
         #taking cross products of LOS vector and HS vectors:
         sineta = np.linalg.norm(np.cross(LOS,mu))
         sineta2 = np.linalg.norm(np.cross(LOS,mu2))

         #adding lightbending with Belodorov's cosine approximation:
         sinalpha = rat + sineta - rat*sineta
         sinalpha2 = rat + sineta2 - rat*sineta2

         #setting the magnitude of the cross-product to zero if it is hidden by the star:
         
         if cosalpha == 0.0:
             sinalpha = 0
             
         if cosalpha2 == 0.0:
             sinalpha2 = 0

         #totalsinalpha = sinalpha + sinalpha2
         #totalcosalpha = cosalpha + cosalpha2
         
         #caluclating the total flux:
         #flux = (frat*(np.power(totalsinalpha, sinpower))) + (prat*(np.power(totalcosalpha, cospower)))
         fluxprimary = (frat*(np.power(sinalpha, sinpower))) + (prat*(np.power(cosalpha, cospower)))
         fluxsecondary = (frat*(np.power(sinalpha2, sinpower))) + (prat*(np.power(cosalpha2, cospower)))
         
         #ratio of strengths:
         fluxprim = fluxprimary*mrat
         fluxsec = fluxsecondary*(1.0-mrat)
         
         #populating the vertical axis of the lightcurve with total flux:
         FP.append(fluxprim)
         FS.append(fluxsec)
         FPA = np.array(FP)
         FSA = np.array(FS)
         
     #FT = np.sum([FP,FS], axis=0)
     FT = FPA + FSA
     
     #nomalizing the flux:
     FN = FT/(max(FT)*pulse_fraction)
     #FPN = FP/(max(FT)*pulse_fraction)
     #FSN = FS/(max(FT)*pulse_fraction)
     
     return FN
     
### Polestar for plotting purposes ###     
def polestar_plot(dataphase, latLOS, latHS, longHS, cospower, sinpower, prat):
     
     #longLOS = 0.0       
     latHS2 = 180.0 - latHS
     longHS2 = 180.0 + longHS
     
     frat = 1.0 - prat
     
     #calculating the angles in the xy plane between the Hot Spot vectors and the Line Of Sight vector:
     thetadeg = longHS - longLOS
     theta = thetadeg*math.pi/180
     #longHSrad = longHS*math.pi/180

     theta2deg = longHS2 - longLOS
     theta2 = theta2deg*math.pi/180
     #longHS2rad = longHS2*math.pi/180

     #calculating the inclination angle from the xy plane of the LOS vector:
     phideg = 90 - latLOS
     phi = phideg*math.pi/180
     #latLOSrad = latLOS*math.pi/180

     #calculating the inclination angles from the xy plane of the HS vectors:
     gammadeg = 90 - latHS
     gamma = gammadeg*math.pi/180
     #latHSrad = latHS*math.pi/180

     gamma2deg = 90 - latHS2
     gamma2 = gamma2deg*math.pi/180
     #latHS2rad = latHS2*math.pi/180

     #creating the unit column vector for the Line Of Sight:
     LOS = np.array([math.cos(phi),0,math.sin(phi)]).T

     # creating the unit column vectors for the Hot Spots:
     mu = np.array([math.cos(theta)*math.cos(gamma),math.sin(theta)*math.cos(gamma),math.sin(gamma)]).T
     mu2 = np.array([math.cos(theta2)*math.cos(gamma2),math.sin(theta2)*math.cos(gamma2),math.sin(gamma2)]).T
     
     T = []
     #F = []
     FP = []
     FS = []
     
     t = 0.0001
     per = 1.0
     steps = 1000.0
     
     dtheta = omega*per/steps
     
     #creating the rotation matrix:
     x = [math.cos(dtheta), math.sin(dtheta),0]
     y = [-math.sin(dtheta), math.cos(dtheta),0]	
     z = [0,0,1]

     R = np.array((x,y,z), dtype=float)
     
     rat = rg/radius
     
     #redshift = 1.0/np.sqrt(1.0 - rat) - 1.0
     
     #loop that rotates NS in dtheta increments through one rotation
     
     while t < per:
         
         t += per/steps
         
         T.append(t/per)
         
         #rotating (and updating) the LOS unit vector (through angle dtheta):
         LOS = np.dot(R,LOS)
         
         #calculating the pencil beam component:
         
         #taking the dot products of LOS vector and HS vectors:
         coseta = np.dot(LOS,mu)
         coseta2 = np.dot(LOS,mu2)

         #calculating the angles (in degrees) between the LOS vector and the HS vectors:
         #eta = np.arccos(coseta)
         #etadeg = eta*180/math.pi
         
         #eta2 = np.arccos(coseta2)
         #eta2deg = eta2*180/math.pi
         
         #adding lightbending with Belodorov's cosine approximation:
         cosalpha = rat + coseta - rat*coseta
         cosalpha2 = rat + coseta2 - rat*coseta2
         
         #setting the products equal to zero if they are negative (i.e. hidden by the star):
         cosalpha = max(0.0, cosalpha)
         cosalpha2 = max(0.0, cosalpha2)
         
         #calculating the fan beam component:
         
         #taking cross products of LOS vector and HS vectors:
         sineta = np.linalg.norm(np.cross(LOS,mu))
         sineta2 = np.linalg.norm(np.cross(LOS,mu2))

         #adding lightbending with Belodorov's cosine approximation:
         sinalpha = rat + sineta - rat*sineta
         sinalpha2 = rat + sineta2 - rat*sineta2

         #setting the magnitude of the cross-product to zero if it is hidden by the star:
         
         if cosalpha == 0.0:
             sinalpha = 0
             
         if cosalpha2 == 0.0:
             sinalpha2 = 0

         #cossquare = cosalpha*cosalpha
         #cossquare2 = cosalpha2*cosalpha2
         
         #sinsquare = sinalpha*sinalpha
         #sinsquare2 = sinalpha2*sinalpha2
         
         #totalsinalpha = sinalpha + sinalpha2
         #totalcosalpha = cosalpha + cosalpha2
         
         #caluclating the total flux:
         #flux = (frat*(np.power(totalsinalpha, sinpower))) + (prat*(np.power(totalcosalpha, cospower)))
         fluxprimary = (frat*(np.power(sinalpha, sinpower))) + (prat*(np.power(cosalpha, cospower)))
         fluxsecondary = (frat*(np.power(sinalpha2, sinpower))) + (prat*(np.power(cosalpha2, cospower)))
         
         #ratio of strengths:
         fluxprim = fluxprimary*mrat
         fluxsec = fluxsecondary*(1.0-mrat)
         
         #populating the vertical axis of the lightcurve with total flux:
         #F.append(flux)
         FP.append(fluxprim)
         FS.append(fluxsec)
         FPA = np.array(FP)
         FSA = np.array(FS)
         
     #FT = np.sum([FP,FS], axis=0)
     FT = FPA + FSA
     
     #nomalizing the flux:
     #FN = FT/(max(FT)*pulse_fraction)
     FPN = FP/(max(FT)*pulse_fraction)
     FSN = FS/(max(FT)*pulse_fraction)
     
     return (T, FPN, FSN)

### fitting routine ###     
def polestar_fitting(gmodel, datanorm, dataphase, var):
    
    print('Coarse fitting.....')
    
    time_start = time.time()
    
    result = gmodel.fit(datanorm, params, dataphase=dataphase, method='brute', weights=var)
    
    fval = result.brute_fval
    print ('Chi squared:    ' + str(np.round(fval, 2)))
    
    ###the model with parameter values###
    
    latLOS = result.params['latLOS'].value
    latHS = result.params['latHS'].value
    longHS = result.params['longHS'].value
    #latHS2 = result.params['latHS2'].value
    #longHS2 = result.params['longHS2'].value
    cospower = result.params['cospower'].value
    #pulse_fraction = result.params['pulse_fraction'].value
    sinpower = result.params['sinpower'].value
    prat = result.params['prat'].value
    #mrat = result.params['mrat'].value
    
    ### polishing the fit ###
        
    param = Parameters()
    
    param.add('latLOS',   min=latLOS - 1.0,  max=latLOS + 2.0,  brute_step=1.0)
    param.add('latHS',    min=latHS - 1.0,  max=latHS + 2.0,  brute_step=1.0)
    param.add('longHS',   min=longHS - 2.0,   max=longHS + 3.0, brute_step=2.0)
    #param.add('latHS2',   min=100.0, max=180.0, brute_step=5.0)
    #param.add('longHS2',  min=0.0,   max=360.0, brute_step=10.0)
    param.add('prat',     value=prat, vary=False)#min=(max((prat - 0.1), 0.0)),   max=(min((prat + 0.2), 1.0)),  brute_step=0.1)
    param.add('cospower', value=cospower, vary=False)#min=cospower - 0.1,   max=cospower + 0.2,   brute_step=0.1)
    param.add('sinpower', value=sinpower, vary=False)#min=sinpower - 0.5,   max=sinpower + 0.6,   brute_step=0.5)
    
    print ('Refining fit.....')
    
    result_fine = gmodel.fit(datanorm, param, dataphase=dataphase, method='brute', weights=var)
    
    fval_fine = result_fine.brute_fval  
    print ('Chi squared:    ' + str(np.round(fval_fine, 2)))
    
    latLOS = result_fine.params['latLOS'].value
    latHS = result_fine.params['latHS'].value
    longHS = result_fine.params['longHS'].value
    #latHS2 = result.params['latHS2'].value
    #longHS2 = result.params['longHS2'].value
    cospower = result_fine.params['cospower'].value
    sinpower = result_fine.params['sinpower'].value
    prat = result_fine.params['prat'].value
    
    time_stop = time.time()
    fitting_time = np.round((time_stop - time_start), 2)
    fitting_time_str = str(fitting_time)
    
    print ('Fitting Time:  ' + fitting_time_str + '  sec.')
    
    return (latLOS, latHS, longHS, cospower, result_fine, sinpower, prat, fitting_time)

### heatmap generation ###    
def polestar_heatmap(gmodel, datanorm, dataphase, var, latLOS, latHS, longHS, cospower, sinpower, prat, name, local_id):
    
    latLOS_str = str(latLOS)
    latHS_str = str(latHS)
    longHS_str = str(longHS)
    cospower_str = str(cospower)
    sinpower_str = str(sinpower)
    prat_str = str(prat)
    
    ### heatmap fitting ###
        
    paramx = Parameters()
    
    paramx.add('latLOS',   min=0.0,  max=91.0,  brute_step=1.0)
    paramx.add('latHS',    min=0.0,  max=91.0,  brute_step=1.0)
    paramx.add('longHS',   value=longHS, vary=False)
    paramx.add('cospower', value=cospower, vary=False)
    paramx.add('sinpower', value=sinpower, vary=False)
    paramx.add('prat',     value=prat, vary=False)
    
    print ('Generating Heat Maps.....')
    
    result_plot = gmodel.fit(datanorm, paramx, dataphase=dataphase, method='brute', weights=var)
    
    testing = result_plot.brute_Jout
    representation = result_plot.brute_grid        
    grid = np.array(result_plot.brute_grid)
    '''
    paramy = Parameters()
    
    paramy.add('latLOS',   min=0.0,  max=91.0,  brute_step=1.0)
    paramy.add('latHS',    value = latHS, vary=False)
    paramy.add('longHS',   value=longHS, vary=False)
    paramy.add('cospower', min=1.0,   max=3.05,   brute_step=0.05)
    paramy.add('sinpower', value=sinpower, vary=False)
    paramy.add('prat',     value=prat, vary=False)
    
    print ('One more.....')
    
    result_heat = gmodel.fit(datanorm, paramy, dataphase=dataphase, method='brute', weights=var)
    
    testing_heat = result_heat.brute_Jout
    representation_heat = result_heat.brute_grid        
    grid_heat = np.array(result_heat.brute_grid)
    '''    
    figg, axx = plot.subplots(1)
    xx, yy = representation
    valuee = np.array(testing)
    imagee = axx.pcolormesh(xx, yy, valuee, cmap='jet')
    cb1 = figg.colorbar(imagee)
    axx.set_xlabel('$i$')
    axx.set_ylabel('$\Theta$')
    cb1.set_label('$\chi^2$ $value$')
    #plot.title('$ID:$ ' + local_id + '     $(\phi$ $=$ ' + longHS_str + ',  $P_{cos}$ $=$ ' + cospower_str + ')', fontsize=14.0)
    plot.savefig(local_id + '_heat_map_i_theta.pdf', format='pdf')
    '''
    figh, axh = plot.subplots(1)
    xh, yh = representation_heat
    valueh = -1 * np.array(testing_heat)
    imageh = axh.pcolormesh(xh, yh, valueh)
    cb1h = figh.colorbar(imageh)
    axh.set_xlabel('$i$')
    axh.set_ylabel('$P_{cos}$')
    cb1h.set_label('- $\chi^2$ $value$')
    plot.title('$ID:$ ' + local_id + '$(\phi$ $=$ ' + longHS_str + ',  $\Theta$ $=$ ' + latHS_str + ')', fontsize=14.0)
    plot.savefig(local_id + '_heat_map_i_cospower.pdf', format='pdf')
    '''
    plot.close('all')
    
def polestar_stats(result_fine):
    
    report = result_fine.fit_report() 
    residual = result_fine.residual        
    chisqr = np.round(result_fine.chisqr, 2)
    chisqr_str = str(chisqr)
    
    return (report, residual, chisqr, chisqr_str)

### plotting profiles and residuals ###    
def polestar_plotting(dataphase, datacounts, datanorm, data_points, errornorm, residual, latLOS_str, latHS_str, longHS_str, cospower_str, name, title, chisqr_str, luminosity_str, lid, T, FPN, FSN):
    
    print ('Plotting pulse profiles.....')
    
    local_id = lid
    
    zero = [0] * data_points
    
    FN = FPN + FSN
        
    #creating a 2 period plot:
    FN2 = np.append(FN, FN)
    FPN2 = np.append(FPN, FPN)
    FSN2 = np.append(FSN, FSN)
    
    T1 = [x + 1.0 for x in T]
    T2 = np.append(T, T1)
    
    dataphase1 = [x + 1.0 for x in dataphase]
    datacounts1 = datacounts
    
    datacounts2 = np.append(datacounts, datacounts1)
    datacounts2 = [float(x) for x in datacounts2]
    datacounts2 = np.array(datacounts2)        
    dataphase2 = np.append(dataphase, dataphase1)
    dataphase2 = [float(x) for x in dataphase2]
    dataphase2 = np.array(dataphase2)
    datanorm2 = np.append(datanorm, datanorm)
    datanorm2 = [float(x) for x in datanorm2]
    datanorm2 = np.array(datanorm2)
    errornorm2 = np.append(errornorm, errornorm)
    errornorm2 = [float(x) for x in errornorm2]
    errornorm2 = np.array(errornorm2)
    residual2 = np.append(residual, residual)
    residual2 = [float(x) for x in residual2]
    residual2 = np.array(residual2)
    residual2 = np.around(residual2, 5)
    zero2 = np.append(zero, zero)
    zero2 = [float(x) for x in zero2]
    zero2 = np.array(zero2)
    
    #Binning for horizontal error bars:
    data_points = len(dataphase2)
    bins = [1.0/data_points] * data_points
    bins = [float(x) for x in bins]
    bins = np.array(bins)
    
    T2 = [float(x) for x in T2]
    T2 = np.array(T2)
    FN2 = [float(x) for x in FN2]
    FN2 = np.array(FN2)
    FPN2 = [float(x) for x in FPN2]
    FPN2 = np.array(FPN2)
    FSN2 = [float(x) for x in FSN2]
    FSN2 = np.array(FSN2)
    
    ### plotting pulse profile with model fit ###
    
    os.chdir(name + '/profile_plots')
    
    plot.figure(facecolor="white")
    plot.xlim(0,2)
    plot.ylim(0, 1.2)
    plot.plot(T2,FN2, 'k-', linewidth=1.5)
    plot.plot(dataphase2,datanorm2,'b.', T2,FPN2,'r--',T2,FSN2,'g--')
    plot.errorbar(dataphase2,datanorm2,lw=0.5, linestyle='None', xerr = bins, yerr=errornorm2, ecolor='b', capsize=0)
    plot.xlabel('$Phase$')
    plot.ylabel('$Normalized$ $Counts$')
    #plot.title('$Local$ $ID:$ ' + local_id + '   $\chi^2$ $=$ ' + chisqr_str + '   $L_X$ $=$ ' + luminosity_str + ' $x10^{35}$ $erg$ $s^{-1}$', fontsize=13.0) # +  '   $Date:$ ' + month_str + '/' + day_str + '/' + year_str, fontsize=13.0)
    
    ymax = 1.2 * max(datacounts)
    
    plot.gca().twinx()
    plot.plot(dataphase2, datacounts2, 'b.')
    plot.xlim(0,2)
    plot.ylim(0,ymax)
    plot.ylabel('$Count$ $Rate$ $(s^{-1})$')
    #plot.show()
    plot.savefig(local_id + '_' + title + '_' + chisqr_str + '_PP.pdf', format='pdf')
    
    T2 = [round(x, 4) for x in T2]
    FN2 = [round(x, 4) for x in FN2]
    FPN2 = [round(x, 4) for x in FPN2]
    FSN2 = [round(x, 4) for x in FSN2]
    dataphase2 = [round(x, 4) for x in dataphase2]
    datacounts2 = [round(x, 4) for x in datacounts2]
    bins = [round(x, 4) for x in bins]
    errornorm2 = [round(x, 4) for x in errornorm2]
    
    os.chdir('../profiles')
    
    model_table = Table([T2, FN2, FPN2, FSN2], names=('model_phase', 'model_counts', 'pencil_counts', 'fan_counts'))
    model_file_name = local_id + '_' + title + '_model_profile.txt'
    ascii.write(model_table, model_file_name)
    
    profile_table = Table([dataphase2, datacounts2, bins, errornorm2], names=('data_phase', 'data_counts', 'x_error', 'y_error'))
    profile_table_name = local_id + '_' + title + '_data_profile.txt'
    ascii.write(profile_table, profile_table_name)
    
    ### plotting residuals ###
    
    os.chdir('../residuals')
        
    plot.figure(facecolor="white")
    plot.ylim(-2, 2)
    plot.xlim(0, 2)
    plot.plot(dataphase2, residual2, 'go')
    plot.plot(dataphase2, zero2, 'k--')
    plot.errorbar(dataphase2, residual2, lw=0.3, linestyle='None', yerr=errornorm2)
    plot.xlabel('$Phase$')
    plot.ylabel('$Residuals$')
    #plot.title(local_id + ' Residuals')
    plot.savefig(local_id + '_' + title + '_res.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    plot.hist(residual2, density=True, stacked=True)
    plot.title('$Local$ $ID:$ ' + local_id + '    $Residuals$')
    plot.xlabel('$Residuals$')
    plot.ylabel('$Normalized$ $Number$ $of$ $Occurrences$')
    plot.xlim(-5.0, 5.0)
    plot.ylim(0.0, 0.7)
    
    res_tab = Table([dataphase2, residual2, zero2], names=('phase', 'residuals', 'zero'))
    ascii.write(res_tab, local_id + '_' + title + '_res_table.txt')
    
    '''
    xgauss = np.linspace(-5.0, 5.0, 1000)
    ygauss = (1.0/((2.0*np.pi)**0.5))*np.exp(-((xgauss*xgauss)/2.0))
    
    ax6 = plot.gca().twinx()
    plot.plot(xgauss, ygauss, 'r-', linewidth=2.0)
    plot.xlim(-5.0, 5.0)
    plot.ylim(0.0, 0.5)
    plot.ylabel('$Gaussian$ $Distribution$   $(\mu$ $=$ $0,$ $\sigma^2$ $=$ $1)$')
    '''
    plot.savefig(local_id + '_' + title + '_res_hist.pdf', format='pdf')
    
    plot.close('all')
    
    os.chdir('../..')
    
def polestar_parameters(local_id, title, report):
    
    file_name = local_id + '_' + title + '_params.txt'
    text_file = open(file_name, 'w')
    text_file.write(report)
    text_file.close()

def distribution_attributes(Residuals_Sum):
    
    N = len(Residuals_Sum)
    print ("Number of Residuals = %f" % N)

    res_sum = np.sum(Residuals_Sum)

    res_squared = [x**2 for x in Residuals_Sum]
    res_cubed = [x**3 for x in Residuals_Sum]
    res_fourth = [x**4 for x in Residuals_Sum]

    res_squared_sum = np.sum(res_squared)
    res_cubed_sum = np.sum(res_cubed)
    res_fourth_sum = np.sum(res_fourth)

    #Statistical Calculations:
    standard_deviation = np.sqrt(res_squared_sum/(N - 1.0))
    variance = standard_deviation**2

    #Fisher-Pearson coefficient of skewness:
    skewness = (res_cubed_sum/N)/(standard_deviation**3)

    #Excess kurtosis:
    kurtosis = ((res_fourth_sum/N)/(standard_deviation**4)) - 3.0

    print ("Standard Deviation = %f" % standard_deviation)
    print ("Variance = %f" % variance)
    print ("Skewness = %f" % skewness)
    print ("Kurtosis = %f" % kurtosis)

    standard_deviation_str = str(np.round(standard_deviation, 3))
    variance_str = str(np.round(variance, 3))
    skewness_str = str(np.round(skewness, 3))
    kurtosis_str = str(np.round(kurtosis, 3))
    
    return (standard_deviation_str, variance_str, skewness_str, kurtosis_str)

def distribution_plotting(Residuals_Sum, period, variance_str, skewness_str, kurtosis_str, title):
    
    print ('Plotting residual distributions.....')
    
    ax3 = plot.figure(facecolor="white")
    plot.hist(Residuals_Sum, bins=int(len(Residuals_Sum)/5.0), density=True, stacked=True, facecolor='grey')
    #plot.title('$Source:$ ' + period + '       $Normalized$ $Distribution$ $of$ $Residuals$')
    plot.xlabel('$Residuals$    $(\sigma^2$ $=$ ' + variance_str + '   $Skewness$ $=$ ' + skewness_str + '   $Kurtosis$ $=$ ' + kurtosis_str + '$)$')
    plot.ylabel('$Normalized$ $Number$ $of$ $Occurrences$')
    plot.xlim(-10.0, 10.0)
    plot.ylim(0.0, 0.5)
    #plot.figtext(-4.5, 0.45, 'Skewness = %f' % skewness)

    xgauss = np.linspace(-5.0, 5.0, 1000)
    ygauss = (1.0/((2.0*np.pi)**0.5))*np.exp(-((xgauss*xgauss)/2.0))

    ax4 = plot.gca().twinx()
    plot.plot(xgauss, ygauss, 'r-', linewidth=2.0)
    plot.xlim(-5.0, 5.0)
    plot.ylim(0.0, 0.5)
    plot.ylabel('$Gaussian$ $Distribution$   $(\mu$ $=$ $0,$ $\sigma^2$ $=$ $1)$')
    plot.savefig(period + '_' + title + '_res_sum.pdf', format='pdf')

    plot.close('all')

def chisqr_plotting(log_LX, All_chisqr, period, title, All_latLOS, All_latHS):
    
    print('Creating chi-squared plots.....')
    
    plot.figure(facecolor="white")
    plot.plot(log_LX, All_chisqr, 'co')
    plot.title('$Source:$ ' + period + '      $\chi^2$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$Chi-Squared$ $Statistic$')
    plot.savefig(period + '_' + title + '_chisqr_vs_Lx.pdf', format='pdf')

    plot.figure(facecolor="white")
    plot.plot(MJD, All_chisqr, 'co')
    plot.title('$Source:$ ' + period + '      $\chi^2$ $vs.$ $MJD$')
    plot.xlabel('$MJD$')
    plot.ylabel('$Chi-Squared$ $Statistic$')
    plot.savefig(period + '_' + title + '_chisqr_vs_MJD.pdf', format='pdf')

    plot.figure(facecolor="white")
    plot.plot(All_latLOS, All_chisqr, 'co')
    plot.title('$Source:$ ' + period + '      $\chi^2$ $vs.$ $i$')
    plot.xlabel('$i$')
    plot.ylabel('$Chi-Squared$ $Statistic$')
    plot.savefig(period + '_' + title + '_chisqr_vs_i.pdf', format='pdf')

    plot.figure(facecolor="white")
    plot.plot(All_latHS, All_chisqr, 'co')
    plot.title('$Source:$ ' + period + '      $\chi^2$ $vs.$ $\Theta$')
    plot.xlabel('$Latitude$ $of$ $Primary$ $Hot$ $Spot$')
    plot.ylabel('$Chi-Squared$ $Statistic$')
    plot.savefig(period + '_' + title + '_chisqr_vs_theta.pdf', format='pdf')

    plot.close('all')

def luminosity_plotting(log_LX, All_latHS, period, title, All_latLOS, LOSHS, All_longHS, All_latHS2, All_longHS2, All_cospower, All_sinpower, All_prat, volume):
    
    print ('Generating luminosity plots.....')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_latHS[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $\Theta$ $vs.$ $L_X$')
    #plot.legend(['latHS','latLOS','longHS'], loc='best')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$\Theta$ $(degrees)$')
    plot.savefig(period + '_' + title + '_theta_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_latLOS[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $i$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$i$ $(degrees)$')
    plot.savefig(period + '_' + title + '_i_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], LOSHS[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '    $Sum$ $of$ $i$ $and$ $\Theta$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$Sum$ $of$ $LOS$ $and$ $HS$ $Latitudes$ $(degrees)$')
    plot.savefig(period + '_' + title + '_LOSHS_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_longHS[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $\phi_1$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$HS$ $Longitude$ $(degrees)$')
    plot.savefig(period + '_' + title + '_phi_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_latHS2[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $\Theta_2$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$HS2$ $Latitude$ $(degrees)$')
    plot.savefig(period + '_' + title + '_theta2_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_longHS2[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $\phi_2$ $vs.$ $L_X$')
    plot.xlabel('$log_{10}$ $L_{x}$ $(erg$ $s^{-1})$')
    plot.ylabel('$\phi_2$ $(degrees)$')
    plot.savefig(period + '_' + title + '_phi2_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_cospower[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $P_{cos}$ $vs.$ $L_X$')
    plot.xlabel('$Luminosity$ $(erg$ $s^{-1})$')
    plot.ylabel('$Power$ $of$ $Cosine$ $Function$')
    plot.savefig(period + '_' + title + '_cospower_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_sinpower[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $P_{sin}$ $vs.$ $L_X$')
    plot.xlabel('$Luminosity$ $(erg$ $s^{-1})$')
    plot.ylabel('$Power$ $of$ $Sine$ $Function$')
    plot.savefig(period + '_' + title + '_sinpower_vs_Lx.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    for i in range (0, len(log_LX)):
        plot.plot(log_LX[i], All_prat[i], markersize=volume[i], marker='o', color='cyan')
    plot.title('$Source:$ ' + period + '       $P_{rat}$ $vs.$ $L_X$')
    plot.xlabel('$Luminosity$ $(erg$ $s^{-1})$')
    plot.ylabel('$Ratio$ $of$ $Sine$ $to$ $Cosine$ $Function$')
    plot.savefig(period + '_' + title + '_prat_vs_Lx.pdf', format='pdf')
    
    plot.close('all') 

def histogram_plotting(period, title, All_latLOS, All_latHS, All_cospower, All_sinpower, All_prat, LOSHS):
    
    print ('Plotting histograms.....')
    
    plot.figure(facecolor="white")
    plot.hist(All_latLOS, bins = int(len(All_latLOS)/(len(All_latLOS)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $Inclination$ $Angle$')
    plot.xlabel('$i$ $(degrees)$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_inclination_angle_hist.pdf', format='pdf')

    plot.figure(facecolor="white")
    plot.hist(All_latHS, bins = int(len(All_latHS)/(len(All_latHS)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $\Theta$')
    plot.xlabel('$\Theta$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_theta_hist.pdf', format='pdf')

    plot.figure(facecolor="white")
    plot.hist(All_cospower, bins = int(len(All_cospower)/(len(All_cospower)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $P_{cos}$')
    plot.xlabel('$Power$ $of$ $Cosine$ $Function$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_cospower_hist.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    plot.hist(All_sinpower, bins = int(len(All_sinpower)/(len(All_sinpower)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $P_{sin}$')
    plot.xlabel('$Power$ $of$ $Sine$ $Function$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_sinpower_hist.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    plot.hist(All_prat, bins = int(len(All_prat)/(len(All_prat)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $P_{rat}$')
    plot.xlabel('$Ratio$ $of$ $Sine$ $to$ $Cosine$ $Function$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_prat_hist.pdf', format='pdf')
    
    plot.figure(facecolor="white")
    plot.hist(LOSHS, bins = int(len(LOSHS)/(len(LOSHS)/10)), facecolor='grey')
    plot.title('$Source:$ ' + period + '       $Sum$ $of$ $i$ $and$ $\Theta$')
    plot.xlabel('$Sum$ $of$ $Theta$ $and$ $i$')
    plot.ylabel('$Number$ $of$ $Occurrences$')
    plot.savefig(period + '_' + title + '_LOSHS_hist.pdf', format='pdf')

    plot.close('all')
    
    hist_tab = Table([All_latLOS, All_latHS, All_cospower, All_sinpower, All_prat, LOSHS], names=('i', 'theta', 'cospower', 'sinpower', 'prat', 'i+theta'))
    hist_tab_name = period + '_' + title + '_histogram_table.txt'
    ascii.write(hist_tab, hist_tab_name)

os.chdir('../fitting_rehash')

if not os.path.exists(os.getcwd() + '/' + name):
	
	os.mkdir(name)
	
else:
	
	shutil.rmtree(os.getcwd() + '/' + name)
	os.mkdir(name)
	
os.chdir(name)
os.mkdir('profiles')
os.mkdir('profile_plots')
os.mkdir('tables')
os.mkdir('residuals')
os.mkdir('histograms')
os.chdir('../../uncertainty_profiles/' + period + '/profiles')

profiles = os.listdir(os.getcwd())

ii = 0
counter = 0

MJD = '0'
luminosity_str = '32'

for profile in sorted(profiles):
    
    counter += 1

    lid = profile.rsplit('_', 4)[4]
    lid = lid.rsplit('.', 1)[0]
	
    print ()
    print ('ID:    ' + lid)
	
    lc = polestar_lc(profile)
	
    dataphase =     lc[0]
    datacounts =    lc[1]
    errornorm =     lc[2]
    datanorm =      lc[3]
    data_points =   lc[4]
    data_max =      lc[5]
    data_mean =     lc[6]
    var =           lc[7]
	
    pulsefrac = pf(datacounts)
	
    PF.append(pulsefrac)
	
    avg_error = np.mean(errornorm)
    avg_error_trunc = np.round(avg_error, 2)
	
    gmodel = Model(polestar_model)
	
    mod = polestar_fitting(gmodel, datanorm, dataphase, var)
	
    latLOS =        mod[0]
    latHS =         mod[1]
    longHS =        mod[2]
    cospower =      mod[3]
    result_fine =   mod[4]
    sinpower =      mod[5]
    prat =          mod[6]
    fitting_time =  mod[7]
	
    latLOS_str =    str(latLOS)
    latHS_str =     str(latHS)
    longHS_str =    str(longHS)
    cospower_str =  str(cospower)
    sinpower_str =  str(sinpower)
    prat_str =      str(prat)
	
    stats = polestar_stats(result_fine)
	
    report =        stats[0]
    residual =      stats[1]
    chisqr =        stats[2]
    chisqr_str =    stats[3]
	
    Residuals_Sum.extend(residual)
    All_chisqr.append(chisqr)
	
    plotting = polestar_plot(dataphase, latLOS, latHS, longHS, cospower, sinpower, prat)
	
    T =             plotting[0]
    FPN =  np.array(plotting[1])
    FSN =  np.array(plotting[2])
	
    os.chdir('../../../fitting_rehash')
	
    polestar_plotting(dataphase, datacounts, datanorm, data_points, errornorm, residual, latLOS_str, latHS_str, longHS_str, cospower_str, name, title, chisqr_str, luminosity_str, lid, T, FPN, FSN)
	
    #os.chdir(products + '/' + title + '/parameters_text')
	
    #polestar_parameters(lid, title, report)
	
	#os.chdir('../heat_maps')
	
	#polestar_heatmap(gmodel, datanorm, dataphase, var, latLOS, latHS, longHS, cospower, sinpower, prat, name, local_id)
	
    latHS2 = 180.0 - latHS
    longHS2 = 180.0 + longHS
	
    All_latLOS.append(latLOS)
    All_latHS.append(latHS)
    All_longHS.append(longHS)
    All_latHS2.append(latHS2)
    All_longHS2.append(longHS2)
    All_cospower.append(cospower)
    All_sinpower.append(sinpower)
    All_prat.append(prat)
    Avg_error.append(avg_error_trunc)
    ID.append(lid)
    WATCH.append(fitting_time)
    
    time_remain = (100.0*np.mean(WATCH) - sum(WATCH))/60.0
    time_rem_str = str(np.round(time_remain, 2))
    
    print('Approx. Time Remaining:  ' + time_rem_str + '   min.')
	
    os.chdir('../uncertainty_profiles/' + period + '/profiles')
           
### subtracting a revolution from longHS2 if longHS2 > 360 ###

for i in range(0, len(All_longHS2)):
    if All_longHS2[i] > 360.0:
        All_longHS2[i] = All_longHS2[i] - 360.0
    else:
        All_longHS2[i] = All_longHS2[i]
        
for i in range(0, len(All_longHS)):
    if All_longHS[i] > 360.0:
        All_longHS[i] = All_longHS[i] - 360.0
    else:
        All_longHS[i] = All_longHS[i]

os.chdir('../../../fitting_rehash/' + period + '/residuals')

dist_at = distribution_attributes(Residuals_Sum)

standard_deviation_str =    dist_at[0]
variance_str =              dist_at[1]
skewness_str =              dist_at[2]
kurtosis_str =              dist_at[3]

distribution_plotting(Residuals_Sum, period, variance_str, skewness_str, kurtosis_str, title)

All_latHS =     np.array(All_latHS)
All_latLOS =    np.array(All_latLOS)
LOSHS =         All_latHS + All_latLOS

#log_LX = [x * 1e35 for x in LX]
#log_LX = np.log10(log_LX)

#os.chdir('../chi_squared')

#chisqr_plotting(log_LX, All_chisqr, period, title, All_latLOS, All_latHS)

#os.chdir('../luminosity')

#volume = [x/2.0 for x in All_chisqr]

#luminosity_plotting(log_LX, All_latHS, period, title, All_latLOS, LOSHS, All_longHS, All_latHS2, All_longHS2, All_cospower, All_sinpower, All_prat, volume)

os.chdir('../histograms')

histogram_plotting(period, title, All_latLOS, All_latHS, All_cospower, All_sinpower, All_prat, LOSHS)

os.chdir('../tables')

## creating and sorting output data table ##

tab = Table([ID, All_latLOS, All_latHS, All_longHS, All_latHS2, All_longHS2, All_cospower, All_sinpower, All_prat, PF, All_chisqr, Avg_error], names=['ID', '$i$', '$\Theta$', '$\phi_1$', '$\Theta_2$', '$\phi_2$', '$P_{cos}$', '$P_{sin}$', '$P_{rat}$', 'Pulsed_Fraction', '$\chi^2$', 'Mean_Error'])
tab.sort('ID')

file_name = period + '_' + title + '_stats_MJD.txt'
file_name_latex = period + '_' + title + '_stats_MJD_latex.txt'
ascii.write(tab, file_name)
ascii.write(tab, file_name_latex, format='latex')

'''
Resdiuals_Sum = np.array(Residuals_Sum)

print (Residuals_Sum)

res_file_name = period + '_' + title + '_res_sum.txt'
res_tab = Table([Residuals_Sum], names=('Resdiuals'))
ascii.write(res_tab, res_file_name)
'''
