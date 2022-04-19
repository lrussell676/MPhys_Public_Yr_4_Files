# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:17:55 2022

@author: Lewis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import time

import MC_funcs as mc # All MC functions in "MC_funcs.py", cleans up main code

sample_size = np.array([250000,1000000,4000000]) #0.25E6,1E6,4E6

""" -------------------------------------------------------------"""
""" ----------------------- TASK 1 ------------------------------"""
""" -------------------------------------------------------------"""

# The function developed for this can be found in "MC_funcs.py"


""" -------------------------------------------------------------"""
""" ----------------------- TASK 2 ------------------------------"""
""" -------------------------------------------------------------"""

def func_2a(x):
    return np.full(len(x),2)

def func_2b(x):
    return -x

def func_2c(x):
    return x**2

def func_2d(rands): # "rands" contains array of n-dimensional random numbers
    x = rands[:,0]
    y = rands[:,1]
    return ((x*y) + x)

func_2d_a_lims = np.array([0,0])
func_2d_b_lims = np.array([1,1])
out_2a = np.zeros((len(sample_size),2))
out_2b = np.zeros((len(sample_size),2))
out_2c = np.zeros((len(sample_size),2))
out_2d = np.zeros((len(sample_size),2))

for i in range(len(sample_size)):
    out_2a[i,:] = mc.multivariate_monte_carlo(func_2a, 0, 1, sample_size[i])
    out_2b[i,:] = mc.multivariate_monte_carlo(func_2b, 0, 1, sample_size[i])
    out_2c[i,:] = mc.multivariate_monte_carlo(func_2c, -2, 2, sample_size[i])
    out_2d[i,:] = mc.multivariate_monte_carlo(func_2d, func_2d_a_lims, \
                                      func_2d_b_lims, sample_size[i])



""" -------------------------------------------------------------"""
""" ----------------------- TASK 3 ------------------------------"""
""" -------------------------------------------------------------"""

func_3d_a_lims = np.full(3,-2)
func_3d_b_lims = np.full(3,2)
func_5d_a_lims = np.full(5,-2)
func_5d_b_lims = np.full(5,2)

def func_3d(rands):
    x = rands[:,0]
    y = rands[:,1]
    z = rands[:,2]
    r_sqr = (x**2)+(y**2)+(z**2)
    for i in range(len(r_sqr)):
        if r_sqr[i] > 4:
            r_sqr[i] = 0
        else:
            r_sqr[i] = 1
    return r_sqr

def func_5d(rands):
    x = rands[:,0]
    y = rands[:,1]
    z = rands[:,2]
    j = rands[:,3]
    k = rands[:,4]
    r_sqr = (x**2)+(y**2)+(z**2)+(j**2)+(k**2)
    for i in range(len(r_sqr)):
        if r_sqr[i] > 4:
            r_sqr[i] = 0
        else:
            r_sqr[i] = 1
    return r_sqr

out_3a = mc.multivariate_monte_carlo(\
            func_3d, func_3d_a_lims, func_3d_b_lims, sample_size[0])
out_3b = mc.multivariate_monte_carlo(\
            func_5d, func_5d_a_lims, func_5d_b_lims, sample_size[0])    
    
    
    
""" -------------------------------------------------------------"""
""" ----------------------- TASK 4 ------------------------------"""
""" -------------------------------------------------------------"""

task_4_a_lims = np.full(9,0)
task_4_b_lims = np.full(9,1)

def func_tk4(rands):
    ax = rands[:,0]
    ay = rands[:,1]
    az = rands[:,2]
    bx = rands[:,3]
    by = rands[:,4]
    bz = rands[:,5]
    cx = rands[:,6]
    cy = rands[:,7]
    cz = rands[:,8]
    dot = np.zeros(len(ax))
    for i in range(len(ax)):
        dot[i] = np.dot(([ax[i]+bx[i],ay[i]+by[i],az[i]+bz[i]]),\
                        [cx[i],cy[i],cz[i]])
    result = 1 / np.abs(dot)
    return result
    
out_tk4 = mc.multivariate_monte_carlo(\
            func_tk4, task_4_a_lims, task_4_b_lims, sample_size[0])  
    
    
    
""" -------------------------------------------------------------"""
""" ----------------------- TASK 5 ------------------------------"""
""" -------------------------------------------------------------"""

nsamp_a = 100000 # Sample Size for the Metropolis and Importance Sampling
                 # 5a
nsamp_b = 100000 # Sample Size for the Metropolis and Importance Sampling
                 # 5b         


''' 5a '''

Aa_lim = -10
Ab_lim = 10

def func_tk5a(x):
    return 2*np.exp((-(x**2)))

def func_tk5a_wp(x):
    return np.exp(-(np.abs(x)))

# Auto_Normed Version
start = time.time()
out_tk5a_auto, wp_5a = mc.univar_mc_importance(func_tk5a, \
            func_tk5a_wp, 0, nsamp_a)
end = time.time() 
elapsed_time_5a_metropolis_auto = end - start

# Manually_Normed Version
normalisation_5a = 2*(1-np.exp(-10))
start = time.time()
out_tk5a_manual, wp_5a = mc.univar_mc_importance(func_tk5a, \
            func_tk5a_wp, 0, nsamp_a, normalisation_5a, auto_norm=False)
end = time.time() 
elapsed_time_5a_metropolis_manual = end - start   


''' 5b '''

Ba_lim = 0
Bb_lim = np.pi

def func_tk5b(x):
    return 1.5*np.sin(x)

def func_tk5b_wp(x):
    return (4/((np.pi)**2))*x*((np.pi)-x)

# Auto_Normed Version
start = time.time()
out_tk5b_auto, wp_5b = mc.univar_mc_importance(func_tk5b, \
            func_tk5b_wp, 1.5, nsamp_b, a=Ba_lim, b=Bb_lim)
end = time.time()     
elapsed_time_5b_metropolis_auto = end - start
    
# Manually_Normed Version
normalisation_5b = (2*np.pi)/3
start = time.time()
out_tk5b_manual, wp_5b = mc.univar_mc_importance(func_tk5b, \
            func_tk5b_wp, 1.5, nsamp_b, normalisation_5b, auto_norm=False)
end = time.time() 
elapsed_time_5b_metropolis_manual = end - start   

    
""" -------------------------------------------------------------"""
""" ----------------------- TASK 6 ------------------------------"""
""" -------------------------------------------------------------"""


nsamp_a = 30000 # Sample Size Used to Match Accuracy of Task 5a
nsamp_b = 4800000 # Sample Size Used to Match Accuracy of Task 5b

''' 5a via Uniform Distribution'''

start = time.time()
out_a_uniform = mc.multivariate_monte_carlo(\
                func_tk5a, Aa_lim, Ab_lim, nsamp_a)
end = time.time()
elapsed_time_5a_uniform = end - start

''' 5b via Uniform Distribution'''

start = time.time()
out_b_uniform = mc.multivariate_monte_carlo(\
                func_tk5b, 0, np.pi, nsamp_b)
end = time.time()
elapsed_time_5b_uniform = end - start



""" -------------------------------------------------------------"""
""" ----------------------- Output ------------------------------"""
""" -------------------------------------------------------------"""


# Some Extra Plot Data that Helped with Debugging   

''' 5a Histogram '''

xa_range = np.arange(Aa_lim,Ab_lim,0.01)
ya1 = func_tk5a_wp(xa_range)
ya2 = func_tk5a(xa_range) 
    
plt.plot(xa_range,ya1, label='w(x)')
plt.plot(xa_range,ya2, label='f(x)')
plt.hist(wp_5a, bins=75, density="True")
plt.xlim(Aa_lim,Ab_lim)
plt.legend()
plt.show()

''' 5b Histogram '''

xb_range = np.arange(Ba_lim,Bb_lim,0.01)
yb1 = func_tk5b_wp(xb_range)
yb2 = func_tk5b(xb_range) 
    
plt.plot(xb_range,yb1, label='w(x)')
plt.plot(xb_range,yb2, label='f(x)')
plt.hist(wp_5b, bins=75, density="True")
plt.xlim(Ba_lim,Bb_lim)
plt.legend()
plt.show()


# Results

print("\n----------------------------------")

for i in range(len(sample_size)):
    print("\n----Sample Size of",sample_size[i]," ----")
    print("\nTask 2a Result = ",out_2a[i,0], ", STD = ",out_2a[i,1])
    print("\nTask 2b Result = ",out_2b[i,0], ", STD = ",out_2b[i,1])
    print("\nTask 2c Result = ",out_2c[i,0], ", STD = ",out_2c[i,1])
    print("\nTask 2d Result = ",out_2d[i,0], ", STD = ",out_2d[i,1])

print("\n----------------------------------")

print("\nTask 3a Result (3-Sphere) = ",out_3a[0],\
      ", STD = ",out_3a[1])
print("\nTask 3b Result (5-Sphere) = ",out_3b[0],\
      ", STD = ",out_3b[1])

print("\n----------------------------------")

print("\nTask 4 Result = ",out_tk4[0],", STD = ",out_tk4[1])

print("\n----------------------------------")

print("\nTask 5a Results")
print("\nAuto-Normalised = ",out_tk5a_auto[0],", STD = ",\
      out_tk5a_auto[1])
print("\nManually-Normalised = ",out_tk5a_manual[0],", STD = ",\
      out_tk5a_manual[1])

print("\n----------------------------------")

print("\nTask 5b Results")
print("\nAuto-Normalised = ", out_tk5b_auto[0],\
      ", STD = ",out_tk5b_auto[1])
print("\nManually-Normalised = ", out_tk5b_manual[0],\
      ", STD = ", out_tk5b_manual[1])

print("\n----------------------------------")

print("\nTask 6a Result = ",out_a_uniform[0],\
      ", STD = ",out_a_uniform[1])
print("\nTask 6b Result = ",out_b_uniform[0],\
      ", STD = ",out_b_uniform[1])

print("\n----------------------------------")

print("\nWall (Calculation) Times for Task 5 vs 6, in Seconds")
print("\n5a Metropolis Auto-Normed = ",\
      elapsed_time_5a_metropolis_auto)
print("\n5a Metropolis Manually-Normed = ",\
      elapsed_time_5a_metropolis_manual)
print("\n5a Standard Uniform Method = ",\
      elapsed_time_5a_uniform)
print("\n5b Metropolis Auto-Normed = ",\
      elapsed_time_5b_metropolis_auto)
print("\n5b Metropolis Manually-Normed = ",\
      elapsed_time_5b_metropolis_manual)
print("\n5b Standard Uniform Method = ",\
      elapsed_time_5b_uniform)

print("\n----------------------------------")

