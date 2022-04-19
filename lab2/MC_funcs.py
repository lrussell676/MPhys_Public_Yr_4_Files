# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:56:58 2022

@author: Lewis
"""

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from inspect import currentframe

"""########################################################################"""
"""############################## RNG Set-Up ##############################"""

# Building off of lab1, here the MT19937 pseudo-RNG is used

seed_1 = 5648 # Specifies Seed used throughout code
ss1 = SeedSequence(seed_1)
MT_bit_gen = MT19937(ss1)
rng_MT = Generator(MT_bit_gen)


"""########################################################################"""
"""########################### Metropolis Steps ###########################"""
"""#######                                                          #######"""

# As per pseudo-code from lecture notes, so no comments given here

def Mstep(x_old, wp, delta):
    delta_i = rng_MT.uniform(-delta,delta)
    x_trial = x_old + delta_i
    w = wp(x_trial)/wp(x_old)
    if w >= 1:
        return x_trial
    else:
        r = rng_MT.uniform(0,1)
        if r <= w:
            return x_trial
        else:
            return x_old
        

"""########################################################################"""
"""####################### Multivariate Monte Carlo #######################"""
"""#######                                                          #######"""

def multivariate_monte_carlo(func, a, b, n):
    # "func" is the inputted integrand
    # "a/b" the lower/upper integration limits
    # "n" the number of points to sample
    
    '''##### Data Structure Handing and Error Raising Before MC Code #####'''
    
    if type(a) == int:        
        a = np.array([a])   
        b = np.array([b])
    # Above: Converts int to array, for univariate cases. This makes the 
    # use of the function more user-friendly as it can accept ints directly,
    # or as a numpy array, whilst still maintaining correct functionality with
    # the rest of the multivariate_monte_carlo function script.
    
    dim = len(a) # "dim" introduces a specification for the dimensionality 
                 # of the integrad.
                 
    if len(a) != len(b):
        raise ValueError("Lower and Upper Limits Must be of the Same Size.")
    
    '''########################### Main MC Code ###########################'''
    
    RNGed_vals = np.zeros((n,dim))
    
    for i in range(dim): # generates all random numbers required
        RNGed_vals[:,i] = rng_MT.uniform(a[i], b[i], n)
    
    y = func(RNGed_vals) # applies random numbers as inputs to integrand 
    
    y_mean_avg =  (y.sum())/len(y)  # mean average 
    
    domain = 0              # the domain takes each a/b limit and multiplies                
    for i in range(dim):    # to create the correct weighting on y_mean_avg
        if domain == 0:
            domain = (b[i]-a[i])
        else:
            domain *= (b[i]-a[i])
    
    # Final Integral Result:
    integral = domain * y_mean_avg
    # Variance Calculation:
    var = (1/n)*( (np.sum(y**2)/n) - (y_mean_avg**2) )
    # Using Standard Deviation as Final Error Estimate in Integral
    std = np.sqrt(var)
    # Bundles Integral and its Standard Deviation into a Single Output
    results = np.array([integral,std])
    
    return results

    
"""########################################################################"""
"""######################## Importance Sampling MC ########################"""
"""#######                                                          #######"""

def univar_mc_importance(func, wp, x0, n, \
                         weight=None, a=None, b=None, auto_norm=True):
    # ----------------------------------------------
    # ------------ Compulsory Arguments ------------
    # ----------------------------------------------
    # "func" is the inputted integrand
    # "wp" the weighted probability function
    # "x0" the initial guess for Metropolis
    # "n" the number of points to sample
    # ----------------------------------------------
    # ------------- Optional Arguments -------------
    # ----------------------------------------------
    # "weight" is the normalisation factor, if manually specified
    # alongside setting "auto_norm=False".
    # "a/b" are the integral limits, if manually specified
    # alongside auto_normalisation mode being set to
    # "auto_norm=False".
    if auto_norm==False and weight==None:
        raise ValueError("Normalisation Factor 'weight' not passed, "
                         "despite setting auto_normalisation mode "
                         "'auto_norm' to 'False'.")
        
    wp_dis = np.zeros(n)
    wp_dis[0] = x0
    for i in range(n-1):
        wp_dis[i+1] = Mstep(wp_dis[i], wp, 2)
    
    if auto_norm:
        integral_top = func(wp_dis)/(wp(wp_dis))
        if (a==None or b==None):
            print("NOTE: ( line",currentframe().f_back.f_lineno,")")
            print("Integral limits 'a' and/or 'b' not passed, "
                         "despite auto_normalisation mode being "
                         "set to 'True'.\nCalculating the integral "
                         "between the limits of +-inf.")
            print("")
            integral_bottom = (wp(wp_dis))
            integral = (np.sum(integral_top) / np.sum(integral_bottom))
        else:
            integral_bottom = (1/wp(wp_dis))
            integral = (b-a)*(np.sum(integral_top) / np.sum(integral_bottom))
        std = 'Not Calculated for auto_norm=True.'
    else:
        y = func(wp_dis) / (wp(wp_dis))
        y_mean_avg = np.sum(y)/n
        #weight = float('.'.join(str(elem) for elem in weight))
        # The line above sorts out a data formatting issue that
        # was occuring, outputting the correct float as a result.
        integral = weight * y_mean_avg
        var = (1/n)*( (np.sum(y**2)/n) - (y_mean_avg**2) )
        # Using Standard Deviation as Final Error Estimate in Integral
        std = np.sqrt(var)
        
    # Bundles Integral and its Standard Deviation into a Single Output
    results = np.array([integral,std])
    
    return results, wp_dis # The Metropolis Array is also given in "wp_dis".
                           # This can allow for plots of the histogram info
                           # within the generated values.