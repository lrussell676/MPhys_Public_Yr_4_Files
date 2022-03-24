# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:16:15 2022

@author: Lewis
"""

import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200



''' --------------------------------------------------------------------- '''
''' ---------------------- Seed Generation and RNGs --------------------- '''
''' --------------------------------------------------------------------- '''

''' Here, the seed and pseudo-random-number-generator is specified.
    This is defined and included to allow for reproducibility in
    the stochastic varibles used throughout the code. '''

from numpy.random import Generator, MT19937, SeedSequence
seed_1 = 94747
ss1 = SeedSequence(seed_1)
MT_bit_gen = MT19937(ss1)
rng_MT = Generator(MT_bit_gen)



''' --------------------------------------------------------------------- '''
''' --------------------- Geometric Brownian Motion --------------------- '''
''' --------------------------------------------------------------------- '''

#################################################################
########## Comment Time Duration for 'Days' or 'Years' ##########
#
timestep = 'Days'
time_on_graphs = 'Years'
#
if timestep != 'Years' and timestep != 'Days':
    raise ValueError("timestep must be 'Days' or 'Years'")
if time_on_graphs != 'Years' and time_on_graphs != 'Days':
    raise ValueError("time_on_graphs must be 'Days' or 'Years'")
#################################################################

########## Main GBM Function ##############

def SGBM(seed,sigma,t_end,dt,x0):
    ##
    ########## Random Numbers #################
    ##
    ss1 = SeedSequence(seed)
    MT_bit_gen = MT19937(ss1)
    rng_MT = Generator(MT_bit_gen)
    RNGs = rng_MT.normal(0, np.sqrt(dt), size=(t_end,len(sigma)))
    ##
    ########## Standard GBM Equation ##########
    ##
    x = np.zeros((t_end,len(sigma)))
    for i in range(len(sigma)):
        x[0,i] = x0
        for j in range(1,(t_end)):
            x[j,:] = x[j-1,:] * np.exp(
                (mu - ((sigma**2)/2)) * dt
                + sigma * RNGs[j,:] )  
        if x[j,i] <= 0:
            x[j,i] = x[j,i] - x[j,i]
    ##
    ########## Setting Time Axis ##############
    ##
    t = np.zeros(len(x))
    t[0] = 0
    for i in range(1,len(t)):
        if timestep == 'Days' and time_on_graphs == 'Days':
            t[i] = i
        if timestep == 'Days' and time_on_graphs == 'Years':
            t[i] = i/365.0
    ##
    return x, t

########## Drift Conditions ###############

mu_percent = 1.03                              # Percentage Drift
if timestep == 'Years':
    mu = np.log(mu_percent)                    # mu%/year, t=year
if timestep == 'Days':
    mu = np.log(np.power(mu_percent,(1/365)))  # mu%/year, t=day

########## Set-up Conditions ##############

sigmas = np.array([0.0,0.01,0.05,0.1])         # Volatility
t_end = 20*365                                 # Time Duration of Simulation
dt = 1                                         # Time-Increment (timestep*dt)
x0 = 100                                       # Initial Stock Price

########## Data Generation ################
    
x = np.zeros((t_end,len(sigmas)))
x, t = SGBM(seed_1,sigmas,t_end,dt,x0)

seeds = rng_MT.uniform(0, 99999, size=10)
seeds = seeds.astype(int)

sigma = np.array([0.01, 0.1])    
x_big = np.zeros((len(seeds),t_end,1))
x_big2 = np.zeros((len(seeds),t_end,1))
for i in range(len(seeds)):
    x_big[i,:,:], t = SGBM(seeds[i],np.array([sigma[0]]),t_end,dt,x0)
    x_big2[i,:,:], t = SGBM(seeds[i],np.array([sigma[1]]),t_end,dt,x0)      

########## Plotting #######################

def SGBM_plot1(x,t,sigma,y_tick_toggle, multi_seed_toggle, *seeds):
    if multi_seed_toggle == 1:
        for i in range(len(x[:,0,0])):
            plt.plot(t,x[i,:,:], lw='0.5')
        plt.title("Realisations of Geometric Brownian Motion \n with"
            " various Seeds, $\mu=${:.3}, $\sigma$ = {:}".format(mu,sigma[0]))
    else:
        for i in range(len(x[0,:])):
            plt.plot(t,x[:,i], label=f'$\sigma$ = {sigma[i]}')
        plt.title("Realisations of Geometric Brownian Motion \n with"
                  " various Percentage Volatilities, $\mu=${:.3}".format(mu))
    plt.xlabel(r"Time in {:}, $t$".format(time_on_graphs))
    plt.ylabel("Stock Valuation, $x$")
    plt.axhline(x0, label='Initial Stock Value', \
                color='red', ls='--', lw='0.5')
    if y_tick_toggle == 1:
        y_max = np.round(np.max(x),\
                         -int(math.floor(math.log10(abs(np.max(x))))))
        plt.yticks((np.arange(0, y_max-99, step=y_max/5)) + x0)
    plt.legend()
    plt.grid(b=True, which='major', ls='-', lw='0.2')
    plt.minorticks_on()
    plt.show()
    
def SGBM_plot2():
    Figure, (SubPlot1a,SubPlot1b) = plt.subplots(1,2, constrained_layout=True,
                                             sharex=True)
    Figure.suptitle("Realisations of Geometric Brownian Motion \n with"
                " various Seeds, $\mu=${:.3}".format(mu))
    SubPlot1a.grid(b=True, which='major', linestyle=':', linewidth='2')
    SubPlot1b.grid(b=True, which='major', linestyle=':', linewidth='2')
    for i in range(len(x_big[:,0,0])):
                SubPlot1a.plot(t,x_big[i,:,:], lw='0.5')
                SubPlot1b.plot(t,x_big2[i,:,:], lw='0.5')
    SubPlot1a.set_xlabel(r"Time in {:}, $t$".format(time_on_graphs))
    SubPlot1a.set_ylabel("Stock Valuation, $x$")
    SubPlot1b.set_xlabel(r"Time in {:}, $t$".format(time_on_graphs))
    #SubPlot1b.set_ylabel("Stock Valuation, $x$")
    SubPlot1a.legend([r"$\sigma$ = {:}".format(sigma[0])],\
                     handlelength=0, handletextpad=0,)
    SubPlot1b.legend([r"$\sigma$ = {:}".format(sigma[1])],\
                     handlelength=0, handletextpad=0,)
    SubPlot1a.grid(b=True, which='major', ls='-', lw='0.2')
    SubPlot1b.grid(b=True, which='major', ls='-', lw='0.2')
    plt.minorticks_on()
    plt.show()

#SGBM_plot1(x_big,t,sigma[0],0,1,*seeds)
#SGBM_plot1(x_big2,t,sigma[1],0,1,*seeds)

SGBM_plot1(x,t,sigmas,1, 0)  # Figure 1 in the Essay

SGBM_plot2()                 # Figure 2 in the Essay



''' ---------------------------------------------------------------------- '''
''' --------------- Logistics Map of Capital Gains Seekers --------------- '''
''' ---------------------------------------------------------------------- '''

t_range = 999

# "all_xs" is not needed for the following bifurcation plot, though a nice 
# addition in order to save the array of inspection through a variable
# explorer after running the script
all_xs = np.zeros(((t_range+1),1000))
all_xs[0,:] = 0.5

x = 0.5 + np.zeros(t_range) # holds a temporary x array
a_s = np.linspace(0,4,1000) # contains all bifurcation coefficients 

for a_i in range(len(a_s)):
    for i in range(t_range-1):
        x[i+1] = a_s[a_i] * x[i] * (1-x[i])    # The Logistics Equation
    all_xs[1:,a_i] = x                         # Saves temp x array to all_xs
    # "final_xs" takes all unique values of the last 10% of x values
    # This will be only be of length one in stable ranges of a_s,  &
    # more in cases of periodic or chaotic regions. 
    final_xs = np.unique(x[round(t_range*0.9):])
    # "final_as" creates an array equal in length to len(final_xs), & is
    # equal in value to the a_s[a_i] of that run. Although only the value of
    # the bifurcation coefficient 'a_s[a_i]' is of interest, it has to match
    # the array size of "final_xs" in order to plot correctly.
    final_as = a_s[a_i]*np.ones(len(final_xs))
    plt.plot(final_as, final_xs, '.', color='black', ms='0.5')
    
plt.title("Basic Logistics Map")
plt.xlabel("Bifurcation Parameter, $'a'$")
plt.ylabel("Unique Occurances of $'x'$")
plt.grid(b=True, which='major', ls='-', lw='0.2')
plt.minorticks_on()
plt.show()                   # Figure 3 in the Essay



''' ---------------------------------------------------------------------- '''
''' -------- Merging the GBM Model with Logistic Mapped Volatility ------- '''
''' ---------------------------------------------------------------------- '''

def SGBM_logistic(sigma_cof, a):
    # "sigma_cof" = additional volatility weighting
    # "a" = bifurcation parameter
    #
    l = 0.5 + np.zeros(t_end)           # holds logistic map over time
    x_l = np.zeros((t_end,len(seeds)))  # holds stock price over time
    ##
    for i in range(t_end-1):                   # loops for logistic map
        l[i+1] = a * l[i] * (1-l[i])
    l = sigma_cof*l
    ##
    for i in range(len(seeds)):
        ##
        ########## Random Numbers #################
        ##
        ss1 = SeedSequence(seeds[i])
        MT_bit_gen = MT19937(ss1)
        rng_MT = Generator(MT_bit_gen)
        RNGs = rng_MT.normal(0, np.sqrt(dt), size=(t_end,len(seeds)))
        ##
        ########## Modified GBM Equation ##########
        ##
        x_l[0,i] = x0             
        for j in range(1,(t_end)):             # loops for GBM
            x_l[j,:] = x_l[j-1,:] * np.exp(
                (mu - (( l[j]**2)/2)) * dt
                + l[j] * RNGs[j,:] )  
        if x_l[j,i] <= 0:
            x_l[j,i] = x_l[j,i] - x_l[j,i]
        ##
        ########## Setting Time Axis ##############
        ## 
        t_l = np.zeros(len(x_l))
        for i in range(1,len(t_l)):
            if timestep == 'Days' and time_on_graphs == 'Days':
                t_l[i] = i
            if timestep == 'Days' and time_on_graphs == 'Years':
                t_l[i] = i/365.0
        ##
    return x_l, t_l, l

x_l, t_l, l = SGBM_logistic(0.01,3.81)


for i in range(len(seeds)):
    plt.plot(t_l, x_l[:,i], lw='0.5')
plt.title("Realisations of Geometric Brownian Motion \n Adapted "
          "to Include Logistic Volatilities")
plt.xlabel(r"Time in {:}, $t$".format(time_on_graphs))
plt.ylabel("Stock Valuation, $x$")
axhline = plt.axhline(x0, label='Initial Stock Value', \
               color='red', ls='--', lw='0.5')
plt.grid(b=True, which='major', ls='-', lw='0.2')
plt.legend()
plt.minorticks_on()
plt.show()                   # Figure 4 in the Essay

per_rtn = np.zeros((19,len(seeds)))
for i in range(len(x_l[0,:])):
    y, j = 0, 0
    while y < t_end-365:
        per_rtn[j,i] = 100 * ( (x_l[y+365,i]/x_l[y,i]) - 1)
        y += 365
        j += 1

compar_rtns = np.zeros((5,len(seeds)))
for i in range(5):
    compar_rtns[i,:] = per_rtn[i+14,:]
    
real_stock_rtns = np.array(( [ 12.36, -8.05, 14.44, 5.56, 13.38] ,
                             [ 8.02, -10.59, 5.79, 12.13, -5.62] ,
                             [ -4.9, 9.7, -6.4, 19.4, 32.7] ,
                             [ 9.65, -8.2, 19.12, 4.25, 25.71] ,
                             [ 28.29, -10.04, 14.04, 15.16, 2.64] ,
                             [ 24.65, 1.67, 41.02, 40.86, 34.97] ,
                             [ 13.35, -9.09, 16.93, 6.24, 16.58] ,
                             [ 7.04, -3.96, 11.37, 5.00, 3.03] ,
                             [ 3.21, -2.29, 8.51, 7.62, -3.57] ,
                             [ 12.53, -12.82, 24.10, -5.33, 18.00] )).T

# Each line of "real_stock_rtns" is sequentially equal to the listing of
# Fund Factsheets in the Appendix of the Essay.
# The ".T" transpose of the matrix was to match the dimensions of the
# "compar_rtns" array from Figure 4. It is easier to visually read in the
# 10x5 form above than the required 5x10 form. 

## --- Correlation Below --- #

# Arrays Flatten into 1D, which is required for the i,j nature of 
# autocorrelation
real_stock_rtns = real_stock_rtns.flatten()
compar_rtns = compar_rtns.flatten()

a_rtns = real_stock_rtns
b_rtns = compar_rtns
# using some simple means and standard deviations to normalise the
# output of 'np.correlate'
a_cor = (a_rtns - np.mean(a_rtns)) / (np.std(a_rtns) * len(a_rtns))     
b_cor = (b_rtns - np.mean(b_rtns)) / (np.std(b_rtns))

general_correlation = np.correlate(a_cor,b_cor, mode='full')

plt.plot(general_correlation)
plt.title("Full Autocorrelation Between the Developed Model \n"
          "and Real Performance of 10 Funds from 2017/18-21/22")
plt.xlabel("x-Dimension Shift Index of Autocorrelation Result")
plt.ylabel("Normalised Correlation Factor")
plt.grid(b=True, which='major', ls='-', lw='0.2')
plt.minorticks_on()
plt.show()                   # Figure 5 in the Essay

    
    
