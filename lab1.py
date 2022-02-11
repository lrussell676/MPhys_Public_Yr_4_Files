# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:17:55 2022

@author: Lewis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
from numpy.random import Generator, MT19937, PCG64, SeedSequence

val_size = np.array([100,100000])

seed_1 = 5647
seed_2 = 4563

time = 5000
particles = 500


""" -------------------------------------------------------------"""
""" ----------------------- TASK 1 ------------------------------"""
""" -------------------------------------------------------------"""

vals100 = np.zeros((4,val_size[0]))
vals100000 = np.zeros((4,val_size[1]))


""" Filliing Vals """

ss1 = SeedSequence(seed_1)
ss2 = SeedSequence(seed_2)
MT_bit_gen_ss1 = MT19937(ss1)
PC_bit_gen_ss1 = PCG64(ss1)
MT_bit_gen_ss2 = MT19937(ss2)
PC_bit_gen_ss2 = PCG64(ss2)

for i in range(0,val_size[0]):
    rng_MT_ss1 = Generator(MT_bit_gen_ss1)
    vals100[0,i] = rng_MT_ss1.uniform(0,1,1)
    rng_PC_ss1 = Generator(PC_bit_gen_ss1)
    vals100[1,i] = rng_PC_ss1.uniform(0,1,1)
    rng_MT_ss2 = Generator(MT_bit_gen_ss2)
    vals100[2,i] = rng_MT_ss2.uniform(0,1,1)
    rng_PC_ss2 = Generator(PC_bit_gen_ss2)
    vals100[3,i] = rng_PC_ss2.uniform(0,1,1)
    
for i in range(0,val_size[1]):
    rng_MT_ss1 = Generator(MT_bit_gen_ss1)
    vals100000[0,i] = rng_MT_ss1.uniform(0,1,1)
    rng_PC_ss1 = Generator(PC_bit_gen_ss1)
    vals100000[1,i] = rng_PC_ss1.uniform(0,1,1)
    rng_MT_ss2 = Generator(MT_bit_gen_ss2)
    vals100000[2,i] = rng_MT_ss2.uniform(0,1,1)
    rng_PC_ss2 = Generator(PC_bit_gen_ss2)
    vals100000[3,i] = rng_PC_ss2.uniform(0,1,1)


""" Plotting Histograms """

Figure, ((SubPlot1a,SubPlot1b),
         (SubPlot2a,SubPlot2b),
         (SubPlot3a,SubPlot3b),
         (SubPlot4a,SubPlot4b)) = plt.subplots(4,2, constrained_layout=False, sharex='col', sharey='row') 
Figure.suptitle("Uniform Distributions with Various Sample Sizes (\"S\")")
Figure.patch.set_facecolor('xkcd:mint green')
SubPlot1a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot1b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot2a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot2b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot3a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot3b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot4a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot4b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
Figure.add_subplot(111, frame_on = False)
plt.tick_params(labelcolor = "none", bottom = False, left = False)
plt.xlabel("x_Values (Randomised)")
plt.ylabel("Occurances of x_Values, Normalised to PDFs")
plt.tick_params(axis='both', which='major', pad=20)
SubPlot4a.set_xlabel("MT19937")
SubPlot4b.set_xlabel("PCG64")

SubPlot1a.hist(vals100[0,:],bins=100,range=(0,1),density=True) 
SubPlot2a.hist(vals100[2,:],bins=100,range=(0,1),density=True) 
SubPlot3a.hist(vals100000[0,:],bins=100,range=(0,1),density=True) 
SubPlot4a.hist(vals100000[2,:],bins=100,range=(0,1),density=True) 

SubPlot1b.hist(vals100[1,:],bins=100,range=(0,1),density=True)
SubPlot2b.hist(vals100[3,:],bins=100,range=(0,1),density=True)
SubPlot3b.hist(vals100000[1,:],bins=100,range=(0,1),density=True)
SubPlot4b.hist(vals100000[3,:],bins=100,range=(0,1),density=True)

# Adding ylabels
SubPlot2b.set_ylabel(r'   S=100', loc="top", fontsize = '8')
SubPlot2b.yaxis.set_label_position("right")
SubPlot4b.set_ylabel(r' S=100000', loc="top", fontsize = '8')
SubPlot4b.yaxis.set_label_position("right")

# Adding Legends
SubPlot1a.legend(['S={:}'.format(seed_1)], loc = 'upper right', fontsize = '6')
SubPlot2a.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')
SubPlot3a.legend(['S={:}'.format(seed_1)], loc = 'lower right', fontsize = '6')
SubPlot4a.legend(['S={:}'.format(seed_2)], loc = 'lower right', fontsize = '6')
SubPlot1b.legend(['S={:}'.format(seed_1)], loc = 'upper right', fontsize = '6')
SubPlot2b.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')
SubPlot3b.legend(['S={:}'.format(seed_1)], loc = 'lower right', fontsize = '6')
SubPlot4b.legend(['S={:}'.format(seed_2)], loc = 'lower right', fontsize = '6')

# Now we can set both plots to the same y scaling
SubPlot1a.set_ylim(0,SubPlot1b.get_ylim()[1])
SubPlot2a.set_ylim(0,SubPlot2b.get_ylim()[1])
SubPlot3a.set_ylim(0,SubPlot3b.get_ylim()[1])
SubPlot4a.set_ylim(0,SubPlot4b.get_ylim()[1])
plt.show()


""" Correlation Testing """

MT_Seed_1 = np.correlate(vals100000[0,:], vals100000[0,:], mode='full')
x_range = np.linspace(0,len(MT_Seed_1)/100,len(MT_Seed_1))

MT_Seed_1_normed1 = np.zeros(len(MT_Seed_1))
MT_Seed_1_normed2 = np.zeros(len(MT_Seed_1))
a = vals100000[0,:]
b = vals100000[0,:]
MT_Seed_1_normed1 = (a - np.mean(a)) / (np.std(a) * len(a))
MT_Seed_1_normed2 = (b - np.mean(b)) / (np.std(b))
MT_Seed_1_normed = np.correlate(MT_Seed_1_normed1,
                                MT_Seed_1_normed2, mode='full')

Figure, (SubPlot1a,SubPlot1b) = plt.subplots(1,2, constrained_layout=True)
Figure.patch.set_facecolor('xkcd:mint green')
Figure.suptitle("MT19937 Data from 100000 Sample Size, Seed={:}".format(seed_1),
          fontsize = '12')
SubPlot1a.grid(b=True, which='major', linestyle=':', linewidth='2')
SubPlot1b.grid(b=True, which='major', linestyle=':', linewidth='2')
SubPlot1a.scatter(x_range,MT_Seed_1,s=1)
SubPlot1a.set_xlabel("x-shift (x10^3)")
SubPlot1a.set_ylabel("Correlation Output")
SubPlot1b.scatter(x_range,MT_Seed_1_normed,s=1)
SubPlot1b.set_xlabel("x-shift (x10^3)")
SubPlot1b.set_ylabel("Correlation Output, Normalised")
SubPlot1b.set_ylim(-1,1)
plt.show()


""" Normed for all 4 types """

MT_Seed_2 = np.correlate(vals100000[2,:], vals100000[2,:], mode='full')
PC_Seed_1 = np.correlate(vals100000[1,:], vals100000[1,:], mode='full')
PC_Seed_2 = np.correlate(vals100000[3,:], vals100000[3,:], mode='full')

a = vals100000[2,:]
b = vals100000[2,:]
MT_Seed_2_normed1 = (a - np.mean(a)) / (np.std(a) * len(a))
MT_Seed_2_normed2 = (b - np.mean(b)) / (np.std(b))
MT_Seed_2_normed = np.correlate(MT_Seed_2_normed1,
                                MT_Seed_2_normed2, mode='full')
a = vals100000[1,:]
b = vals100000[1,:]
PC_Seed_1_normed1 = (a - np.mean(a)) / (np.std(a) * len(a))
PC_Seed_1_normed2 = (b - np.mean(b)) / (np.std(b))
PC_Seed_1_normed = np.correlate(PC_Seed_1_normed1,
                                PC_Seed_1_normed2, mode='full')
a = vals100000[3,:]
b = vals100000[3,:]
PC_Seed_2_normed1 = (a - np.mean(a)) / (np.std(a) * len(a))
PC_Seed_2_normed2 = (b - np.mean(b)) / (np.std(b))
PC_Seed_2_normed = np.correlate(PC_Seed_2_normed1,
                                PC_Seed_2_normed2, mode='full')


""" Plotting Correlation """

Figure, ((SubPlot1a,SubPlot1b),
         (SubPlot2a,SubPlot2b)) = plt.subplots(2,2, constrained_layout=False, sharex='col', sharey='row') 
Figure.suptitle("Correlation Outputs, Normalised")
Figure.patch.set_facecolor('xkcd:mint green')

SubPlot1a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot1b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot2a.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot2b.grid(b=True, which='major', linestyle='-', linewidth='0.5')
Figure.add_subplot(111, frame_on = False)
plt.tick_params(labelcolor = "none", bottom = False, left = False)

plt.xlabel("x-shift (x10^3)")
plt.ylabel("Correlation")
plt.tick_params(axis='both', which='major', pad=20)

SubPlot2a.set_xlabel("MT19937")
SubPlot2b.set_xlabel("PCG64")

# Adding Plots
SubPlot1a.scatter(x_range,MT_Seed_1_normed,s=0.001)
SubPlot1b.scatter(x_range,PC_Seed_1_normed,s=0.001)
SubPlot2a.scatter(x_range,MT_Seed_2_normed,s=0.001)
SubPlot2b.scatter(x_range,PC_Seed_2_normed,s=0.001)

# Adding Legends
SubPlot1a.legend(['S={:}'.format(seed_1)], loc = 'upper right', fontsize = '6')
SubPlot2a.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')
SubPlot1b.legend(['S={:}'.format(seed_1)], loc = 'upper right', fontsize = '6')
SubPlot2b.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')

# Now we can set all plots to the same y scaling
SubPlot1a.set_ylim(-0.02,0.02)
SubPlot2a.set_ylim(-0.02,0.02)
SubPlot3a.set_ylim(-0.02,0.02)
SubPlot4a.set_ylim(-0.02,0.02)
plt.show()



""" -------------------------------------------------------------"""
""" ----------------------- TASK 2 ------------------------------"""
""" -------------------------------------------------------------"""

box = np.zeros((time,2,particles))

def random_particle_selectors(a, b):
    if (a==1):
        rng_MT_ss1 = Generator(MT_bit_gen_ss1)
        ID = int(np.round(rng_MT_ss1.uniform(0,b,1)))
        return ID
    if (a==2):
        rng_PC_ss1 = Generator(PC_bit_gen_ss1)
        ID = int(np.round(rng_PC_ss1.uniform(0,b,1)))
        return ID
    if (a==3):
        rng_MT_ss2 = Generator(MT_bit_gen_ss2)
        ID = int(np.round(rng_MT_ss2.uniform(0,b,1)))
        return ID
    if (a==4):
        rng_PC_ss2 = Generator(PC_bit_gen_ss2)
        ID = int(np.round(rng_PC_ss2.uniform(0,b,1)))
        return ID
    else:
        raise ValueError("a must be 1,2,3, or 4. See function for info.")

for i in range(particles):
    box[0,0,i] = 1
    
for t in range(1,(time)):
    for i in range(2):
        for j in range(particles):
            box[t,i,j] = box[t-1,i,j]
    ID = random_particle_selectors(1,(particles-1))
    if (box[t,1,ID] == 1):
        box[t,0,ID] = 1
        box[t,1,ID] = 0
    elif (box[t,0,ID] == 1):
        box[t,0,ID] = 0
        box[t,1,ID] = 1
        
position_count = np.zeros((2,time))

for t in range(time):
    c = 0
    for i in range(particles):
        if (box[t,0,i] == 1):
            c += 1
    position_count[0,t] = c
    position_count[1,t] = particles - c
        
time_range = np.arange(0,time,1)    
    
Figure, (SubPlot1) = plt.subplots(1,1)
Figure.patch.set_facecolor('xkcd:mint green')
SubPlot1.grid(b=True, which='major', linestyle=':', linewidth='2')
SubPlot1.plot(time_range,position_count[0,:])
SubPlot1.plot(time_range,position_count[1,:])
SubPlot1.set_xlabel("Timestep")
SubPlot1.set_ylabel("Particle Count")
SubPlot1.legend(['Partition 1', 'Partition 2'])
plt.title("Particle Position in Time, MT19937, Seed={:}".format(seed_1),
          fontsize = '12')
plt.show()



""" -------------------------------------------------------------"""
""" ----------------------- TASK 3 ------------------------------"""
""" -------------------------------------------------------------"""

box_b = np.zeros((time,2,particles)) # MT Seed 2
box_c = np.zeros((time,2,particles)) # PC Seed 1
box_d = np.zeros((time,2,particles)) # PC Seed 2

for i in range(particles):
    box_b[0,0,i] = 1
    box_c[0,0,i] = 1
    box_d[0,0,i] = 1
    
for t in range(1,(time)):
    for i in range(2):
        for j in range(particles):
            box_b[t,i,j] = box_b[t-1,i,j]
    ID = random_particle_selectors(3,(particles-1))
    if (box_b[t,1,ID] == 1):
        box_b[t,0,ID] = 1
        box_b[t,1,ID] = 0
    elif (box_b[t,0,ID] == 1):
        box_b[t,0,ID] = 0
        box_b[t,1,ID] = 1
        
for t in range(1,(time)):
    for i in range(2):
        for j in range(particles):
            box_c[t,i,j] = box_c[t-1,i,j]
    ID = random_particle_selectors(2,(particles-1))
    if (box_c[t,1,ID] == 1):
        box_c[t,0,ID] = 1
        box_c[t,1,ID] = 0
    elif (box_c[t,0,ID] == 1):
        box_c[t,0,ID] = 0
        box_c[t,1,ID] = 1
        
for t in range(1,(time)):
    for i in range(2):
        for j in range(particles):
            box_d[t,i,j] = box_d[t-1,i,j]
    ID = random_particle_selectors(4,(particles-1))
    if (box_d[t,1,ID] == 1):
        box_d[t,0,ID] = 1
        box_d[t,1,ID] = 0
    elif (box_d[t,0,ID] == 1):
        box_d[t,0,ID] = 0
        box_d[t,1,ID] = 1
        
position_count_b = np.zeros((2,time))
position_count_c = np.zeros((2,time))
position_count_d = np.zeros((2,time))

for t in range(time):
    b = 0
    c = 0
    d = 0
    for i in range(particles):
        if (box_b[t,0,i] == 1):
            b += 1
        if (box_c[t,0,i] == 1):
            c += 1
        if (box_d[t,0,i] == 1):
            d += 1
    position_count_b[0,t] = b
    position_count_b[1,t] = particles - b
    position_count_c[0,t] = c
    position_count_c[1,t] = particles - c
    position_count_d[0,t] = d
    position_count_d[1,t] = particles - d
    
Figure, ((SubPlot1),(SubPlot2),(SubPlot3)) = \
    plt.subplots(3,1, constrained_layout=False, sharex='col', sharey='row') 
Figure.suptitle("Particle Positions in Time")
Figure.patch.set_facecolor('xkcd:mint green')

SubPlot1.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot2.grid(b=True, which='major', linestyle='-', linewidth='0.5')
SubPlot3.grid(b=True, which='major', linestyle='-', linewidth='0.5')
Figure.add_subplot(111, frame_on = False)
plt.tick_params(labelcolor = "none", bottom = False, left = False)
plt.tight_layout()

plt.xlabel("Timestep")
plt.ylabel("Particle Count")
plt.tick_params(axis='both', which='major', pad=20)

SubPlot1.set_xlabel("    MT19937",loc='left')
SubPlot2.set_xlabel("    PCG64",loc='left')
SubPlot3.set_xlabel("    PCG64",loc='left')

# Adding Plots
SubPlot1.plot(time_range,position_count_b[0,:])
SubPlot1.plot(time_range,position_count_b[1,:])
SubPlot2.plot(time_range,position_count_c[0,:])
SubPlot2.plot(time_range,position_count_c[1,:])
SubPlot3.plot(time_range,position_count_d[0,:])
SubPlot3.plot(time_range,position_count_d[1,:])

# Adding Legends
SubPlot1.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')
SubPlot2.legend(['S={:}'.format(seed_1)], loc = 'upper right', fontsize = '6')
SubPlot3.legend(['S={:}'.format(seed_2)], loc = 'upper right', fontsize = '6')

plt.show()



""" -------------------------------------------------------------"""
""" ----------------------- TASK 4 ------------------------------"""
""" -------------------------------------------------------------"""

prob = 0.75
box_2 = np.zeros((time,2,particles))

def random_float_generators(a, b):
    if (a==1):
        rng_MT_ss1 = Generator(MT_bit_gen_ss1)
        ID = rng_MT_ss1.uniform(0,b,1)
        return ID
    if (a==2):
        rng_PC_ss1 = Generator(PC_bit_gen_ss1)
        ID = rng_PC_ss1.uniform(0,b,1)
        return ID
    if (a==3):
        rng_MT_ss2 = Generator(MT_bit_gen_ss2)
        ID = rng_MT_ss2.uniform(0,b,1)
        return ID
    if (a==4):
        rng_PC_ss2 = Generator(PC_bit_gen_ss2)
        ID = rng_PC_ss2.uniform(0,b,1)
        return ID
    else:
        raise ValueError("a must be 1,2,3, or 4. See function for info.")


for i in range(particles):
     box_2[0,0,i] = 1

for t in range(1,(time)):
    for i in range(2):
        for j in range(particles):
            box_2[t,i,j] = box_2[t-1,i,j]
    chance_value = random_float_generators(1,1)  
    ID = random_particle_selectors(1,(particles-1))
    if (chance_value < prob) and (box_2[t,1,ID] == 1): 
        box_2[t,0,ID] = 1
        box_2[t,1,ID] = 0
    elif (chance_value >= prob) and (box_2[t,1,ID] == 0):
        box_2[t,0,ID] = 0
        box_2[t,1,ID] = 1
            
        
position_count_task4 = np.zeros((2,time))

for t in range(time):
    c = 0
    for i in range(particles):
        if (box_2[t,0,i] == 1):
            c += 1
    position_count_task4[0,t] = c
    position_count_task4[1,t] = particles - c
        
time_range = np.arange(0,time,1)    
    
Figure, (SubPlot1) = plt.subplots(1,1)
Figure.patch.set_facecolor('xkcd:mint green')
SubPlot1.grid(b=True, which='major', linestyle=':', linewidth='2')
plt.axhline(y=(particles*prob), color='r', linestyle='--', linewidth='0.8')
plt.axhline(y=(particles*(1-prob)), color='r', linestyle='--', linewidth='0.8')
SubPlot1.plot(time_range,position_count_task4[0,:])
SubPlot1.plot(time_range,position_count_task4[1,:])
SubPlot1.set_xlabel("Timestep")
SubPlot1.set_ylabel("Particle Count")
plt.title("Particle Position in Time",
          fontsize = '12')
plt.show()