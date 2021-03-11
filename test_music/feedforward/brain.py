#!/usr/bin/env python3

import nest
import numpy as np
import time
import sys
import music
#import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../../')

import trajectories as tj
from motorcortex import MotorCortex
from population_view import plotPopulation
from util import savePattern
from population_view import PopView
from pointMass import PointMass

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

# Randomize
# msd = int( time.time() * 1000.0 )
# N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
# nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})

m      = 5.0
ptMass = PointMass(mass=m)
njt    = ptMass.numVariables()

# Neuron neurons
N = 5

time_span = 1000.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

pthDat = "./data/"

init_pos = np.array([0.0,0.0])
tgt_pos  = np.array([10.0,10.0])
trj, pol = tj.minimumJerk(init_pos, tgt_pos, time_vect)

#tgt_real = np.array([15.0,0.0])
#trj_real, pol = tj.minimumJerk(init_pos, tgt_real, time_vect)
trj_real = trj # NO ERROR

# Error in joint trajectory
trj_err = trj-trj_real
savePattern(trj_err, pthDat+"error")


####### Error population (goes into motor cortex feedback)
err_param = {
    "base_rate": 0.0, # Feedforward neurons
    "kp": 1.0
    }

err_pop_p = []
err_pop_n = []
for i in range(njt):
    # Positive population (joint i)
    tmp_pop_p = nest.Create("tracking_neuron", n=N, params=err_param)
    nest.SetStatus(tmp_pop_p, {"pos": True, "pattern_file": pthDat+"error"+"_"+str(i)+".dat"})
    err_pop_p.append( PopView(tmp_pop_p,time_vect) )

    # Negative population (joint i)
    tmp_pop_n = nest.Create("tracking_neuron", n=N, params=err_param)
    nest.SetStatus(tmp_pop_n, {"pos": False, "pattern_file": pthDat+"error"+"_"+str(i)+".dat"})
    err_pop_n.append( PopView(tmp_pop_n,time_vect) )


####### Motor cortex
mc_param = {
    "ffwd_base_rate": 0.0, # Feedforward neurons
    "ffwd_kp": 10.0,
    "fbk_base_rate": 0.0,  # Feedback neurons
    "fbk_kp": 10.0,
    "out_base_rate": 0.0,  # Summation neurons
    "out_kp":1.0,
    "wgt_ffwd_out": 1.0,   # Connection weight from ffwd to output neurons (must be positive)
    "wgt_fbk_out": 1.0     # Connection weight from fbk to output neurons (must be positive)
    }

mc = MotorCortex(N, time_vect, trj, ptMass, pthDat, **mc_param)


######## Connections (error to motor cortex feedback)
for i in range(njt):
    err_pop_p[i].connect(mc.fbk_p[i], rule='one_to_one', w= 1.0)
    err_pop_n[i].connect(mc.fbk_n[i], rule='one_to_one', w=-1.0)


######## Create MUSIC output

proxy_out = nest.Create('music_event_out_proxy', 1, params = {'port_name':'music_out'})

ii=0
for j in range(njt):
    for i, n in enumerate(mc.out_p[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        #print(ii,n)
        ii=ii+1
    for i, n in enumerate(mc.out_n[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        #print(ii,n)
        ii=ii+1

# THIS DOES NOT WORK!
# nest.Connect([n], proxy_out, "one_to_one",{'music_channel': n})

# # 80 neurons (i.e. channels) in total
# print(ii)

# n1 = nest.Create('poisson_generator', 5, [{'rate': 10.0}])
# proxy_out = nest.Create('music_event_out_proxy', 1, params = {'port_name':'music_out'})
# for i, n in enumerate(n1):
#     nest.Connect([n], proxy_out, "one_to_one",{'music_channel': i})

###################### SIMULATE
# Simulate
nest.Simulate(time_span)
