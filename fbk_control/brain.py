#!/usr/bin/env python3

import nest
import numpy as np
import time
import sys
import music
import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../')

import trajectories as tj
from motorcortex import MotorCortex
from population_view import plotPopulation
from util import savePattern
from population_view import PopView
from pointMass import PointMass
from settings import Experiment, Simulation, Brain

nest.Install("util_neurons_module")


########### Simulation

sim = Simulation()

res       = sim.resolution
time_span = sim.timeMax
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

nest.SetKernelStatus({"resolution": res})
nest.SetKernelStatus({"overwrite_files": True})

# Randomize
msd = int( time.time() * 1000.0 )
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})


##################### Experiment

exp = Experiment()

dynSys   = exp.dynSys
njt      = exp.dynSys.numVariables()

init_pos = exp.IC_pos
tgt_pos  = exp.tgt_pos
trj, pol = tj.minimumJerk(init_pos, tgt_pos, time_vect)

pthDat   = exp.pathData


####### Motor cortex

brain = Brain()

# Number of neurons (for each subpopulation positive/negative)
N = brain.nNeurPop

# Precise or approximatesd motor control?
preciseControl = brain.precCtrl

# Motor cortex parameters
mc_param = brain._motCtx_param

# Create motor cortex
mc = MotorCortex(N, time_vect, trj, dynSys, pthDat, preciseControl, **mc_param)


######## Create MUSIC output (to send motor commands)

proxy_out = nest.Create('music_event_out_proxy', 1, params = {'port_name':'music_out'})

ii=0
for j in range(njt):
    for i, n in enumerate(mc.out_p[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1
    for i, n in enumerate(mc.out_n[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1


###################### SIMULATE
# Simulate
nest.Simulate(time_span)


############# PLOTTING


# motCmd = mc.getMotorCommands()
# fig, ax = plt.subplots(2,1)
# ax[0].plot(time_vect,trj)
# ax[1].plot(time_vect,motCmd)
#
#
# lgd = ['x','y']
#
# fig, ax = plt.subplots(2,1)
# for i in range(njt):
#     mc.out_p[i].plot_rate(time_vect,ax=ax[i],bar=False,color='r',label='out')
#     mc.out_n[i].plot_rate(time_vect,ax=ax[i],bar=False,title=lgd[i]+" (Hz)",color='b')
#
#     b,c,pos_r = mc.out_p[i].computePSTH(time_vect,buffer_sz=25)
#     b,c,neg_r = mc.out_n[i].computePSTH(time_vect,buffer_sz=25)
#     if i==0:
#         plt.figure()
#     plt.plot(b[:-1],pos_r-neg_r)
#     plt.xlabel("time (ms)")
#     plt.ylabel("spike rate positive - negative")
#     plt.legend(lgd)
#
# #plt.savefig("mctx_out_pos-neg.png")
# plt.show()
