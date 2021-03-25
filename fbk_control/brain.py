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
from settings import Experiment, Simulation, Brain, MusicCfg

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

# End-effector space
init_pos_ee = exp.init_pos
tgt_pos_ee  = exp.tgt_pos
trj_ee, pol = tj.minimumJerk(init_pos_ee, tgt_pos_ee, time_vect)

# Joint space
init_pos = dynSys.inverseKin( init_pos_ee )
tgt_pos  = dynSys.inverseKin( tgt_pos_ee )
trj      = dynSys.inverseKin( trj_ee )

pthDat   = exp.pathData


####### Motor cortex

brain = Brain()

# Number of neurons
N    = brain.nNeurPop # For each subpopulation positive/negative
nTot = 2*N*njt        # Total number of neurons

# Precise or approximatesd motor control?
preciseControl = brain.precCtrl

# Motor cortex parameters
mc_param = brain._motCtx_param

# Create motor cortex
mc = MotorCortex(N, time_vect, trj, dynSys, pthDat, preciseControl, **mc_param)

delay_fbk = brain.spine_param["fbk_delay"]
wgt_spine = brain.spine_param["wgt_sensNeur_spine"]


### RECEIVING NETWORK

sn_p=[]
sn_n=[]
for j in range(njt):
    # Positive neurons
    tmp_p = nest.Create ("parrot_neuron", N)
    sn_p.append( PopView(tmp_p, time_vect, to_file=True, label=pthDat+'sens_fbk_'+str(j)+'_p') )
    # Negative neurons
    tmp_n = nest.Create ("parrot_neuron", N)
    sn_n.append( PopView(tmp_n, time_vect, to_file=True, label=pthDat+'sens_fbk_'+str(j)+'_n') )


######## Create MUSIC output

msc = MusicCfg()

proxy_out = nest.Create('music_event_out_proxy', 1, params = {'port_name':'mot_cmd_out'})

ii=0
for j in range(njt):
    for i, n in enumerate(mc.out_p[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1
    for i, n in enumerate(mc.out_n[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1


# Creation of MUSIC input port with N input channels
proxy_in = nest.Create ('music_event_in_proxy', nTot, params = {'port_name': 'fbk_in'})
for i, n in enumerate(proxy_in):
    nest.SetStatus([n], {'music_channel': i})

# Divide channels based on function (using channel order)
for j in range(njt):
    #### Positive channels
    idxSt_p = 2*N*j
    idxEd_p = idxSt_p + N
    nest.Connect( proxy_in[idxSt_p:idxEd_p], sn_p[j].pop, 'one_to_one', {"weight":wgt_spine, "delay":delay_fbk} )
    #### Negative channels
    idxSt_n = idxEd_p
    idxEd_n = idxSt_n + N
    nest.Connect( proxy_in[idxSt_n:idxEd_n], sn_n[j].pop, 'one_to_one', {"weight":wgt_spine, "delay":delay_fbk} )

# We need to tell MUSIC, through NEST, that it's OK (due to the delay)
# to deliver spikes a bit late. This is what makes the loop possible.
nest.SetAcceptableLatency('fbk_in', delay_fbk-msc.const)


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
