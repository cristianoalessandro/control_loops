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
from planner import Planner
from stateestimator import StateEstimator
from population_view import plotPopulation
from util import savePattern
from population_view import PopView
from pointMass import PointMass
from settings import Experiment, Simulation, Brain, MusicCfg

nest.Install("util_neurons_module")


##################### SIMULATION ########################

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


##################### EXPERIMENT ########################

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


##################### BRAIN ########################

brain = Brain()

# Number of neurons
N    = brain.nNeurPop # For each subpopulation positive/negative
nTot = 2*N*njt        # Total number of neurons

#### Planner
pl_param = brain.plan_param
kpl      = brain.kpl

planner = Planner(N, time_vect, tgt_pos_ee, dynSys, kpl, pthDat, **pl_param)

#### Motor cortex
preciseControl = brain.precCtrl # Precise or approximated ffwd commands?
mc_param       = brain.motCtx_param # Motor cortex parameters

mc = MotorCortex(N, time_vect, trj, dynSys, pthDat, preciseControl, **mc_param)

#### State Estimator
kpred    = brain.k_prediction
ksens    = brain.k_sensory
se_param = brain.stEst_param

se = StateEstimator(N, time_vect, dynSys, kpred, ksens, pthDat, **se_param)

#### Connection Planner - Motor Cortex feedback (excitatory)
wgt_plnr_mtxFbk = brain.connections["wgt_plnr_mtxFbk"]

for j in range(njt):
    planner.pops_p[j].connect( mc.fbk_p[j], rule='one_to_one', w= wgt_plnr_mtxFbk )
    planner.pops_p[j].connect( mc.fbk_n[j], rule='one_to_one', w= wgt_plnr_mtxFbk )
    planner.pops_n[j].connect( mc.fbk_p[j], rule='one_to_one', w=-wgt_plnr_mtxFbk )
    planner.pops_n[j].connect( mc.fbk_n[j], rule='one_to_one', w=-wgt_plnr_mtxFbk )

#### Connection State Estimator - Motor Cortex feedback (Inhibitory)
wgt_stEst_mtxFbk = brain.connections["wgt_stEst_mtxFbk"]

# for j in range(njt):
#     se.out_p[j].connect( mc.fbk_p[j], rule='one_to_one', w= wgt_stEst_mtxFbk )
#     se.out_p[j].connect( mc.fbk_n[j], rule='one_to_one', w= wgt_stEst_mtxFbk )
#     se.out_n[j].connect( mc.fbk_p[j], rule='one_to_one', w=-wgt_stEst_mtxFbk )
#     se.out_n[j].connect( mc.fbk_n[j], rule='one_to_one', w=-wgt_stEst_mtxFbk )


##################### SPINAL CORD ########################

delay_fbk          = brain.spine_param["fbk_delay"]
wgt_sensNeur_spine = brain.spine_param["wgt_sensNeur_spine"]

#### Sensory feedback (Parrot neurons on Sensory neurons)
sn_p=[]
sn_n=[]
for j in range(njt):
    # Positive neurons
    tmp_p = nest.Create ("parrot_neuron", N)
    sn_p.append( PopView(tmp_p, time_vect, to_file=True, label=pthDat+'sens_fbk_'+str(j)+'_p') )
    # Negative neurons
    tmp_n = nest.Create ("parrot_neuron", N)
    sn_n.append( PopView(tmp_n, time_vect, to_file=True, label=pthDat+'sens_fbk_'+str(j)+'_n') )

#### Connection Sensory feedback - State estimator (excitatory)
wgt_spine_stEst = brain.connections["wgt_spine_stEst"]

# for j in range(njt):
#     sn_p[j].connect( se.sens_p[j], rule='one_to_one', w= wgt_spine_stEst )
#     sn_n[j].connect( se.sens_n[j], rule='one_to_one', w=-wgt_spine_stEst )

### DIRECT FROM SENSORY TO MOTOR CORTEX
for j in range(njt):
    sn_p[j].connect( mc.fbk_p[j], rule='one_to_one', w= wgt_stEst_mtxFbk )
    sn_p[j].connect( mc.fbk_n[j], rule='one_to_one', w= wgt_stEst_mtxFbk )
    sn_n[j].connect( mc.fbk_p[j], rule='one_to_one', w=-wgt_stEst_mtxFbk )
    sn_n[j].connect( mc.fbk_n[j], rule='one_to_one', w=-wgt_stEst_mtxFbk )


##################### MUSIC CONFIG ########################

msc = MusicCfg()

#### MUSIC output port (with nTot channels)
proxy_out = nest.Create('music_event_out_proxy', 1, params = {'port_name':'mot_cmd_out'})

ii=0
for j in range(njt):
    for i, n in enumerate(mc.out_p[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1
    for i, n in enumerate(mc.out_n[j].pop):
        nest.Connect([n], proxy_out, "one_to_one",{'music_channel': ii})
        ii=ii+1

#### MUSIC input ports (nTot ports with one channel each)
proxy_in = nest.Create ('music_event_in_proxy', nTot, params = {'port_name': 'fbk_in'})
for i, n in enumerate(proxy_in):
    nest.SetStatus([n], {'music_channel': i})

# Divide channels based on function (using channel order)
for j in range(njt):
    #### Positive channels
    idxSt_p = 2*N*j
    idxEd_p = idxSt_p + N
    nest.Connect( proxy_in[idxSt_p:idxEd_p], sn_p[j].pop, 'one_to_one', {"weight":wgt_sensNeur_spine, "delay":delay_fbk} )
    #### Negative channels
    idxSt_n = idxEd_p
    idxEd_n = idxSt_n + N
    nest.Connect( proxy_in[idxSt_n:idxEd_n], sn_n[j].pop, 'one_to_one', {"weight":wgt_sensNeur_spine, "delay":delay_fbk} )

# We need to tell MUSIC, through NEST, that it's OK (due to the delay)
# to deliver spikes a bit late. This is what makes the loop possible.
nest.SetAcceptableLatency('fbk_in', delay_fbk-msc.const)


###################### SIMULATE

# Simulate
nest.Simulate(time_span)


############# PLOTTING


lgd = ['x','y']

# Positive
fig, ax = plt.subplots(2,1)
for i in range(njt):
    planner.pops_p[i].plot_rate(time_vect,ax=ax[i],bar=False,color='r',label='planner')
    sn_p[i].plot_rate(time_vect,ax=ax[i],bar=False,title=lgd[i]+" (Hz)",color='b',label='sens')
    plt.legend()
plt.suptitle("Positive")

# Negative
fig, ax = plt.subplots(2,1)
for i in range(njt):
    planner.pops_n[i].plot_rate(time_vect,ax=ax[i],bar=False,color='r',label='planner')
    sn_n[i].plot_rate(time_vect,ax=ax[i],bar=False,title=lgd[i]+" (Hz)",color='b',label='sens')
    plt.legend()
plt.suptitle("Negative")

# # MC fbk
for i in range(njt):
    plotPopulation(time_vect, mc.fbk_p[i],mc.fbk_n[i], title=lgd[i],buffer_size=10)
    plt.suptitle("MC fbk")

# # MC ffwd
for i in range(njt):
    plotPopulation(time_vect, mc.ffwd_p[i],mc.ffwd_n[i], title=lgd[i],buffer_size=10)
    plt.suptitle("MC ffwd")

plt.show()


# lgd = ['x','y']
#
# for i in range(njt):
#     plotPopulation(time_vect, planner.pops_p[i],planner.pops_n[i], title=lgd[i],buffer_size=15)
#     plt.suptitle("Planner")
#
# # Sensory feedback
# for i in range(njt):
#     plotPopulation(time_vect, sn_p[i], sn_n[i], title=lgd[i],buffer_size=15)
#     plt.suptitle("Sensory feedback")


# lgd = ['x','y']
#
# # State estimator
# for i in range(njt):
#     plotPopulation(time_vect, se.out_p[i],se.out_n[i], title=lgd[i],buffer_size=15)
#     plt.suptitle("State estimator")
#
# # Sensory feedback
# for i in range(njt):
#     plotPopulation(time_vect, sn_p[i], sn_n[i], title=lgd[i],buffer_size=15)
#     plt.suptitle("Sensory feedback")
#


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
