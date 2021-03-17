#!/usr/bin/env python3

import music
import sys
import queue
import numpy as np
import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../')
import trajectories as tj
import perturbation as pt
from pointMass import PointMass
from settings import Experiment, Simulation, Brain, MusicCfg

import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)


###################### SIMULATION ######################

sim = Simulation()

res     = sim.resolution/1e3            # Resolution (translate into seconds)
timeMax = sim.timeMax/1e3               # Maximum time (translate into seconds)
time    = np.arange(0,timeMax+res,res)  # Time vector
n_time  = len(time)

scale   = 10     # Scaling coefficient to translate spike rates into forces
bufSize = 100/1e3 # Buffer to calculate spike rate (seconds)


##################### EXPERIMENT #####################

exp = Experiment()

pthDat = exp.pathData

# Perturbation
angle = exp.frcFld_angle
k     = exp.frcFld_k

pos_init = exp.IC_pos
vel_init = exp.IC_vel

dynSys = exp.dynSys
njt    = exp.dynSys.numVariables()

# Desired trajectories (only used for testing)
tgt_pos   = exp.tgt_pos
trj,  pol = tj.minimumJerk(pos_init, tgt_pos, time)
aDes, pol = tj.minimumJerk_ddt(pos_init, tgt_pos, time)
inputDes  = exp.dynSys.inverseDyn([],[],aDes)


############################ BRAIN ############################

brain = Brain()

# Number of neurons (for each subpopulation positive/negative)
N = brain.nNeurPop

# Weight (motor cortex - motor neurons)
w = brain.spine_param["wgt_motCtx_motNeur"]


############################## MUSIC CONFIG ##############################

msc = MusicCfg()

in_latency = msc.input_latency

firstId = 0        # First neuron taken care of by this MPI rank
nlocal  = N*2*njt  # Number of neurons taken care of by this MPI rank

# Creation of MUSIC ports
# The MUSIC setup object is used to configure the simulation
setup  = music.Setup()
indata = setup.publishEventInput("music_in")

#NOTE: The actual neuron IDs from the sender side are LOST!!!
# By knowing how many joints and neurons, one should be able to retreive the
# function of each neuron population.

# Input handler function (it is called every time a spike is received)
def inhandler(t, indextype, channel_id):
    # Get the variable corrsponding to channel_id
    var_id = int( channel_id/(N*2) )
    # Get the neuron number within those associated to the variable
    tmp_id = channel_id%(N*2)
    # Identify whether the neuron ID is from the positive otr negative population
    flagSign = tmp_id/N
    if flagSign<1: # Positive population
        spikes_pos[var_id].append([t, channel_id])
        #f_pos[var_id].write("{0}\t{1:3.4f}\n".format(channel_id, t))
    else: # Negative population
        spikes_neg[var_id].append([t, channel_id])
        #f_neg[var_id].write("{0}\t{1:3.4f}\n".format(channel_id, t))
    # Just to handle possible errors
    if flagSign<0 or flagSign>=2:
        raise Exception("Wrong neuron number during reading!")

# Config of the input port
indata.map(inhandler,
        music.Index.GLOBAL,
        base=firstId,
        size=nlocal,
        accLatency=in_latency)


######################## SETUP FILES ##########################

# Files that will contain received spikes
f_pos  = [] # Positive (spikes)
fr_pos = [] #          (rates)
f_neg  = [] # Negative
fr_neg = []
for i in range(njt):
    # Spikes
    tmp = pthDat+"received_dynSys_j"+str(i)
    f_pos.append( open(tmp+"_p.txt", "a") )
    f_neg.append( open(tmp+"_n.txt", "a") )
    # Spike rates
    tmp = pthDat+"motNeur_rate_j"+str(i)
    fr_pos.append( open(tmp+"_p.txt", "a") )
    fr_neg.append( open(tmp+"_n.txt", "a") )


######################## SETUP ARRAYS ################################

# Lists that will contain the positive and negative spikes
# Each element of the list corrspond to a variable to be controlled.
# Each variable is controlled by N*2 nuerons (N positive, N negative).
# These lists will be pupulated by the handler function while spikes are received.
spikes_pos = []
spikes_neg = []
for i in range(njt):
    spikes_pos.append( [] )
    spikes_neg.append( [] )

# Arrays that will contain the spike rates at each time instant
spkRate_pos  = np.zeros([n_time,njt])
spkRate_neg  = np.zeros([n_time,njt])
spkRate_net  = np.zeros([n_time,njt])

# Sequence of position and velocities
pos = np.zeros([n_time,njt])
vel = np.zeros([n_time,njt])

# Sequence of motor commands, perturbation and total input
inputCmd     = np.zeros([n_time,njt]) # Input commands (from motor commands)
perturb      = np.zeros([n_time,njt]) # Perturbation (end-effector)
perturb_j    = np.zeros([n_time,njt]) # Perturbation (joint)
inputCmd_tot = np.zeros([n_time,njt]) # Total input to dynamical system


######################## RUNTIME ##########################

# Function to copute spike rates within within a buffer
def computeRate(spikes, w, nNeurons, timeSt, timeEnd):
    count = 0
    rate  = 0
    if len(spikes)>0:
        tmp = np.array(spikes)
        idx = np.bitwise_and(tmp[:,0]>=timeSt, tmp[:,0]<timeEnd)
        count = w*tmp[idx,:].shape[0] # Number of spiked by weight
        rate  = count/((timeEnd-timeSt)*nNeurons)
    return rate, count


# Start the runtime phase
runtime = music.Runtime(setup, res)
step    = 0 # simulation step

tickt = runtime.time()
while tickt <= timeMax:

    # Position and velocity at the beginning of the timestep
    pos[step,:] = dynSys.pos
    vel[step,:] = dynSys.vel

    # Compute input commands for this timestep
    buf_st = tickt-bufSize # Start of buffer
    buf_ed = tickt         # End of buffer
    for i in range(njt):
        spkRate_pos[step,i], c = computeRate(spikes_pos[i], w, N, buf_st, buf_ed)
        spkRate_neg[step,i], c = computeRate(spikes_neg[i], w, N, buf_st, buf_ed)
        spkRate_net[step,i]    = spkRate_pos[step,i] - spkRate_neg[step,i]
        inputCmd[step,i]       = spkRate_net[step,i] / scale

    perturb[step,:]      = pt.curledForceField(vel[step,:], angle, k)                   # End-effector forces
    perturb_j[step,:]    = np.matmul( dynSys.jacobian(pos[step,:]) , perturb[step,:] )  # Torques
    inputCmd_tot[step,:] = inputCmd[step,:] + perturb_j[step,:]                         # Total torques

    # Integrate dynamical system
    dynSys.integrateTimeStep(inputCmd_tot[step,:], res)

    step = step+1
    runtime.tick()
    tickt = runtime.time()

runtime.finalize()

# To save into disk
#np.savetxt("rate_pos.csv",spkRate_pos,delimiter=',')
#np.savetxt("rate_neg.csv",spkRate_neg,delimiter=',')

for i in range(njt):
    f_pos[i].close()
    fr_pos[i].close()
    f_neg[i].close()
    fr_neg[i].close()


########################### PLOTTING ###########################

plt.figure()
#plt.plot(spkRate_net)
plt.plot(inputCmd)
plt.plot(inputDes,linestyle=':')
plt.xlabel("time (ms)")
plt.ylabel("spike rate positive - negative")
plt.legend(['x','y'])
#plt.savefig("plant_in_pos-neg.png")

plt.figure()
plt.plot(time,pos)
plt.plot(time,trj,linestyle=':')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(['x','y','x_des','y_des'])

plt.figure()
plt.plot(pos[:,0],pos[:,1],color='k')
plt.plot(pos_init[0],pos_init[1],marker='o',color='blue')
plt.plot(tgt_pos[0],tgt_pos[1],marker='o',color='red')
plt.plot(pos[n_time-1,0],pos[n_time-1,1],marker='x',color='k')
plt.xlabel('position x (m)')
plt.ylabel('position y (m)')
plt.legend(['trajectory', 'init','target','final'])

# step_n = 5
# step   = int(pos.shape[0]/step_n)
# for i in range(step_n):
#     idx = i*step
#     plt.arrow(pos[idx,0],pos[idx,1], vel[idx,0],vel[idx,1])
#     #plt.arrow(pos[idx,0],pos[idx,1], inputCmd[idx,0],inputCmd[idx,1])
#     plt.arrow(pos[idx,0],pos[idx,1], perturb[idx,0],perturb[idx,1],color='red')

# fig, ax = plt.subplots(2,1)
# ax[0].plot(time,inputCmd[:,0])
# ax[0].plot(time,perturb[:,0])
# ax[0].plot(time,inputCmd_tot[:,0])
# ax[0].set_ylabel("x")
# ax[1].plot(time,inputCmd[:,1])
# ax[1].plot(time,perturb[:,1])
# ax[1].plot(time,inputCmd_tot[:,1])
# ax[1].set_ylabel("y")

plt.show()
