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
from sensoryneuron import SensoryNeuron
from settings import Experiment, Simulation, Brain, MusicCfg
from util import plotPopulation

import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)


saveFig = False
pathFig = './fig/'
cond = 'ffwd_noFF_'


###################### SIMULATION ######################

sim = Simulation()

res     = sim.resolution/1e3            # Resolution (translate into seconds)
timeMax = sim.timeMax/1e3               # Maximum time (translate into seconds)
time    = np.arange(0,timeMax+res,res)  # Time vector
n_time  = len(time)

scale   = 10.0   # Scaling coefficient to translate spike rates into forces (must be >=1)
bufSize = 10/1e3 # Buffer to calculate spike rate (seconds)


##################### EXPERIMENT #####################

exp = Experiment()

pthDat = exp.pathData

# Perturbation
angle = exp.frcFld_angle
k     = exp.frcFld_k

# Dynamical system
dynSys = exp.dynSys
njt    = exp.dynSys.numVariables()

# Desired trajectories (only used for testing)
# End-effector space
init_pos_ee = exp.init_pos
tgt_pos_ee  = exp.tgt_pos
trj_ee, pol = tj.minimumJerk(init_pos_ee, tgt_pos_ee, time)

# Joint space
init_pos = dynSys.inverseKin( init_pos_ee )
tgt_pos  = dynSys.inverseKin( tgt_pos_ee )
trj      = dynSys.inverseKin( trj_ee )
trj_d    = np.gradient(trj,res,axis=0)
trj_dd   = np.gradient(trj_d,res,axis=0)
inputDes = exp.dynSys.inverseDyn(trj,trj_d,trj_dd)


############################ BRAIN ############################

brain = Brain()

# Number of neurons (for each subpopulation positive/negative)
N = brain.nNeurPop

# Weight (motor cortex - motor neurons)
w = brain.spine_param["wgt_motCtx_motNeur"]

# Sensory feedback delay (seconds)
delay_fbk = brain.spine_param["fbk_delay"]/1e3

# First ID sensory neurons
sensNeur_idSt     = brain.firstIdSensNeurons
sensNeur_baseRate = brain.spine_param["sensNeur_base_rate"]
sensNeur_gain     = brain.spine_param["sensNeur_kp"]


############################## MUSIC CONFIG ##############################

msc = MusicCfg()

# Compute the acceptable latency (AL) of this input port to make sure that
# Sum(ALs)>=Sum(ITIs). ITI stands for Inter Tick Intervals.
accLat = 2*res-(delay_fbk-msc.const/1e3)
# If AL<0, set it to zero (this will satisfy the relationship above)
if accLat<0:
    accLat=0

firstId = 0        # First neuron taken care of by this MPI rank
nlocal  = N*2*njt  # Number of neurons taken care of by this MPI rank

# Creation of MUSIC ports
# The MUSIC setup object is used to configure the simulation
setup   = music.Setup()
indata  = setup.publishEventInput("mot_cmd_in")
outdata = setup.publishEventOutput("fbk_out")

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
    else: # Negative population
        spikes_neg[var_id].append([t, channel_id])
    # Just to handle possible errors
    if flagSign<0 or flagSign>=2:
        raise Exception("Wrong neuron number during reading!")

# Config of the input port
indata.map(inhandler,
           music.Index.GLOBAL,
           base=firstId,
           size=nlocal,
           accLatency=accLat)

# Config of the output port
outdata.map (music.Index.GLOBAL,
             base=firstId,
             size=nlocal)


################ SENSORY NEURONS

sn_p = [] # Positive sensory neurons
sn_n = [] # Negative sensory neurons
for i in range(njt):
    # Positive
    idSt_p = sensNeur_idSt+2*N*i
    tmp    = SensoryNeuron(N, pos=True, idStart=idSt_p, bas_rate=sensNeur_baseRate, kp=sensNeur_gain)
    tmp.connect(outdata)   # Connect to output port
    sn_p.append(tmp)
    # Negative
    idSt_n = idSt_p+N
    tmp    = SensoryNeuron(N, pos=False, idStart=idSt_n, bas_rate=sensNeur_baseRate, kp=sensNeur_gain)
    tmp.connect(outdata)   # Connect to output port
    sn_n.append(tmp)


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
pos   = np.zeros([n_time,2])   # End-effector space
vel   = np.zeros([n_time,2])
pos_j = np.zeros([n_time,njt]) # Joint space
vel_j = np.zeros([n_time,njt])

# Sequence of motor commands, perturbation and total input
inputCmd     = np.zeros([n_time,njt]) # Input commands (from motor commands)
perturb      = np.zeros([n_time,2])   # Perturbation (end-effector)
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
    pos_j[step,:] = dynSys.pos                      # Joint space
    vel_j[step,:] = dynSys.vel
    pos[step,:]   = dynSys.forwardKin( dynSys.pos ) # End-effector space
    vel[step,:]   = dynSys.forwardKin( dynSys.vel )

    # Send sensory feedback and compute input commands for this timestep
    buf_st = tickt-bufSize # Start of buffer
    buf_ed = tickt         # End of buffer
    for i in range(njt):
        # Generate and send sensory feedback spikes given plan position
        sn_p[i].update(pos_j[step,i], res, tickt)
        sn_n[i].update(pos_j[step,i], res, tickt)
        # Compute input commands
        spkRate_pos[step,i], c = computeRate(spikes_pos[i], w, N, buf_st, buf_ed)
        spkRate_neg[step,i], c = computeRate(spikes_neg[i], w, N, buf_st, buf_ed)
        spkRate_net[step,i]    = spkRate_pos[step,i] - spkRate_neg[step,i]
        inputCmd[step,i]       = spkRate_net[step,i] / scale

    perturb[step,:]      = pt.curledForceField(vel[step,:], angle, k)                     # End-effector forces
    perturb_j[step,:]    = np.matmul( dynSys.jacobian(pos_j[step,:]) , perturb[step,:] )  # Torques
    inputCmd_tot[step,:] = inputCmd[step,:] + perturb_j[step,:]                           # Total torques

    # Integrate dynamical system
    dynSys.integrateTimeStep(inputCmd_tot[step,:], res)

    step = step+1
    runtime.tick()
    tickt = runtime.time()

runtime.finalize()


########################### SAVING INTO DISK ###########################

def firstElement(elem):
    return elem[0]

# Spikes (of each neuron within the population for each joint)
for i in range(njt):

    ########### Motor commands (input from MUSIC)
    # Positive
    tmp_fnm_p = pthDat+"motNeur_inSpikes_j"+str(i)+"_p.txt"
    if len(spikes_pos[i])>0:
        spikes_pos[i].sort(key=firstElement)
        tmp_dat_p = np.array( spikes_pos[i] )
        np.savetxt(tmp_fnm_p, tmp_dat_p, fmt='%3.4f\t%d', delimiter='\t')
    else:
        tmp_dat_p = np.array( spikes_pos[i] )
        np.savetxt(tmp_fnm_p, tmp_dat_p)

    # Negative
    tmp_fnm_n = pthDat+"motNeur_inSpikes_j"+str(i)+"_n.txt"
    if len(spikes_neg[i])>0:
        spikes_neg[i].sort(key=firstElement)
        tmp_dat_n = np.array( spikes_neg[i] )
        np.savetxt(tmp_fnm_n, tmp_dat_n, fmt='%3.4f\t%d', delimiter='\t' )
    else:
        tmp_dat_n = np.array( spikes_neg[i] )
        np.savetxt(tmp_fnm_n, tmp_dat_n)

    ########### Sensory neurons (output to MUSIC)
    # Positive
    tmp_fnm_p = pthDat+"sensNeur_outSpikes_j"+str(i)+"_p.txt"
    if len(sn_p[i].spike)>0:
        sn_p[i].spike.sort(key=firstElement)
        tmp_dat_p = np.array( sn_p[i].spike )
        np.savetxt(tmp_fnm_p, tmp_dat_p, fmt='%3.4f\t%d', delimiter='\t')
    else:
        tmp_dat_p = np.array( sn_p[i].spike )
        np.savetxt(tmp_fnm_p, tmp_dat_p)

    # Negative
    tmp_fnm_n = pthDat+"sensNeur_outSpikes_j"+str(i)+"_n.txt"
    if len(sn_n[i].spike)>0:
        sn_n[i].spike.sort(key=firstElement)
        tmp_dat_n = np.array( sn_n[i].spike )
        np.savetxt(tmp_fnm_n, tmp_dat_n, fmt='%3.4f\t%d', delimiter='\t')
    else:
        tmp_dat_n = np.array( sn_n[i].spike )
        np.savetxt(tmp_fnm_n, tmp_dat_n)


# Motor neuron spike rates (of each population for each joint)
np.savetxt(pthDat+"motNeur_rate_pos.csv",spkRate_pos, delimiter=',')
np.savetxt(pthDat+"motNeur_rate_neg.csv",spkRate_neg, delimiter=',')

# Position and velocities (joint space)
np.savetxt( pthDat+"pos_real_joint.csv", pos_j, delimiter=',' )
np.savetxt( pthDat+"vel_real_joint.csv", vel_j, delimiter=',' )

# Position and velocities (end-effector space)
np.savetxt( pthDat+"pos_real_ee.csv", pos, delimiter=',' )
np.savetxt( pthDat+"vel_real_ee.csv", vel, delimiter=',' )

# Desired trajectory
np.savetxt( pthDat+"pos_des_ee.csv", trj_ee, delimiter=',' ) # End-effector
np.savetxt( pthDat+"pos_des_joint.csv", trj, delimiter=',' ) # Joints

# Motor commands
np.savetxt( pthDat+"inputCmd_des.csv", inputDes, delimiter=',' )     # Desired torques
np.savetxt( pthDat+"inputCmd_motNeur.csv", inputCmd, delimiter=',' ) # Torques from motor neurons
np.savetxt( pthDat+"inputCmd_tot.csv", inputCmd_tot, delimiter=',' ) # Torques from motor neurons + perturbation
np.savetxt( pthDat+"perturbation_ee.csv", perturb, delimiter=',' )   # Perturbation force in end-effector space
np.savetxt( pthDat+"perturbation_j.csv", perturb_j, delimiter=',' )  # Perturbation torque


########################### PLOTTING ###########################

lgd = ['x','y','des']

plt.figure()
plt.plot(time,inputCmd)
plt.plot(time,inputDes,linestyle=':')
plt.xlabel("time (s)")
plt.ylabel("motor commands (N)")
plt.legend(lgd)
if saveFig:
    plt.savefig(pathFig+cond+"motCmd.png")

# Joint space
plt.figure()
plt.plot(time,pos_j)
plt.plot(time,trj,linestyle=':')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(['x','y','x_des','y_des'])
if saveFig:
    plt.savefig(pathFig+cond+"position_joint.png")

# End-effector space
plt.figure()
plt.plot(pos[:,0],pos[:,1],color='k')
plt.plot(init_pos_ee[0],init_pos_ee[1],marker='o',color='blue')
plt.plot(tgt_pos_ee[0],tgt_pos_ee[1],marker='o',color='red')
plt.plot(pos[n_time-1,0],pos[n_time-1,1],marker='x',color='k')
plt.axis('equal')
plt.xlabel('position x (m)')
plt.ylabel('position y (m)')
plt.legend(['trajectory', 'init','target','final'])
if saveFig:
    plt.savefig(pathFig+cond+"position_ee.png")

# Show sensory neurons
# for i in range(njt):
#     plotPopulation(time, sn_p[i], sn_n[i], title=lgd[i],buffer_size=0.015)

plt.show()
