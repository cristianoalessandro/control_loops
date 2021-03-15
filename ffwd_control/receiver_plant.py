#!/usr/bin/env python3

import music
import sys
import queue
import numpy as np
import matplotlib.pyplot as plt

import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

pthDat = "./data/"

N   = 5  # Number of postive/negative neurons (will eventually come from an external parameter)
njt = 2  # Number of joints (will eventually come from plant object)

bufSize = 100/1e3 # Buffer to calculate spike rate (seconds)

IN_LATENCY = 0.001
timestep   = 0.001 # seconds (should come from NEST simulation)
stoptime   = 1.0   # seconds (should come from NEST simulation)

w = 1 # Weight (motor cortex - motor neurons)

time   = np.arange(0,stoptime+timestep,timestep)
n_time = len(time)

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
spkRate_pos = np.zeros([n_time,njt])
spkRate_neg = np.zeros([n_time,njt])


############################## MUSIC CONFIG ##############################

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
        accLatency=IN_LATENCY)


######################## SETUP FILES ##########################

# Files that will contain received spikes
f_pos  = [] # Positive (spikes)
fr_pos = [] #          (rates)
f_neg  = [] # Negative
fr_neg = []
for i in range(njt):
    # Spikes
    tmp = pthDat+"receiver_j"+str(i)
    f_pos.append( open(tmp+"_p.txt", "a") )
    f_neg.append( open(tmp+"_n.txt", "a") )
    # Spike rates
    tmp = pthDat+"rate_j"+str(i)
    fr_pos.append( open(tmp+"_p.txt", "a") )
    fr_neg.append( open(tmp+"_n.txt", "a") )


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
runtime = music.Runtime(setup, timestep)
step    = 1 # simulation step

tickt = runtime.time()
while tickt < stoptime:

    runtime.tick()
    tickt  = runtime.time()
    buf_st = tickt-bufSize # Start of buffer
    buf_ed = tickt         # End of buffer
    #print(buf_st,buf_ed)

    for i in range(njt):
        spkRate_pos[step,i], c = computeRate(spikes_pos[i], w, N, buf_st, buf_ed)
        spkRate_neg[step,i], c = computeRate(spikes_neg[i], w, N, buf_st, buf_ed)

    step = step+1

runtime.finalize()

# To save into disk
#np.savetxt("rate_pos.csv",spkRate_pos,delimiter=',')
#np.savetxt("rate_neg.csv",spkRate_neg,delimiter=',')

plt.figure()
plt.plot(spkRate_pos)
plt.plot(spkRate_neg)

plt.figure()
plt.plot(spkRate_pos-spkRate_neg)
plt.xlabel("time (ms)")
plt.ylabel("spike rate positive - negative")
plt.legend(['x','y'])
#plt.savefig("plant_in_pos-neg.png")
plt.show()
