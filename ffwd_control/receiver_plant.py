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

bufSize = 50/1e3 # Buffer to calculate spike rate (seconds)

IN_LATENCY = 0.001
timestep   = 0.001 # seconds
stoptime   = 1.0   # seconds

time   = np.arange(0,stoptime+timestep,timestep)
n_time = len(time)

w = 1 # Weight (motor cortex - motor neurons)

# Creation of MUSIC ports
# The MUSIC setup object is used to configure the simulation
setup  = music.Setup()
indata = setup.publishEventInput("music_in")

spikes_pos = []
spikes_neg = []
for i in range(njt):
    #spikes_pos.append( queue.Queue() )
    #spikes_neg.append( queue.Queue() )
    spikes_pos.append( [] )
    spikes_neg.append( [] )


########## CONFIG OF INPUT PORT ##########

#NOTE: The actual neuron IDs from the sender side are LOST!!!
# By knowing how many joints and neurons, I should be able to retreive the
# function of each neuron population.

# Our input handler function
def inhandler(t, indextype, channel_id):
    # Each variable is controlled by N*2 nuerons (N positive, N negative)
    # Get the variable corrsponding to channel_id
    var_id = int( channel_id/(N*2) )
    # Get the neuron number within those associate to the variable
    tmp_id = channel_id%(N*2)
    #print(channel_id, var_id, tmp_id)
    flagSign = tmp_id/N
    if flagSign<1: # Positive population
        #print("Positive")
        #spikes_pos[var_id].put([t, channel_id])
        spikes_pos[var_id].append([t, channel_id])
    else: # Negative population
        #print("Negative")
        #spikes_neg[var_id].put([t, channel_id])
        spikes_neg[var_id].append([t, channel_id])
    if flagSign<0 or flagSign>=2:
        raise Exception("Wrong neuron number during reading!")


firstId = 0        # First neuron taken care of by this MPI rank
nlocal  = N*2*njt  # Number of neurons taken care of by this MPI rank

indata.map(inhandler,
        music.Index.GLOBAL,
        base=firstId,
        size=nlocal,
        accLatency=IN_LATENCY)


################ SETUP FILES ##################

# Files that will contain received spikes
f_pos  = [] # Positive
fr_pos = []
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


################ RUNTIME ##################

# Function to copute spike rates within within a buffer
def computeRate(spikes, w, nNeurons, timeSt, timeEnd):
    count = 0
    rate  = 0
    if len(spikes)>0:
        tmp = np.array(spikes)
        idx = np.bitwise_and(tmp[:,0]>=timeSt, tmp[:,0]<timeEnd)
        count = w*tmp[idx,:].shape[0]
        rate  = count/((timeEnd-timeSt)*nNeurons)
    return rate, count



spkRate_pos = np.zeros([n_time,2])
spkRate_neg = np.zeros([n_time,2])


# Now we start the runtime phase
runtime = music.Runtime(setup, timestep)
step    = 1

tickt = runtime.time()
while tickt < stoptime:

    runtime.tick()
    tickt  = runtime.time()
    buf_st = tickt-bufSize

    for i in range(njt):
        spkRate_pos[step,i], c = computeRate(spikes_pos[i], w, N, buf_st, tickt)
        spkRate_neg[step,i], c = computeRate(spikes_neg[i], w, N, buf_st, tickt)

    step = step+1

    # for i in range(njt):
    #     # Get positive spikes
    #     while not spikes_pos[i].empty():
    #         ev = spikes_pos[i].get()
    #         f_pos[i].write("{0}\t{1:8.4f}\n".format(ev[1], ev[0]))
    #     # Get negative spikes
    #     while not spikes_neg[i].empty():
    #         ev = spikes_neg[i].get()
    #         f_neg[i].write("{0}\t{1:8.4f}\n".format(ev[1], ev[0]))

runtime.finalize()

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
