#!/usr/bin/env python3

import music
import sys
import queue

import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

IN_LATENCY = 0.001
timestep   = 0.001
stoptime   = 1.0

# Creation of MUSIC ports
# The MUSIC setup object is used to configure the simulation
setup  = music.Setup()
indata = setup.publishEventInput("music_in")

in_q = queue.Queue()

########## CONFIG OF INPUT PORT ##########
# Our input handler function
def inhandler(t, indextype, channel_id):
    in_q.put([t, channel_id])

firstId = 0  # First neuron taken care of by this MPI rank
nlocal  = 20 # Number of neurons taken care of by this MPI rank

indata.map(inhandler,
        music.Index.GLOBAL,
        base=firstId,
        size=nlocal,
        accLatency=IN_LATENCY)

#NOTE: The actual neuron IDs from the sender side are LOST!!!
# By knowing how many joints and neurons, I should be able to retreive the
# functoin of each neuron population.

################ RUNTIME ##################

# Now we start the runtime phase
runtime = music.Runtime(setup, timestep)

tickt = runtime.time()
while tickt < stoptime:
    runtime.tick()
    tickt = runtime.time()
    while not in_q.empty():
        ev = in_q.get()
        print(ev[1], ev[0])
        #f.write("{0}\t{1:8.4f}\n".format(ev[1], ev[0]))

runtime.finalize()
