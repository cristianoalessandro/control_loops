"""Sensory neuron class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import music

class SensoryNeuron:

    def __init__(self, id, bas_rate=0.0, kp=1.0):

        self._baseline_rate = bas_rate
        self._gain = kp
        self._rate = 0.0
        self._id = id
        self._spike = []

        # Output port over which sending spikes
        self._outPort = []

    @property
    def id(self):
        return self._id

    @id.setter
    def baseline_rate(self, value):
        self._id = id

    @property
    def baseline_rate(self):
        return self._baseline_rate

    @baseline_rate.setter
    def baseline_rate(self, value):
        self._baseline_rate = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        self._rate = value

    @property
    def spike(self):
        return self._spike

    @property
    def outPort(self):
        return self._outPort

    def connect(self, port):
        self._outPort = port


    # Update theoretical spike rate based on input signal, and generate spikes
    def update(self, signal, resolution, simStep):

        self.rate = self.baseline_rate + self.gain * signal

        # Transform rate into lambda coefficient (Poisson)
        lmd = self.rate*resolution    # Lamda of Poisson distribution
        p   = 1-np.exp(-lmd)          # Probability of at least one spike
        rd  = np.random.uniform(low=0.0,high=1.0,size=1) # Draw uniformly

        #print(p,rd)

        # Send a spike if the probability of drawing at least one
        # Poisson event (p) is higher than chance (rd)
        if (rd<=p):
            self.spike.append([simStep, self.id])
            if self.outPort:
                self.outPort.insertEvent(simStep, self.id, music.Index.GLOBAL)
            else:
                print("Sensory neuron "+str(self.id)+" not connected!")
