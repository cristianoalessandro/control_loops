"""Sensory neuron class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import music

class SensoryNeuron:

    def __init__(self, numNeurons, idStart=0, bas_rate=0.0, kp=1.0):

        self._numNeurons = numNeurons
        self._baseline_rate = bas_rate
        self._gain = kp
        self._rate = 0.0
        self._spike = []

        # Set IDs starting from idStart
        id_vect = np.zeros(shape=numNeurons)
        for i in range(numNeurons):
            id_vect[i]=i+idStart
        self._id=id_vect

        # Output port over which sending spikes
        self._outPort = []

    @property
    def numNeurons(self):
        return self._numNeurons

    @numNeurons.setter
    def baseline_rate(self, value):
        self._numNeurons = value

    @property
    def id(self):
        return self._id

    @id.setter
    def baseline_rate(self, value):
        self._id = value

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

        # Tehoretical rate
        self.rate = self.baseline_rate + self.gain * signal

        # Transform theoretical rate into lambda coefficient (Poisson)
        lmd = self.rate*resolution    # Lamda of Poisson distribution
        rng = np.random.default_rng()
        nEv = rng.poisson(lam=lmd, size=(self.numNeurons))

        for i in range(self.numNeurons):
            if (nEv[i])>0:
                spk_time=simStep*resolution
                self.spike.append([spk_time, self.id[i]])
                if self.outPort:
                    self.outPort.insertEvent(spk_time, self.id[i], music.Index.GLOBAL)
                else:
                    #print("Sensory neuron "+str(self.id)+" not connected!")
                    pass

        # nEv = nEv[0]
        #
        # # Send a spike if at least one event is drawn
        # if (nEv>=1):
        #     spk_time=simStep*resolution
        #     self.spike.append([spk_time, self.id])
        #     if self.outPort:
        #         self.outPort.insertEvent(spk_time, self.id, music.Index.GLOBAL)
        #     else:
        #         #print("Sensory neuron "+str(self.id)+" not connected!")
        #         pass

    def get_events(self):
        spk = self.spike
        # Sort list of spikes based on time of event
        spk.sort(key=sortFirst)
        # Extract times and neuron ids of events
        ts  = np.array(spk)[:,0]
        evs = np.array(spk)[:,1]
        return evs, ts


#################### Class ends here ####################

def sortFirst(val):
    return val[0]
