#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import trajectories as tj

from sensoryneuron import SensoryNeuron
from population_view import PopViewSpine

# Time span and time vector expressed in s
res = 0.0001
time_span = 1.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)
time_n    = len(time_vect)

# Poitions expressed in meters (mks system)
init_pos  = np.array([0.0])
final_pos = np.array([2.0])
trj, pol  = tj.minimumJerk(init_pos, final_pos, time_vect)

id = 0
sn = SensoryNeuron(id, bas_rate=0.0, kp=1.0)

for i in range(time_n):
    sn.update(trj[i], res, i)

print(np.array(sn.spike))

# plt.figure()
# plt.plot(time_vect,trj)
# plt.show()
