import nest
import numpy as np
import matplotlib.pyplot as plt

from planner import Planner
from pointMass import PointMass

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

m      = 5.0
ptMass = PointMass(mass=m)

# Neuron neurons
N = 50

time_span = 1000.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

#plant=0.0
tgt = np.array([10.0,2.0])
kpl = 0.2
pthDat = "./data/"

plan_pop = Planner(N, time_vect, plant=ptMass, target=tgt, kPlan=kpl, pathData=pthDat, kp=10.0)


######################## Check trajectories ########################

#print( plan_pop.getTargetDes() )
#print( plan_pop.getTargetPlan() )

final = np.array([10.0,2.0])            # Exemplary reached target (this will be the output of a simulation)
err   = final-plan_pop.getTargetDes()   # Error
plan_pop.updateTarget(err)              # Compute new plan, given error

# End effector space
#plt.figure()
#plt.plot( plan_pop.getTrajDes() )                   # Desired trajectory
#plt.plot( plan_pop.getTrajPlan(), linestyle=':' )   # Planned trajectory
#plt.grid()
#plt.show()

# Joint space
#plt.figure()
#plt.plot( plan_pop.getTrajPlan_j(), linestyle=':' )   # Planned trajectory
#plt.grid()
#plt.show()

# NOTE: a different planned trajectory could lead to the desired trajectory
#       under the influence of the perturbation!


########################### SIMULATE ###########################

# Reset planner to desired trajectory
#err = np.array([0.0,0.0])
#plan_pop.updateTarget(err)
#print(plan_pop.getTargetPlan())

plt.figure()
plt.plot( time_vect, plan_pop.getTrajPlan() )   # Planned trajectory
plt.grid()

# Simulate
nest.Simulate(time_span)

# Array of positive and negative populations
pos = plan_pop.pops_p
neg = plan_pop.pops_n

pos[1].plot_spikes(time_vect)
pos[1].plot_rate(time_vect, 15)
plt.show()

##############

# Add error
#final = np.array([10.0,4.0],ndmin=2)    # Exemplary reached target (this will be the output of a simulation)
#err   = final-plan_pop.getTargetDes()   # Error
#plan_pop.updateTarget(err)              # Compute new plan, given error
#print(plan_pop.getTargetPlan())

#plt.figure()
#plt.plot( plan_pop.getTrajPlan(), linestyle=':' )   # Planned trajectory
#plt.grid()

# Simulate
#nest.SetKernelStatus({'time': 0.0}) # This reset time
#TODO: I should empty all the spike detectors. how to do this?
#nest.Simulate(time_span)

# Array of positive and negative populations
#pos = plan_pop.pops_p
#neg = plan_pop.pops_n

#pos[1].plot_spikes()
#plt.show()
#pos[1].plot_rate(15)
