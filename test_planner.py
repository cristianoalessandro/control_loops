import nest
import numpy as np
import matplotlib.pyplot as plt
import trajectories as tj

from planner import Planner
from pointMass import PointMass
from population_view import plotPopulation

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

flagSaveFig = False
figPath = './fig/planner/'
pthDat = "./data/"

pos_i  = np.array([0.0,0.0])
tgt    = np.array([10.0,2.0]) # Desired target
final  = np.array([10.0,2.0]) # Exemplary reached target (this will be the output of a simulation)
kpl    = 0.5                  # Coefficient across-trial plan adjustement

m      = 5.0
ptMass = PointMass(mass=m,IC_pos=pos_i)
njt    = ptMass.numVariables()

N   = 50   # Neuron neurons

time_span = 1000.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

plan_pop = Planner(N, time_vect, plant=ptMass, target=tgt, kPlan=kpl, pathData=pthDat, kp=10.0)


######################## Check trajectories ########################

#plan_pop.setTargetPlan(np.array([10.0,0.5]))

final_trj, pol = tj.minimumJerk(pos_i, final, time_vect)
err = final-plan_pop.getTargetDes()     # Error
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

# Simulate
nest.Simulate(time_span)


########################### PLOTTING ###########################

lgd = ['x plan','y plan','x real','y real','x tgt des','y tgt des']

plt.figure()
plt.plot( time_vect, plan_pop.getTrajPlan() )   # Planned trajectory
plt.plot( time_vect, final_trj, linestyle=':' )   # Planned trajectory
plt.plot( time_span, np.reshape(tgt,(1,2)), marker='o' )   # Planned trajectory
plt.ylabel("Trajectory (cm)")
plt.xlabel("time (ms)")
plt.legend(lgd)
plt.grid()

if flagSaveFig:
    plt.savefig(figPath+"planner_trj_i_"+str(pos_i)+"_tgt_"+str(tgt)+"_err_"+str(err)+".png",format="png")
    plt.savefig(figPath+"planner_trj_i_"+str(pos_i)+"_tgt_"+str(tgt)+"_err_"+str(err)+".svg",format="svg")



# Array of positive and negative populations
pos = plan_pop.pops_p
neg = plan_pop.pops_n

# Populations
lgd = ['x','y']
max_y = np.empty(shape=(njt,2))
axv = []
fgv = []
for i in range(njt):
    fig, ax = plotPopulation(time_vect, pos[i], neg[i], lgd[i] )
    fgv.append(fig)
    axv.append(ax)
    max_y[i,:] = plt.gca().get_ylim()

max_y=np.max(max_y[:,1])
for i in range(njt):
    #axv[i][1].set_ylim(top=max_y)
    if i==1:
        axv[i][1].set_ylim(top=30)
    if flagSaveFig:
        fgv[i].savefig(figPath+"planner_neural_j"+str(i)+"_i_"+str(pos_i)+"_tgt_"+str(tgt)+"_err_"+str(err)+"_noUniform.png",format="png")
        #fgv[i].savefig(figPath+"planner_neural_j"+str(i)+"_i_"+str(pos_i)+"_tgt_"+str(tgt)+"_err_"+str(err)+".svg",format="svg")

plt.show()


#pos[1].plot_spikes(time_vect)
#pos[1].plot_rate(time_vect, 15)

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
