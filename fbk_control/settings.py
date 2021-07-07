"""Experiment"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"


import sys
import numpy as np


# Just to get the following imports right!
sys.path.insert(1, '../')

from pointMass import PointMass
import perturbation as pt
import os


####################################################################
class Experiment:

    def __init__(self):

        # Where to save data
        self._pathData = "./data/"

        # Initial and target position (end-effector space)
        self._init_pos = np.array([0.0,0.0])
        self._tgt_pos  = np.array([1.0,0.5])
        #self._tgt_pos  = np.array([0.25,0.43])
        #self._tgt_pos  = np.array([-0.25,0.43])

        # Perturbation
        self._frcFld_angle = 90  # Angle of the perturbation force wrt movement velocity
        self._frcFld_k     = 0   # Gain of the perturbation force wrt movement velocity

        # Dynamical system to be controlled (mass and dyn sys object)
        self._m          = 2.0                                     # Mass (kg)
        self._dynSys     = PointMass(mass=self._m)                 # Point mass
        self._dynSys.pos = self._dynSys.inverseKin(self._init_pos) # Initial condition (position)
        self._dynSys.vel = np.array([0.0,0.0])                     # Initial condition (velocity)

    def remove_files(self):
        for f in os.listdir(self._pathData):
            if '.gdf' in f or '.dat' in f:
                os.remove(self._pathData+f)

    @property
    def pathData(self):
        return self._pathData

    @property
    def dynSys(self):
        return self._dynSys

    @property
    def init_pos(self):
        return self._init_pos

    @property
    def tgt_pos(self):
        return self._tgt_pos

    @property
    def frcFld_angle(self):
        return self._frcFld_angle

    @property
    def frcFld_k(self):
        return self._frcFld_k


####################################################################
class Simulation():

    def __init__(self):

        # Nest resolution (milliseconds)
        self._resolution = 0.1

        # Simulation time (milliseconds)
        self._timeMax = 500.0

        # Pause after movement (milliseconds)
        self._timePause = 200.0

        # Number of trials
        self._n_trials = 2

    @property
    def resolution(self):
        return self._resolution

    @property
    def timeMax(self):
        return self._timeMax

    @property
    def timePause(self):
        return self._timePause
    
    @property
    def n_trials(self):
        return self._n_trials


####################################################################
class Brain():

    def __init__(self):

        # Number of neurons for each subpopulation (positive/negative)
        self._nNeurPop = 50

        # Which joint is controlled by the cerebellum (0 = x, 1 = y)
        self._cerebellum_controlled_joint = 0

        self.initPlanner()        # Initialize planner settings
        self.initMotorCortex()    # Initialize motor cortex settings
        self.initStateEstimator() # Initialize state estimator
        self.initSpine()          # Initialize spinal cord settings

        self._connections = {
            "wgt_plnr_mtxFbk"  :  1.0, # Connection weight (excitatory) between Planner and Motor Cortex FBK
            "wgt_stEst_mtxFbk" : -1.0, # Connection weight (inhibitory) between State Estimator and Motor Cortex FBK
            "wgt_spine_stEst"  :  1.0  # Connection weigth (excitatory) between Spine and State Estimator
        }


    def initPlanner(self):

        # Replanning gain
        self._kpl = 0.5

        # Population parameteres
        self._plan_param = {
            "base_rate":  0.0, # Base rate
            "kp":      1200.0  # Gain
            }

    def initMotorCortex(self):

        # If true, motor cortex computes precise motor commands using inv. dynamics
        self._precCtrl = False

        self._motCtx_param = {
            "ffwd_base_rate":  0.0, # Feedforward neurons
            "ffwd_kp":        10.0,
            "fbk_base_rate":   0.0, # Feedback neurons
            "fbk_kp":          0.3, # 0.1 se bidirezionale; 0.01 se unidirezionale
            "out_base_rate":   0.0, # Output neurons
            "out_kp":          1.0,
            "wgt_ffwd_out":    1.0, # Connection weight from ffwd to output neurons (must be positive)
            "wgt_fbk_out":     1.0, # Connection weight from fbk to output neurons (must be positive)
            "buf_sz":         20.0  # Size of the buffer to compute spike rate in basic_neurons (ms)
            }

    def initStateEstimator(self):

        self._k_prediction = 0.0     # Reiability of the prediction input of the state estimator
        self._k_sensory    = 1.0     # Reiability of the sensory feedback input of the state estimator

        self._stEst_param = {
            "out_base_rate":  0.0,   # Summation neurons (i.e. basic_neurons)
            "out_kp":         1.0,   # Gain of the output neurons
            "wgt_scale":      1.0,   # Scale of connection weight from input to output populations (must be positive)
            "buf_sz":        20.0    # Size of the buffer to compute spike rate in basic_neurons (ms)
            }

    def initSpine(self):

        self._firstIdSensNeurons = 0    # First ID of the sensory neurons (keep it at zero)

        self._spine_param = {
            "wgt_motCtx_motNeur" : 1.0, # Weight motor cortex - motor neurons
            "wgt_sensNeur_spine" : 1.0, # Weight sensory neurons - spine
            "sensNeur_base_rate":  0.0, # Sensory neurons baseline rate
            "sensNeur_kp":      1200.0, # Sensory neurons gain
            "fbk_delay":           80.0  # It cannot be less than resolution (ms) [0.1 oppure 80.0]
            }

    @property
    def nNeurPop(self):
        return self._nNeurPop

    @property
    def cerebellum_controlled_joint(self):
        return self._cerebellum_controlled_joint

    @property
    def connections(self):
        return self._connections

    @property
    def plan_param(self):
        return self._plan_param

    @property
    def kpl(self):
        return self._kpl

    @property
    def k_prediction(self):
        return self._k_prediction

    @property
    def k_sensory(self):
        return self._k_sensory

    @property
    def stEst_param(self):
        return self._stEst_param

    @property
    def motCtx_param(self):
        return self._motCtx_param

    @property
    def precCtrl(self):
        return self._precCtrl

    @property
    def firstIdSensNeurons(self):
        return self._firstIdSensNeurons

    @property
    def spine_param(self):
        return self._spine_param


####################################################################
class MusicCfg():

    def __init__(self):
        self._const = 1e-6 # Constant to subtract to avoid rounding errors (ms)

    @property
    def const(self):
        return self._const
