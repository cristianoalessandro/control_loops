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


####################################################################
class Experiment:

    def __init__(self):

        # Where to save data
        self._pathData = "./data/"

        # Initial and target position (end-effector space)
        self._init_pos = np.array([0.0,0.0])
        self._tgt_pos  = np.array([0.0,0.5])

        # Perturbation
        self._frcFld_angle = 90
        self._frcFld_k     = 1

        # Dynamical system to be controlled (mass and dyn sys object)
        self._m          = 2.0
        self._dynSys     = PointMass(mass=self._m)
        self._dynSys.pos = self._dynSys.inverseKin(self._init_pos) # Initial condition (position)
        self._dynSys.vel = np.array([0.0,0.0])                     # Initial condition (velocity)


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
        self._timeMax = 1000.0

    @property
    def resolution(self):
        return self._resolution

    @property
    def timeMax(self):
        return self._timeMax


####################################################################
class Brain():

    def __init__(self):

        # Number of neurons for each subpopulation (positive/negative)
        self._nNeurPop = 20

        # Initialize motor cortex settings
        self.initMotorCortex()

        # Initialize spinal cord settings
        self.initSpine()


    def initMotorCortex(self):

        # If true, motor cortex computes precise motor commands using inv. dynamics
        self._precCtrl = True

        self._motCtx_param = {
            "ffwd_base_rate":  0.0, # Feedforward neurons
            "ffwd_kp":        10.0,
            "fbk_base_rate":   0.0, # Feedback neurons
            "fbk_kp":         10.0,
            "out_base_rate":   0.0, # Output neurons
            "out_kp":          1.0,
            "wgt_ffwd_out":    1.0, # Connection weight from ffwd to output neurons (must be positive)
            "wgt_fbk_out":     1.0  # Connection weight from fbk to output neurons (must be positive)
            }

    def initSpine(self):

        self._spine_param = {
            "wgt_motCtx_motNeur" : 1.0
            }

    @property
    def nNeurPop(self):
        return self._nNeurPop

    @property
    def motCtx_param(self):
        return self._motCtx_param

    @property
    def precCtrl(self):
        return self._precCtrl

    @property
    def spine_param(self):
        return self._spine_param


####################################################################
class MusicCfg():

    def __init__(self):

        self._input_latency = 0.0001 # seconds

    @property
    def input_latency(self):
        return self._input_latency
