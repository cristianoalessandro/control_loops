__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np

# Save pattern into file
def savePattern(pattern, file_bas):

    nj = pattern.shape[1]

    for i in range(nj):
        cmd_file = file_bas + "_" + str(i) + ".dat"
        a_file = open(cmd_file, "w")
        np.savetxt( a_file, pattern[:,i] )
        a_file.close()
