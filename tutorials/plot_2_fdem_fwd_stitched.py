"""
Forward Simulation of 1D Frequency-Domain Data
==============================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG import maps
from SimPEG.utils import mkvc
import simpegEM1D as em1d
from simpegEM1D.Utils1D import plotLayer
from simpegEM1D.EM1DSimulation import get_vertical_discretization_frequency

plt.rcParams.update({'font.size': 16})
save_file = True


#####################################################################
# topography
# -------------
#
#
x = np.linspace(50,4950,50)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)





#####################################################################
# Create Survey
# -------------
#
#
x = np.linspace(50,5050,50)
n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 30 *np.ones(n_sounding)]
source_current = 1.
source_radius = 5.

receiver_locations = np.c_[x+10., np.zeros(n_sounding), 30 *np.ones(n_sounding)]
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"  # "secondary", "total" or "ppm"

frequencies = np.array([25., 100., 382, 1822, 7970, 35920], dtype=float)

source_list = []

for ii in range(0, n_sounding):
    
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    
    receiver_list = []
    
    receiver_list.append(
        em1d.receivers.HarmonicPointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="real"
        )
    )
    receiver_list.append(
        em1d.receivers.HarmonicPointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="imag"
        )
    )

#     Sources
#    source_list = [
#        em1d.sources.HarmonicHorizontalLoopSource(
#            receiver_list=receiver_list, location=source_location, a=source_radius,
#            I=source_current
#        )
#    ]
    
    source_list.append(
        em1d.sources.HarmonicMagneticDipoleSource(
            receiver_list=receiver_list, location=source_location, orientation="z",
            I=source_current
        )
    )

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Defining a Global Mesh
# ----------------------
#

dx = 100.
hz = get_vertical_discretization_frequency(frequencies, sigma_background=0.1, n_layer=30)
hx = np.ones(n_sounding) * dx
mesh = TensorMesh([hx, hz], x0='00')

###############################################
# Defining a Model
# ----------------------
#

from scipy.spatial import Delaunay
def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.gridCC)>=0
    return inds


background_conductivity = 0.1
slope_conductivity = 1

model = np.ones(mesh.nC) * background_conductivity

x0 = np.r_[0., 10.]
x1 = np.r_[dx*n_sounding, np.sum(hz)]
x2 = np.r_[dx*n_sounding, 10.]
pts = np.vstack((x0, x1, x2, x0))
poly_inds = PolygonInd(mesh, pts)
model[poly_inds] = 1./50

mapping = maps.ExpMap(mesh)
sounding_models = np.log(model.reshape(mesh.vnC, order='F').flatten())

chi = np.zeros_like(sounding_models)



cb = plt.colorbar(
    mesh.plotImage(model, grid=False, clim=(1e-2, 1e-1),pcolorOpts={"norm":LogNorm()})[0],
    fraction=0.03, pad=0.04
)

plt.ylim(mesh.vectorNy.max(), mesh.vectorNy.min())
plt.gca().set_aspect(1)

#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#



# Simulate response for static conductivity
simulation = em1d.simulation_stitched1d.GlobalEM1DSimulationFD(
    mesh, survey=survey, sigmaMap=mapping, chi=chi, hz=hz, topo=topo, parallel=False, n_cpu=2, verbose=True,
    Solver=PardisoSolver
)

#simulation.model = sounding_models
#
#ARGS = simulation.input_args(0)
#print("Number of arguments")
#print(len(ARGS))
#print("Print arguments")
#for ii in range(0, len(ARGS)):
#    print(ARGS[ii])

dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


d = np.reshape(dpred, (n_sounding, 2*len(frequencies))).T

fig, ax = plt.subplots(1,1, figsize = (7, 7))

for ii in range(0, n_sounding):
    ax.loglog(frequencies, np.abs(d[0:len(frequencies), ii]), '-', lw=2)
    ax.loglog(frequencies, np.abs(d[len(frequencies):, ii]), '--', lw=2)
    
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend(["real", "imaginary"])

#
#d = np.reshape(dpred, (n_sounding, 2*len(frequencies)))
#fig = plt.figure(figsize = (10, 5))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#
#for ii in range(0, n_sounding):
#    ax1.semilogy(x, np.abs(d[:, 0:len(frequencies)]), 'k-', lw=2)
#    ax2.semilogy(x, np.abs(d[:, len(frequencies):]), 'k--', lw=2)



if save_file == True:

    noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dfm_stitched_data.obs'
    
    loc = np.repeat(source_locations, len(frequencies), axis=0)
    fvec = np.kron(np.ones(n_sounding), frequencies)
    dout = np.c_[dpred[0::2], dpred[1::2]]
    
    np.savetxt(
        fname,
        np.c_[loc, fvec, dout],
        fmt='%.4e'
    )























