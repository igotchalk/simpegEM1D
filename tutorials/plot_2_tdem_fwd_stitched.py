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
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG import maps
from SimPEG.utils import mkvc
import simpegEM1D as em1d
from simpegEM1D.utils import plotLayer, get_vertical_discretization_time

plt.rcParams.update({'font.size': 16})
save_file = True


#####################################################################
# topography
# -------------
#
#

x = np.linspace(50,4950,50)
#x = np.linspace(50,250,3)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)





#####################################################################
# Create Survey
# -------------
#
#
x = np.linspace(50,4950,50)
#x = np.linspace(50,250,3)
n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 20.*np.ones(n_sounding)]
source_current = 1.
source_radius = 5.

receiver_locations = np.c_[x, np.zeros(n_sounding), 20.*np.ones(n_sounding)]
receiver_orientation = "z"  # "x", "y" or "z"

times = np.logspace(-5, -2, 21)

source_list = []

for ii in range(0, n_sounding):
    
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    
    receiver_list = [
        em1d.receivers.TimeDomainPointReceiver(
            receiver_location, times, orientation=receiver_orientation,
            component="dbdt"
        )
    ]

#     Sources
    source_list.append(
        em1d.sources.TimeDomainHorizontalLoopSource(
            receiver_list=receiver_list, location=source_location, a=source_radius,
            I=source_current
        )
    )
    
    # source_list.append(
    #     em1d.sources.TimeDomainMagneticDipoleSource(
    #         receiver_list=receiver_list, location=source_location, orientation="z",
    #         I=source_current
    #     )
    # )

# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a Global Mesh
# ----------------------
#

dx = 100.
hz = get_vertical_discretization_time(times, sigma_background=0.1, n_layer=40)
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
overburden_conductivity = 0.025
slope_conductivity = 0.4

model = np.ones(mesh.nC) * background_conductivity

layer_ind = mesh.gridCC[:, -1] < 30.
model[layer_ind] = overburden_conductivity


x0 = np.r_[0., 30.]
x1 = np.r_[dx*n_sounding, 30.]
x2 = np.r_[dx*n_sounding, 150.]
x3 = np.r_[0., 50.]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh, pts)
model[poly_inds] = slope_conductivity

mapping = maps.ExpMap(mesh)
sounding_models = np.log(model.reshape(mesh.vnC, order='F').flatten())

chi = np.zeros_like(sounding_models)



fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
log_mod = np.log10(model)

mesh.plotImage(
    log_mod, ax=ax1, grid=False,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh.vectorNy.max(), mesh.vectorNy.min())

ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(overburden_conductivity), vmax=np.log10(slope_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)



#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#



# Simulate response for static conductivity
simulation = em1d.simulation_stitched1d.GlobalEM1DSimulationTD(
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


d = np.reshape(dpred, (n_sounding, len(times)))

fig, ax = plt.subplots(1,1, figsize = (7, 7))

for ii in range(0, len(times)):
    ax.semilogy(x, np.abs(d[:, ii]), '-', lw=2)
    
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dBdt| (T/s)")






if save_file == True:

    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dtm_stitched_data.obs'
    
    loc = np.repeat(source_locations, len(times), axis=0)
    fvec = np.kron(np.ones(n_sounding), times)
    
    np.savetxt(
        fname,
        np.c_[loc, fvec, dpred],
        fmt='%.4e'
    )























