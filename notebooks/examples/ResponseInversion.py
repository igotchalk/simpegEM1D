import sys, os
from SimPEG import *
from simpegEM1D import (
    EM1D, EM1DSurveyTD, Utils1D, get_vertical_discretization_time, 
    set_mesh_1d, skytem_HM_2015
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle
import ipysheet
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from IPython.display import clear_output,display
from ipywidgets import (
    interactive,
    IntSlider,
    widget,
    FloatText,
    FloatSlider,
    FloatLogSlider,
    Checkbox,
    ToggleButtons,
    Button,
    Output
)

  
rhomin=0.1
rhomax=1000.
rho_default=50.
zmin=0.
zmax=350
rho_half=30.
tx_alt= 30.

time = np.logspace(-5, -2, 31)
hz = get_vertical_discretization_time(time, facter_tmax=0.3, factor_tmin=10., n_layer=19)
mesh1D = set_mesh_1d(hz)
depth = -mesh1D.gridN[:-1]
LocSigZ = -mesh1D.gridCC
TDsurvey = EM1DSurveyTD(
    rx_location = np.array([0., 0., 100.+tx_alt]),
    src_location = np.array([0., 0., 100.+tx_alt]),
    topo = np.r_[0., 0., 100.],
    depth = depth,
    rx_type = 'dBzdt',
    wave_type = 'stepoff',
    src_type = 'CircularLoop',
    a = 13.,
    I = 1.,
    time = time,
    base_frequency = 25.,
    use_lowpass_filter=False,
    high_cut_frequency=210*1e3        
)
sig_half = 1./20.
sig_blk = sig_half * 20.
chi_half = 0.
expmap = Maps.ExpMap(mesh1D)


# decorater used to block function printing to the console
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

def define_sigma_layers(sig_half,sig_lay,layers):
    sig  = np.ones(TDsurvey.n_layer)*sig_half
    for i in range(len(layers)):
        if i==len(layers)-1:
            ind = (layers[i]>LocSigZ)
        else:
            ind = np.where((layers[i]>LocSigZ) & (layers[i+1]<=LocSigZ))
    #     print(i,sig_lay[i],ind)
        print(sig_half,sig_lay,layers)
        sig[ind] = sig_lay[i]
    return sig
        
    
def plot_mesh(sig,mesh1D):
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(1,1, figsize=(5, 8))
    Utils1D.plotLayer(1./sig, mesh1D, showlayers=True,xlim=(rhomin*.9,rhomax*1.1),label='True resistivity')
    ax.set_ylim(-50,350)
    xlims = ax.get_xlim()
    rect = Rectangle((xlims[0],-50),xlims[1]-xlims[0],50,fc=[.8,.8,.8])
    ax.annotate('Ground surface', xy=(xlims[1],0),  xycoords='data',
                fontsize=14,xytext=(2000.,40), textcoords='data',
                arrowprops=dict(arrowstyle="->"),
                horizontalalignment='left', verticalalignment='top',
                )
    ax.annotate('Helicopter transmitter height', xy=(xlims[1],-tx_alt),  xycoords='data',
                fontsize=14,xytext=(2000.,-tx_alt-40), textcoords='data',
                arrowprops=dict(arrowstyle="->"),
                horizontalalignment='left', verticalalignment='top',
                )
    plt.hlines(-tx_alt, xlims[0], xlims[1], colors='k', linestyles='dashed')
    txt = ax.text(0.5, 0.94, "Air",transform=ax.transAxes,
                size=22,ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=8, foreground=[.8,.8,.8])])


#     plt.annotate()
    ax.add_patch(rect)
    ax.invert_yaxis()
    ax.set_xlabel(r'Resistivity $\rho$ (ohm-m)')
    return fig,ax



def plot_res_model(rho1,rho2,rho_half,lay1,lay2):
    layers = -np.r_[0,lay1,lay2]
    sig1=1./rho1
    sig2=1./rho2
    sig_half=1./rho_half
    sig_lay = np.r_[sig1,sig2,sig_half]
    sig = define_sigma_layers(sig_half,sig_lay,layers)
    plot_mesh(sig,mesh1D)
    plt.show()
    return sig



def interactive_res_model():
    app = interactive(
        plot_res_model,
        lay1=FloatSlider(
            min=zmin,
            max=zmax,
            step=1.,
            value=30.,
            continuous_update=True,
            description="$layer2_{top}$",
        ),
        lay2=FloatSlider(
            min=zmin,
            max=zmax,
            value=100.,
            step=1.,
            continuous_update=True,
            description="$layer2_{bottom}$",
        ),
        rho_half=FloatLogSlider(
            base=10,
            min=np.log10(rhomin),
            max=np.log10(rhomax),
            value=50.,
            step=0.1,
            continuous_update=True,
            description=r"$\rho_{background}$",
        ),
        rho1=FloatLogSlider(
            base=10,
            min=np.log10(rhomin),
            max=np.log10(rhomax),
            value=100.,
            step=0.1,
            continuous_update=True,
            description=r"$\rho_{1}$",
        ),
        rho2=FloatLogSlider(
            base=10,
            min=np.log10(rhomin),
            max=np.log10(rhomax),
            value=10.,
            step=0.1,
            continuous_update=True,
            description=r"$\rho_{2}$",
        )
    )
    return app






def create_prob(w):
    '''
    w: ipywidget containing values for model (in conductivity), rho1,rho2,lay1,lay2
    '''
    #Problem
    m_true = np.log(w.result)
    prob = EM1D(mesh1D, sigmaMap=expmap, verbose=False)
    if prob.ispaired:
        prob.unpair()
    if TDsurvey.ispaired:
        TDsurvey.unpair()
    prob.pair(TDsurvey)
    prob.chi = np.zeros(TDsurvey.n_layer)
    d_true = TDsurvey.dpred(m_true)
    return m_true,d_true,prob,TDsurvey

def plot_data(data,TDSurvey):
    f, ax = plt.subplots(1,1, figsize = (5,4))
    ax.plot(TDsurvey.time, -data, '.-',label='True data')
    ax.set_xscale('log');
    ax.set_yscale('log');
    ax.legend()
    plt.xlabel('time (s)')
    plt.ylabel('db/dt ($V/m^{2}$)')
    plt.show()
    return

def interactive_fwd(res_model_app):

    button = Button(description="Acquire AEM data!")
    output = Output()
    display(button, output)

    def on_button_clicked(b):
        with output:
            clear_output(True)
            m_true,d_true,prob,TDsurvey = create_prob(res_model_app)
            plot_data(d_true,TDsurvey)
            show_inline_matplotlib_plots()
        return
    button.on_click(on_button_clicked)

    show_inline_matplotlib_plots()
    return button,output






def prob_and_run(w):

    m_true,d_true,prob,TDsurvey = create_prob(w)
    #Data and noise 
    np.random.seed(1)
    TDsurvey.dtrue = d_true
    std = 0.05
    noise = std*abs(TDsurvey.dtrue)*np.random.randn(*TDsurvey.dtrue.shape)
    floor = 0.
    std = 0.07
    TDsurvey.dobs = TDsurvey.dtrue+noise
    uncert = abs(TDsurvey.dobs)*std+floor

    dmisfit = DataMisfit.l2_DataMisfit(TDsurvey)
    uncert = (abs(TDsurvey.dobs)*std+floor)
    dmisfit.W = 1./ uncert

    #Starting model
    m0 = np.log(np.ones_like(w.result)* (1./w.kwargs['rho_half']))
    d_0 = TDsurvey.dpred(m0)

    reg = Regularization.Sparse(
        mesh1D,
        mapping=Maps.IdentityMap(mesh1D),
        alpha_s=1.,
        alpha_x=1.
    )
    p = 0
    qx, qz = 2., 2.
    reg.norms = np.c_[p, qx, qz, 0.]
    IRLS = Directives.Update_IRLS(
        maxIRLSiter=20, minGNiter=1, fix_Jmatrix=True, coolingRate=2, betaSearch=False,
        chifact_start = 1.
    )
    opt = Optimization.ProjectedGNCG(maxIter = 25)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    beta = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    target = Directives.TargetMisfit()
    # update_sense = Directives.UpdateSensitivityWeights(threshold=delta)
    # inv = Inversion.BaseInversion(invProb, directiveList=[IRLS, betaest])
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
    prob.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    #Run
    mopt = inv.run(m0)
    return mopt


def plot_results(mopt,expmap,res_model_app):
    f,ax = plot_mesh(res_model_app.result, mesh1D)

    sig = expmap*mopt
    # print(sig)
    # Utils1D.plotLayer(expmap*mopt, mesh1D, showlayers=True, **{'color':'r'},ax=ax)
    z_grid = mesh1D.vectorNx
    n_sig = sig.size
    sigma = 1./np.repeat(sig, 2)
    z = []
    for i in range(n_sig):
        z.append(np.r_[z_grid[i], z_grid[i+1]])
    z = np.hstack(z)
    ax.plot(sigma, z, 'r-',label='Estimated resistivity')
    ax.legend(bbox_to_anchor=(1.05, 0.3) )
    return f,ax


def interactive_inversion(res_model_app):

    button = Button(description="Run inversion!")
    output = Output()
    display(button, output)

    def on_button_clicked(b):
        print('Running inversion...')
        with output:
            clear_output(True)
            mopt = prob_and_run(res_model_app)
            f,ax = plot_results(mopt,expmap,res_model_app)
            show_inline_matplotlib_plots()
        return
    button.on_click(on_button_clicked)

    show_inline_matplotlib_plots()
    return button,output



# button = ipywidgets.Button(description="Button")
# out = ipywidgets.Output()

# def on_button_clicked(b):
#     with out:
#         clear_output(True)
#         plot([1,2],[2,1])
#         show()

# button.on_click(on_button_clicked)

# display(button)

# with out:
#     plot([1,2],[1,2])
#     show()

# out





# @blockPrinting


if __name__ == "__main__":
    plot_res_model()
