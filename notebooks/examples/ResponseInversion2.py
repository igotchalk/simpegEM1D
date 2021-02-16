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
import ipywidgets as widgets


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

def define_sigma_layers(sig_lay,layers):
    '''
    layers  (n,): tops of layers, negative values indicate below surface
    sig_lay (n,): conductivity of each layer, same size as layers. Final value considered to be conductivity of background '''
    sig  = np.ones(TDsurvey.n_layer)*1.0
    for i in range(len(layers)):
        if i==len(layers)-1:
            ind = (layers[i]>LocSigZ)
        else:
            ind = np.where((layers[i]>LocSigZ) & (layers[i+1]<=LocSigZ))
        sig[ind] = sig_lay[i]
#             print(sig)
    return sig 

def plot_mesh(sig,mesh1D,ax=None):
    import matplotlib.patheffects as pe
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5, 8))
    Utils1D.plotLayer(1./sig, mesh1D, showlayers=True,xlim=(rhomin*.9,rhomax*1.1),label='True resistivity',ax=ax)
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
    return ax

def define_sigma_layers_from_rho(rhos,lays):
    layers = -np.asarray(lays)
    sig_lay=1./np.asarray(rhos)
    sig = define_sigma_layers(sig_lay,layers)
    return sig

def plot_res_model(rhos,lays,ax=None):
    sig = define_sigma_layers_from_rho(rhos,lays)
    ax = plot_mesh(sig,mesh1D,ax=ax)
    plt.show()
    return sig, ax

def create_table():
    align_kw = {'justify-content':'flex-start'}
    def new_rho_logfloatslider(lay_name='1',value=rho_default):
        return widgets.FloatLogSlider(
                base=10,
                min=np.log10(rhomin),
                max=np.log10(rhomax),
                value=value,
                step=0.1,
                continuous_update=False,
                description=r"$\rho_{" + str(lay_name) + "}$",
                readout=False,
                layout=widgets.Layout(width='250px'),
                **align_kw
            )

    def insert_rho_sliders(rho_sliders,col=0,row_start=0):
        column1 = ipysheet.column(col, rho_sliders,row_start=row_start)
    #     for i,slide in enumerate(rho_sliders):
    #         cell(row_start+i,col,slide)
        cells = []
        for i in range(len(rho_sliders)):
            cells.append(ipysheet.cell(row_start+i,col+1,rho_sliders[i].value,numeric_format='0.0'))
        for c,s in zip(cells,rho_sliders):
            widgets.jslink((c, "value"),(s,"value"))
        return cells

    #Create depth values
    def new_thk_cell(row,col=2,thk=30):
        if thk is None:
            '''
            dep: set to previous layer
            '''
        return ipysheet.cell(row,col,thk,numeric_format='0')

    #Create depth values
    def new_dep_cell(row,col=3,cell_thk=None):
        c=ipysheet.cell(row,col,value=0,
                    color='black',
                    background_color='grey',
                    numeric_format='0',
                    read_only=True)
        @ipysheet.calculation(inputs=cell_thk, output=c)
        def add(*a):
            return sum(a)
        return c
    sheet = ipysheet.sheet(rows=4,columns=4,column_width=[100,25,25,25],
                           column_headers=['Resistivity slider','Resistivity value',
                                           r'Thickness (m)','Top of layer depth (m)'],
                          )
#     #Set background values
#     ipysheet.row(0, ['--','--'],column_start=2,column_end=3,
#                  read_only=True,color='black', background_color='grey')

    #Thickness col
    cell_thk = []
    cell_dep = []
    for i in range(0,4):
        cell_thk.append(new_thk_cell(row=i,col=2,thk=30))
        cell_dep.append(new_dep_cell(row=i,col=3,cell_thk=cell_thk[:-1]))
    #     @ipysheet.calculation(inputs=cell_thk[:-1], output=cell_dep[-1])
    #     def add(*a):
    #         return sum(a)

    #Create rho sliders and depth cells
    cells_rho = []
    rho_sliders = []
    rho_init_values=[200,50,25,50]
#     rho_sliders.append(new_rho_logfloatslider('background',rho_default*10))
    for i in range(4):
        rho_sliders.append(new_rho_logfloatslider(str(i+1),value=rho_init_values[i]))
    #     cells.append(new_depfrom_cell(row=i,col=2,dep=10))
    cells_rho.extend(insert_rho_sliders(rho_sliders))

    #Create button
    button = widgets.Button(description='Add Row')
    out = widgets.Output()
    def add_row(_):
        with out:
            sheet.rows += 1
            rho_sliders.append(new_rho_logfloatslider(sheet.rows))
            cells_rho.extend(insert_rho_sliders((rho_sliders[-1],),row_start=sheet.rows-1))
            cell_thk.append(new_thk_cell(row=sheet.rows-1,col=2,thk=30))
            cell_dep.append(new_dep_cell(row=sheet.rows-1,col=3,cell_thk=cell_thk[:-1]))
            [c.observe(update_plot,'value') for c in rho_sliders]
            [c.observe(update_plot,'value') for c in cell_dep]
    button.on_click(add_row)
#     f,ax = plt.subplots(1,figsize=(4,8))
    
    

    out = widgets.Output(layout={'height':'600px',
                                   'width': '600px'})
    with out:
        sig,_ = plot_res_model([r.value for r in rho_sliders],[d.value for d in cell_dep])
        show_inline_matplotlib_plots()

    @out.capture()
    def update_plot(change):
        clear_output(True)
#         output.clear_output(wait=True)
        sig,_ = plot_res_model([r.value for r in rho_sliders],[d.value for d in cell_dep])
        show_inline_matplotlib_plots()

#     with output:
#         update_plot()

    [c.observe(update_plot,'value') for c in rho_sliders]
    [c.observe(update_plot,'value') for c in cell_dep]
    
    vbox=widgets.VBox([sheet,button,out])
    display(vbox)
    
#     rho_sliders[0].value=rho_default
#     sig,ax= plot_res_model([r.value for r in rho_sliders],[d.value for d in cell_dep],ax=ax)
#     show_inline_matplotlib_plots()
    return rho_sliders,cell_dep




def create_prob(rho_sliders,cell_dep):
    '''
    w: ipywidget containing values for model (in conductivity), rho1,rho2,lay1,lay2
    '''
    #Problem
    sig = define_sigma_layers_from_rho([r.value for r in rho_sliders],[d.value for d in cell_dep])
    
    m_true = np.log(sig)
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

def interactive_fwd(rho_sliders,dep_cells):

    button = Button(description="Acquire AEM data!")
    output = Output()
    display(button, output)

    def on_button_clicked(b):
        with output:
            clear_output(True)
            m_true,d_true,prob,TDsurvey = create_prob(rho_sliders,dep_cells)
            plot_data(d_true,TDsurvey)
            show_inline_matplotlib_plots()
        return
    button.on_click(on_button_clicked)

    show_inline_matplotlib_plots()
    return button,output



def prob_and_run(rho_sliders,dep_cells):
    sig = define_sigma_layers_from_rho([r.value for r in rho_sliders],[d.value for d in dep_cells])
    m_true,d_true,prob,TDsurvey = create_prob(rho_sliders,dep_cells)
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
    m0 = np.log(np.ones_like(sig)* (1./rho_sliders[-1].value))
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


def plot_results(mopt,expmap,rho_sliders,dep_cells):
    sig = define_sigma_layers_from_rho([r.value for r in rho_sliders],[d.value for d in dep_cells])
    ax = plot_mesh(sig, mesh1D)

    sig = expmap*mopt
    z_grid = mesh1D.vectorNx
    n_sig = sig.size
    sigma = 1./np.repeat(sig, 2)
    z = []
    for i in range(n_sig):
        z.append(np.r_[z_grid[i], z_grid[i+1]])
    z = np.hstack(z)
    ax.plot(sigma, z, 'r-',label='Estimated resistivity')
    ax.legend(bbox_to_anchor=(1.05, 0.3) )
    return ax


def interactive_inversion(rho_sliders,dep_cells):

    button = Button(description="Run inversion!")
    output = Output()
    display(button, output)

    def on_button_clicked(b):
        print('Running inversion...')
        with output:
            clear_output(True)
            mopt = prob_and_run(rho_sliders,dep_cells)
            ax = plot_results(mopt,expmap,rho_sliders,dep_cells)
            show_inline_matplotlib_plots()
        return
    button.on_click(on_button_clicked)

    show_inline_matplotlib_plots()
    return button,output
