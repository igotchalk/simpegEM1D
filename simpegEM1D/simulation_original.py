from SimPEG import maps, utils, props
from SimPEG.simulation import BaseSimulation
import numpy as np
from .survey_original import BaseEM1DSurvey
from .supporting_functions.kernels import *
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
import properties

from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel




class BaseEM1DSimulation(BaseSimulation):
    """
    Pseudo analytic solutions for frequency and time domain EM problems
    assumingLayered earth (1D).
    """
    surveyPair = BaseEM1DSurvey
    mapPair = maps.IdentityMap
    chi = None
    hankel_filter = 'key_101_2009'  # Default: Hankel filter
    hankel_pts_per_dec = None       # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None

    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(sigma, rho)

    chi = props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = props.Invertible(
        "Electrical chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = props.Invertible(
        "Frequency Dependency, 0 < c < 1",
        default=0.5
    )

    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    survey = properties.Instance(
        "a survey object", BaseEM1DSurvey, required=True
    )

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).

        ht, htarg = check_hankel(
            'dlf',
            {
                'dlf': self.hankel_filter,
                'pts_per_dec': 0
            },
            1
        )

        self.fhtfilt = htarg['dlf']                 # Store filter
        self.hankel_pts_per_dec = htarg['pts_per_dec']      # Store pts_per_dec
        if self.verbose:
            print(">> Use "+self.hankel_filter+" filter for Hankel Transform")

        # if self.hankel_pts_per_dec != 0:
        #     raise NotImplementedError()

    # make it as a property?

    def sigma_cole(self):
        """
        Computes Pelton's Cole-Cole conductivity model
        in frequency domain.

        Parameter
        ---------

        n_filter: int
            the number of filter values
        f: ndarray
            frequency (Hz)

        Return
        ------

        sigma_complex: ndarray (n_layer x n_frequency x n_filter)
            Cole-Cole conductivity values at given frequencies

        """
        n_layer = self.survey.n_layer
        n_frequency = self.survey.n_frequency
        n_filter = self.n_filter
        f = self.survey.frequency

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))
        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(
            2*np.pi*f,
            (n_layer, 1)
        )

        sigma_complex = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order='F'
        )
        sigma_complex[:, :] = (
            sigma -
            sigma*eta/(1+(1-eta)*(1j*w*tau)**c)
        )

        sigma_complex_tensor = np.empty(
            [n_layer, n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        sigma_complex_tensor[:, :, :] = np.tile(sigma_complex.reshape(
            (n_layer, n_frequency, 1)), (1, 1, n_filter)
        )

        return sigma_complex_tensor

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    def forward(self, m, output_type='response'):
        """
            Return Bz or dBzdt
        """

        self.model = m

        n_frequency = self.survey.n_frequency
        flag = self.survey.field_type
        n_layer = self.survey.n_layer
        depth = self.survey.depth
        I = self.survey.I
        n_filter = self.n_filter

        # Get lambd and offset, will depend on pts_per_dec
        if self.survey.src_type == "VMD":
            r = self.survey.offset
        else:
            # a is the radius of the loop
            r = self.survey.a * np.ones(n_frequency)

        # Use function from empymod
        # size of lambd is (n_frequency x n_filter)
        lambd = np.empty([self.survey.frequency.size, n_filter], order='F')
        lambd[:, :], _ = get_dlf_points(
            self.fhtfilt, r, self.hankel_pts_per_dec
        )

        # TODO: potentially store
        f = np.empty([self.survey.frequency.size, n_filter], order='F')
        f[:, :] = np.tile(
            self.survey.frequency.reshape([-1, 1]), (1, n_filter)
        )
        # h is an inversion parameter
        if self.hMap is not None:
            h = self.h
        else:
            h = self.survey.h

        z = h + self.survey.dz

        chi = self.chi

        if np.isscalar(self.chi):
            chi = np.ones_like(self.sigma) * self.chi

        # TODO: potentially store
        sig = self.sigma_cole()

        if output_type == 'response':
            # for simulation
            if self.survey.src_type == 'VMD':
                hz = hz_kernel_vertical_magnetic_dipole(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':
                hz = hz_kernel_circular_loop(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (None, hz, None)  # PJ1

            # TODO: This has not implemented yet!
            elif self.survey.src_type == "piecewise_line":
                # Need to compute y
                hz = hz_kernel_horizontal_electric_dipole(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )
                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        elif output_type == 'sensitivity_sigma':

            # for simulation
            if self.survey.src_type == 'VMD':
                hz = hz_kernel_vertical_magnetic_dipole(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':

                hz = hz_kernel_circular_loop(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

            r = np.tile(r, (n_layer, 1))

        elif output_type == 'sensitivity_height':

            # for simulation
            if self.survey.src_type == 'VMD':
                hz = hz_kernel_vertical_magnetic_dipole(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':

                hz = hz_kernel_circular_loop(
                    self, lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        # Carry out Hankel DLF
        # ab=66 => 33 (vertical magnetic src and rec)
        # For response
        # HzFHT size = (n_frequency,)
        # For sensitivity
        # HzFHT size = (n_layer, n_frequency)

        HzFHT = dlf(PJ, lambd, r, self.fhtfilt, self.hankel_pts_per_dec,
                    ang_fact=None, ab=33)

        if output_type == "sensitivity_sigma":
            return HzFHT.T

        return HzFHT

    # @profile
    def fields(self, m):
        f = self.forward(m, output_type='response')
        # self.survey._pred = utils.mkvc(self.survey.projectFields(f))
        return f

    def getJ_height(self, m, f=None):
        """

        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        else:

            if self.verbose:
                print(">> Compute J height ")

            dudz = self.forward(m, output_type="sensitivity_height")

            self._Jmatrix_height = (
                self.projectFields(dudz)
            ).reshape([-1, 1])

            return self._Jmatrix_height

    # @profile
    def getJ_sigma(self, m, f=None):

        if self.sigmaMap is None:
            return utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print(">> Compute J sigma")

            dudsig = self.forward(m, output_type="sensitivity_sigma")

            self._Jmatrix_sigma = self.projectFields(dudsig)
            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma

    def getJ(self, m, f=None):
        return (
            self.getJ_sigma(m, f=f) * self.sigmaDeriv +
            self.getJ_height(m, f=f) * self.hDeriv
        )

    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        if self.hMap is not None:
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jtv = self.sigmaDeriv.T*np.dot(J_sigma.T, v)
        if self.hMap is not None:
            Jtv += self.hDeriv.T*np.dot(J_height.T, v)
        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.survey.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = (np.power((utils.sdiag(1./uncert)*J), 2)).sum(axis=0)
        return JtJdiag


    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Here we do not store predicted data
            because projection (`d = P(f)`) is cheap.
        """

        if f is None:
            f = self.fields(m)
        return utils.mkvc(self.projectFields(f))



class EM1DFMSimulation(BaseEM1DSimulation):

    def __init__(self, mesh, **kwargs):
        BaseEM1DSimulation.__init__(self, mesh, **kwargs)


    @property
    def hz_primary(self):
        # Assumes HCP only at the moment
        if self.survey.src_type == 'VMD':
            return -1./(4*np.pi*self.survey.offset**3)
        elif self.survey.src_type == 'CircularLoop':
            return self.I/(2*self.survey.a) * np.ones_like(self.survey.frequency)
        else:
            raise NotImplementedError()

    
    def projectFields(self, u):
        """
            Decompose frequency domain EM responses as real and imaginary
            components
        """ 

        ureal = (u.real).copy()
        uimag = (u.imag).copy()

        if self.survey.rx_type == 'Hz':
            factor = 1.
        elif self.survey.rx_type == 'ppm':
            factor = 1./self.hz_primary * 1e6

        if self.survey.switch_real_imag == 'all':
            ureal = (u.real).copy()
            uimag = (u.imag).copy()
            if ureal.ndim == 1 or 0:
                resp = np.r_[ureal*factor, uimag*factor]
            elif ureal.ndim == 2:
                if np.isscalar(factor):
                    resp = np.vstack(
                            (factor*ureal, factor*uimag)
                    )
                else:
                    resp = np.vstack(
                        (utils.sdiag(factor)*ureal, utils.sdiag(factor)*uimag)
                    )
            else:
                raise NotImplementedError()
        elif self.survey.switch_real_imag == 'real':
            resp = (u.real).copy()
        elif self.survey.switch_real_imag == 'imag':
            resp = (u.imag).copy()
        else:
            raise NotImplementedError()

        return resp






class EM1DTMSimulation(BaseEM1DSimulation):





    def __init__(self, mesh, **kwargs):
        BaseEM1DSimulation.__init__(self, mesh, **kwargs)


    def projectFields(self, u):
        """
            Transform frequency domain responses to time domain responses
        """
        # Compute frequency domain reponses right at filter coefficient values
        # Src waveform: Step-off

        if self.survey.use_lowpass_filter:
            factor = self.survey.lowpass_filter.copy()
        else:
            factor = np.ones_like(self.survey.frequency, dtype=complex)

        if self.survey.rx_type == 'Bz':
            factor *= 1./(2j*np.pi*self.survey.frequency)

        if self.survey.wave_type == 'stepoff':
            # Compute EM responses
            if u.size == self.survey.n_frequency:
                resp, _ = fourier_dlf(
                    u.flatten()*factor, self.survey.time,
                    self.survey.frequency, self.survey.ftarg
                )
            # Compute EM sensitivities
            else:
                resp = np.zeros(
                    (self.survey.n_time, self.survey.n_layer), dtype=np.float64, order='F')
                # )
                # TODO: remove for loop
                for i in range(self.survey.n_layer):
                    resp_i, _ = fourier_dlf(
                        u[:, i]*factor, self.survey.time,
                        self.survey.frequency, self.survey.ftarg
                    )
                    resp[:, i] = resp_i

        # Evaluate piecewise linear input current waveforms
        # Using Fittermann's approach (19XX) with Gaussian Quadrature
        elif self.survey.wave_type == 'general':
            # Compute EM responses
            if u.size == self.survey.n_frequency:
                resp_int, _ = fourier_dlf(
                    u.flatten()*factor, self.survey.time_int,
                    self.survey.frequency, self.survey.ftarg
                )
                # step_func = interp1d(
                #     self.time_int, resp_int
                # )
                step_func = iuSpline(
                    np.log10(self.survey.time_int), resp_int
                )

                resp = piecewise_pulse_fast(
                    step_func, self.survey.time,
                    self.survey.time_input_currents,
                    self.survey.input_currents,
                    self.survey.period,
                    n_pulse=self.survey.n_pulse
                )

                # Compute response for the dual moment
                if self.survey.moment_type == "dual":
                    resp_dual_moment = piecewise_pulse_fast(
                        step_func, self.survey.time_dual_moment,
                        self.survey.time_input_currents_dual_moment,
                        self.survey.input_currents_dual_moment,
                        self.survey.period_dual_moment,
                        n_pulse=self.survey.n_pulse
                    )
                    # concatenate dual moment response
                    # so, ordering is the first moment data
                    # then the second moment data.
                    resp = np.r_[resp, resp_dual_moment]

            # Compute EM sensitivities
            else:
                if self.survey.moment_type == "single":
                    resp = np.zeros(
                        (self.survey.n_time, self.survey.n_layer),
                        dtype=np.float64, order='F'
                    )
                else:
                    # For dual moment
                    resp = np.zeros(
                        (self.survey.n_time+self.survey.n_time_dual_moment, self.survey.n_layer),
                        dtype=np.float64, order='F')

                # TODO: remove for loop (?)
                for i in range(self.survey.n_layer):
                    resp_int_i, _ = fourier_dlf(
                        u[:, i]*factor, self.survey.time_int,
                        self.survey.frequency, self.survey.ftarg
                    )
                    # step_func = interp1d(
                    #     self.time_int, resp_int_i
                    # )

                    step_func = iuSpline(
                        np.log10(self.survey.time_int), resp_int_i
                    )

                    resp_i = piecewise_pulse_fast(
                        step_func, self.survey.time,
                        self.survey.time_input_currents, self.survey.input_currents,
                        self.survey.period, n_pulse=self.survey.n_pulse
                    )

                    if self.survey.moment_type == "single":
                        resp[:, i] = resp_i
                    else:
                        resp_dual_moment_i = piecewise_pulse_fast(
                            step_func,
                            self.survey.time_dual_moment,
                            self.survey.time_input_currents_dual_moment,
                            self.survey.input_currents_dual_moment,
                            self.survey.period_dual_moment,
                            n_pulse=self.survey.n_pulse
                        )
                        resp[:, i] = np.r_[resp_i, resp_dual_moment_i]
        return resp * (-2.0/np.pi) * mu_0


    # def dpred(self, m, f=None):
    #     """
    #         Computes predicted data.
    #         Predicted data (`_pred`) are computed and stored
    #         when self.prob.fields(m) is called.
    #     """
    #     if f is None:
    #         f = self.fields(m)

    #     return f




if __name__ == '__main__':
    main()