from CoolProp.CoolProp import PropsSI, PhaseSI
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from system import Component, psd


class FVMBCouplingAlg:

    """
    Z. Chu et al., "Moving-boundary and finite volume coupling algorithm for heat
    exchanger with fluid phase changen", International Journal of Heat and Mass Transfer, 2014
    """

    N = 5  # number of finite volumes per phase
    A_c = None  # heat transfer area hot side
    A_h = None  # heat transfer area cold side
    t = 0.003  # wall thickness [m]
    L = 0.5  # heat excahnge Length [m]
    Dh = 0.000909633  # hydraulic diameter [m]
    n = 22  # number of channels
    S = 0.000227615  # Cross sectional area per channel [m2]
    k = 16.7  # heat conductivity [W/mK]

    fluid_list = ['Water', 'Air', 'R134a', 'R744']

    def __init__(self, Fluid_h, p_hi, h_hi, m_h, Fluid_c, p_ci, h_ci, m_c):

        self.Fluid_h = Fluid_h
        self.p_hi = p_hi
        self.h_hi = h_hi
        self.m_h = m_h
        self.Fluid_c = Fluid_c
        self.p_ci = p_ci
        self.h_ci = h_ci
        self.m_c = m_c

        # Determine the inlet temperatures from the pressure/enthalpy pairs
        self.T_ci = PropsSI('T', 'P', self.p_ci, 'H', self.h_ci, self.Fluid_c)
        self.T_hi = PropsSI('T', 'P', self.p_hi, 'H', self.h_hi, self.Fluid_h)

        # Calculate the bubble and dew enthalpies for each stream
        self.T_cbubble = PropsSI('T', 'P', self.p_ci, 'Q', 0, self.Fluid_c)
        self.T_cdew = PropsSI('T', 'P', self.p_ci, 'Q', 1, self.Fluid_c)
        self.T_hbubble = PropsSI('T', 'P', self.p_hi, 'Q', 0, self.Fluid_h)
        self.T_hdew = PropsSI('T', 'P', self.p_hi, 'Q', 1, self.Fluid_h)
        self.h_cbubble = PropsSI('H', 'T', self.T_cbubble, 'Q', 0, self.Fluid_c)
        self.h_cdew = PropsSI('H', 'T', self.T_cdew, 'Q', 1, self.Fluid_c)
        self.h_hbubble = PropsSI('H', 'T', self.T_hbubble, 'Q', 0, self.Fluid_h)
        self.h_hdew = PropsSI('H', 'T', self.T_hdew, 'Q', 1, self.Fluid_h)
        self.T_cmax = PropsSI("TMAX", self.Fluid_c)
        self.P_cmax = PropsSI('PMAX', self.Fluid_c)
        self.T_hmin = PropsSI("TMIN", self.Fluid_c)
        self.P_hmin = PropsSI('PMIN', self.Fluid_c)

    def external_pinching(self):
        """ Determine the maximum heat transfer rate based on the external pinching analysis """

        if self.T_hmin < self.T_ci:
            if abs(self.T_ci - PropsSI('T', 'P', self.p_hi, 'Q', 1.0, self.Fluid_h)) < 1e-4:
                self.h_ho = PropsSI('H', 'P', self.p_hi, 'Q', 1.0, self.Fluid_h)
            else:
                self.h_ho = PropsSI('H', 'T', self.T_ci, 'P', self.p_hi, self.Fluid_h)
        else:
            self.h_ho = PropsSI('H', 'T', self.T_hmin, 'P', self.p_hi, self.Fluid_h)

        Qmaxh = self.m_h * (self.h_hi - self.h_ho)

        if self.T_cmax > self.T_hi:
            if abs(self.T_hi - PropsSI('T', 'P', self.p_hi, 'Q', 1.0, self.Fluid_h)) < 1e-4:
                self.h_co = PropsSI('H', 'P', self.p_ci, 'Q', 1.0, self.Fluid_c)
            else:
                self.h_co = PropsSI('H', 'T', self.T_hi, 'P', self.p_ci, self.Fluid_c)
        else:
            self.h_co = PropsSI('H', 'T', self.T_cmax, 'P', self.p_ci, self.Fluid_c)

        Qmaxc = self.m_c * (self.h_co - self.h_ci)

        Qmax = min(Qmaxh, Qmaxc)

        self.calculate_cell_boundaries(Qmax)

        return Qmax

    def internal_pinching(self, stream):
        """
        Determine the maximum heat transfer rate based on the internal pinching analysis
        """

        if stream == 'hot':

            # Try to find the dew point enthalpy as one of the cell boundaries
            # that is not the inlet or outlet

            # Check for the hot stream pinch point
            for i in range(1, len(self.hvec_h) - 1):

                # Check if enthalpy is equal to the dewpoint enthalpy of hot
                # stream and hot stream is colder than cold stream (impossible)
                if (abs(self.hvec_h[i] - self.h_hdew) < 1e-6
                        and self.Tvec_c[i] > self.Tvec_h[i]):
                    # Enthalpy of the cold stream at the pinch temperature
                    h_c_pinch = PropsSI('H', 'T', self.T_hdew, 'P', self.p_ci, self.Fluid_c)

                    # Heat transfer in the cell
                    Qright = self.m_h * (self.h_hi - self.h_hdew)

                    # New value for the limiting heat transfer rate
                    Qmax = self.m_c * (h_c_pinch - self.h_ci) + Qright

                    # Recalculate the cell boundaries
                    self.calculate_cell_boundaries(Qmax)

                    return Qmax

        elif stream == 'cold':
            # Check for the cold stream pinch point
            for i in range(1, len(self.hvec_c) - 1):

                # Check if enthalpy is equal to the bubblepoint enthalpy of cold
                # stream and hot stream is colder than cold stream (impossible)
                if (abs(self.hvec_c[i] - self.h_cbubble) < 1e-6
                        and self.Tvec_c[i] > self.Tvec_h[i]):
                    # Enthalpy of the cold stream at the pinch temperature
                    h_h_pinch = PropsSI('H', 'T', self.T_cbubble, 'P', self.p_hi, self.Fluid_h)

                    # Heat transfer in the cell
                    Qleft = self.m_c * (self.h_cbubble - self.h_ci)

                    # New value for the limiting heat transfer rate
                    Qmax = Qleft + self.m_h * (self.h_hi - h_h_pinch)

                    # Recalculate the cell boundaries
                    self.calculate_cell_boundaries(Qmax)

                    return Qmax
        else:
            raise ValueError

    def calculate_cell_boundaries(self, Q):
        """ Calculate the cell boundaries for each fluid """

        # Re-calculate the outlet enthalpies of each stream
        self.h_co = self.h_ci + Q / self.m_c
        self.h_ho = self.h_hi - Q / self.m_h

        # Start with the external boundaries (sorted in increasing enthalpy)
        self.hvec_c = [self.h_ci, self.h_co]
        self.hvec_h = [self.h_ho, self.h_hi]

        # Add the bubble and dew enthalpies for the hot stream
        if self.h_hdew is not None and self.h_hi > self.h_hdew > self.h_ho:
            self.hvec_h.insert(-1, self.h_hdew)
        if self.h_hbubble is not None and self.h_hi > self.h_hbubble > self.h_ho:
            self.hvec_h.insert(1, self.h_hbubble)

        # Add the bubble and dew enthalpies for the cold stream
        if self.h_cdew is not None and self.h_ci < self.h_cdew < self.h_co:
            self.hvec_c.insert(-1, self.h_cdew)
        if self.h_cbubble is not None and self.h_ci < self.h_cbubble < self.h_co:
            self.hvec_c.insert(1, self.h_cbubble)

        # Fill in the complementary cell boundaries
        # Start at the first element in the vector
        k = 0
        while k < len(self.hvec_c) - 1 or k < len(self.hvec_h) - 1:
            if len(self.hvec_c) == 2 and len(self.hvec_h) == 2:
                break

            # Determine which stream is the limiting next cell boundary
            Qcell_hk = self.m_h * (self.hvec_h[k + 1] - self.hvec_h[k])
            Qcell_ck = self.m_c * (self.hvec_c[k + 1] - self.hvec_c[k])

            if abs(Qcell_hk / Qcell_ck - 1) < 1e-6:
                k += 1
                break
            elif Qcell_hk > Qcell_ck:
                # Hot stream needs a complementary cell boundary
                self.hvec_h.insert(k + 1, self.hvec_h[k] + Qcell_ck / self.m_h)
            else:
                # Cold stream needs a complementary cell boundary
                self.hvec_c.insert(k + 1, self.hvec_c[k] + Qcell_hk / self.m_c)

            Qcell_hk = self.m_h * (self.hvec_h[k + 1] - self.hvec_h[k])
            Qcell_ck = self.m_c * (self.hvec_c[k + 1] - self.hvec_c[k])
            assert (abs(Qcell_hk / Qcell_ck - 1) < 1e-6)

            # Increment index
            k += 1

        assert (len(self.hvec_h) == len(self.hvec_c))
        Qhs = np.array([self.m_h * (self.hvec_h[i + 1] - self.hvec_h[i]) for i in range(len(self.hvec_h) - 1)])
        Qcs = np.array([self.m_c * (self.hvec_c[i + 1] - self.hvec_c[i]) for i in range(len(self.hvec_c) - 1)])

        # Calculate the temperature and entropy at each cell boundary
        self.Tvec_c = PropsSI('T', 'H', self.hvec_c, 'P', self.p_ci, self.Fluid_c)
        self.Tvec_h = PropsSI('T', 'H', self.hvec_h, 'P', self.p_hi, self.Fluid_h)
        self.svec_c = PropsSI('S', 'H', self.hvec_c, 'P', self.p_ci, self.Fluid_c)
        self.svec_h = PropsSI('S', 'H', self.hvec_h, 'P', self.p_hi, self.Fluid_h)

        # Calculate the phase in each cell
        self.phases_h = []
        for i in range(len(self.hvec_h) - 1):
            havg = (self.hvec_h[i] + self.hvec_h[i + 1]) / 2.0
            if havg < self.h_hbubble:
                self.phases_h.append('liquid')
            elif havg > self.h_hdew:
                self.phases_h.append('vapor')
            else:
                self.phases_h.append('two-phase')

        self.phases_c = []
        for i in range(len(self.hvec_c) - 1):
            havg = (self.hvec_c[i] + self.hvec_c[i + 1]) / 2.0
            if havg < self.h_cbubble:
                self.phases_c.append('liquid')
            elif havg > self.h_cdew:
                self.phases_c.append('vapor')
            else:
                self.phases_c.append('two-phase')

    def objective_function(self, Q):

        self.calculate_cell_boundaries(Q)
        self.Areq = np.zeros([len(self.hvec_c) - 1])

        for i in range(0, len(self.hvec_c) - 1):
            Qcell = self.m_h * (self.hvec_h[i + 1] - self.hvec_h[i])
            self.Areq[i] = self.fvm_cell_solve_2(Qcell, self.hvec_h[i + 1], self.hvec_c[i])
        return 1 - sum(self.Areq) / self.A_h

    def fvm_cell_solve(self, Q, h_hi, h_ci):

        self.h_cell_vec_h = [h_hi - j * Q / (self.N * self.m_h) for j in range(0, self.N + 1)]
        self.T_cell_vec_h = PropsSI('T', 'H', self.h_cell_vec_h, 'P', self.p_hi, self.Fluid_h)
        self.h_cell_vec_c = [h_ci + j * Q / (self.N * self.m_c) for j in range(0, self.N + 1)]
        self.h_cell_vec_c.reverse()
        self.T_cell_vec_c = PropsSI('T', 'H', self.h_cell_vec_c, 'P', self.p_ci, self.Fluid_c)

        self.h_cell_vec_h_c = [(self.h_cell_vec_h[k] + self.h_cell_vec_h[k + 1]) / 2 for k in range(0, self.N)]
        self.h_cell_vec_c_c = [(self.h_cell_vec_c[k] + self.h_cell_vec_c[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_h_c = [(self.T_cell_vec_h[k] + self.T_cell_vec_h[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_c_c = [(self.T_cell_vec_c[k] + self.T_cell_vec_c[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_w = [(self.T_cell_vec_h_c[k] + self.T_cell_vec_c_c[k]) / 2 for k in range(0, self.N)]

        R = self.A_h / self.A_c
        while True:
            A_req_h = 0
            stopflag = 0
            Tw_old = self.T_cell_vec_w.copy()
            for j in range(0, self.N):
                alpha_h = self.alpha_correlation(self.p_hi, self.h_cell_vec_h_c[j], self.T_cell_vec_h_c[j],
                                                 self.T_cell_vec_w[j], self.m_h, self.Fluid_h)
                alpha_c = self.alpha_correlation(self.p_ci, self.h_cell_vec_c_c[j], self.T_cell_vec_c_c[j],
                                                 self.T_cell_vec_w[j], self.m_c, self.Fluid_c)
                self.T_cell_vec_w[j] = (R * alpha_h * self.T_cell_vec_h_c[j] + alpha_c * self.T_cell_vec_c_c[j]) / (
                            R * alpha_h + alpha_c)
                if abs(self.T_cell_vec_w[j] - Tw_old[j]) < 0.1:
                    A_req_h += Q / self.N / (alpha_c * (self.T_cell_vec_w[j] - self.T_cell_vec_c_c[j]))
                    stopflag += 1
            if stopflag == self.N:
                break

        return A_req_h

    def fvm_cell_solve_2(self, Q, h_hi, h_ci):

        self.h_cell_vec_h = [h_hi - j * Q / (self.N * self.m_h) for j in range(0, self.N + 1)]
        self.T_cell_vec_h = PropsSI('T', 'H', self.h_cell_vec_h, 'P', self.p_hi, self.Fluid_h)
        self.h_cell_vec_c = [h_ci + j * Q / (self.N * self.m_c) for j in range(0, self.N + 1)]
        self.h_cell_vec_c.reverse()
        self.T_cell_vec_c = PropsSI('T', 'H', self.h_cell_vec_c, 'P', self.p_ci, self.Fluid_c)

        self.h_cell_vec_h_c = [(self.h_cell_vec_h[k] + self.h_cell_vec_h[k + 1]) / 2 for k in range(0, self.N)]
        self.h_cell_vec_c_c = [(self.h_cell_vec_c[k] + self.h_cell_vec_c[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_h_c = [(self.T_cell_vec_h[k] + self.T_cell_vec_h[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_c_c = [(self.T_cell_vec_c[k] + self.T_cell_vec_c[k + 1]) / 2 for k in range(0, self.N)]
        self.T_cell_vec_w = [(self.T_cell_vec_h_c[k] + self.T_cell_vec_c_c[k]) / 2 for k in range(0, self.N)]

        R = 1
        Areq_h = 0
        for j in range(0, self.N):
            alpha_h = self.alpha_correlation(self.p_hi, self.h_cell_vec_h_c[j], self.T_cell_vec_h_c[j],
                                             self.T_cell_vec_w[j], self.m_h, self.Fluid_h)
            alpha_c = self.alpha_correlation(self.p_ci, self.h_cell_vec_c_c[j], self.T_cell_vec_c_c[j],
                                             self.T_cell_vec_w[j], self.m_c, self.Fluid_c)
            self.T_cell_vec_w[j] = (R * alpha_h * self.T_cell_vec_h_c[j] + alpha_c * self.T_cell_vec_c_c[j]) / (
                        R * alpha_h + alpha_c)
            Areq_h += Q / self.N / (alpha_c * (self.T_cell_vec_w[j] - self.T_cell_vec_c_c[j]))

        return Areq_h

    def run(self):

        """starts solving algorithm"""

        # Check the external pinching & update cell boundaries
        Qmax_ext = self.external_pinching()
        Qmax = Qmax_ext

        # Check the internal pinching
        for stream in ['hot', 'cold']:
            # Check stream internal pinching & update cell boundaries
            Qmax_int = self.internal_pinching(stream)
            if Qmax_int is not None:
                Qmax = Qmax_int

        self.Qmax = Qmax

        self.Q = self.solve()

        self.pressure_solve()

        # self.fvm_cells_solve()

    def solve(self):
        """

        solve the objective function using Brent's method and the maximum heat transfer
        rate calculated from the pinching analysis

        """
        if self.objective_function(self.Qmax) > 0:
            return self.Qmax
        else:
            self.Q = scipy.optimize.brentq(self.objective_function, 1e-5, self.Qmax - 1e-10, rtol=1e-14, xtol=1e-10)
            return self.Q

    def fvm_cells_solve(self):

        """solves for all boundary state values for each phase"""

        self.h_cell_vec_h = []
        self.h_cell_vec_c = []
        self.T_cell_vec_h = []
        self.T_cell_vec_c = []
        for i in range(0, len(self.hvec_h) - 1):
            Q_cell = self.m_h * (self.hvec_h[i + 1] - self.hvec_h[i])
            for j in range(0, self.N + 1):
                self.h_cell_vec_h.append(self.hvec_h[i] + j * Q_cell / (self.N * self.m_h))
                self.T_cell_vec_h.append(PropsSI('T', 'H', self.h_cell_vec_h[-1], 'P', self.p_hi, self.Fluid_h))
                self.h_cell_vec_c.append(self.hvec_c[i] + j * Q_cell / (self.N * self.m_c))
                self.T_cell_vec_c.append(PropsSI('T', 'H', self.h_cell_vec_c[-1], 'P', self.p_ci, self.Fluid_c))

    def pressure_solve(self):

        """solves for pressure drop in each phase"""

        N = len(self.hvec_h) - 1
        dp_h = []
        dp_c = []
        for i in range(0, N):
            h_h = (self.hvec_h[i] + self.hvec_h[i + 1]) / 2
            T_h = (self.Tvec_h[i] + self.Tvec_h[i + 1]) / 2
            h_c = (self.hvec_c[i] + self.hvec_c[i + 1]) / 2
            T_c = (self.Tvec_c[i] + self.Tvec_c[i + 1]) / 2
            T_w = ((self.Tvec_h[i] + self.Tvec_h[i + 1]) / 2 + (self.Tvec_c[i] + self.Tvec_c[i + 1]) / 2) / 2
            T_w = (T_h + T_c) / 2
            dp_h.append(
                self.pressure_loss(self.p_hi, h_h, T_h, T_w, self.m_h, self.phases_h[i], self.Fluid_h, self.Areq[i]))
            dp_c.append(
                self.pressure_loss(self.p_ci, h_c, T_c, T_w, self.m_c, self.phases_c[i], self.Fluid_c, self.Areq[i]))

        self.p_co = self.p_ci - sum(dp_c)
        self.p_ho = self.p_hi - sum(dp_h)

    def alpha_correlation(self, p, h, T, Tw, m, fluid):

        if fluid == 'R134a':

            S = self.S
            D_h = self.Dh
            n = self.n

            if PhaseSI('P', p, 'H', h, fluid) == "twophase":

                """"

                R134a Evaporation Heat Transfer Correlation by
                Yan, Y.Y., Lin, T.F., 1999. Evaporation heat transfer and pressure drop of refrigerant R-134a
                in a plate heat exchanger. J. Heat Transfer 121

                """

                x = PropsSI('Q', 'P', p, 'H', h, fluid)
                rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)
                rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)
                mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)
                lamda_l = PropsSI('L', 'P', p, 'Q', 0.0, fluid)
                h_evap = PropsSI('H', 'P', p, 'Q', 1.0, fluid) - PropsSI('H', 'P', p, 'Q', 0.0, fluid)
                Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0.0, fluid)
                G = m / n / S
                Re = G * D_h / mu_l
                G_eq = G * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
                Re_eq = G_eq * D_h / mu_l
                q = 10.5
                Bo_eq = q / (G_eq * h_evap)

                return 1.926 * (lamda_l / D_h) * Re_eq * Pr_l ** (1 / 3) * Bo_eq ** (0.3) * Re ** (-0.5)
                # return 2500

            elif PhaseSI('H', h, 'P', p, fluid) == "gas":
                return 100

            else:
                return 1000

        elif fluid == 'R744':

            S = self.S
            t = self.t
            D_h = self.Dh
            phi = 1.5

            if PhaseSI('H', h, 'P', p, fluid) == "twophase":

                """"
                R744 Condensation Heat Transfer Correlation by
                Longo, G.A., Righetti, G., Zilio, C., 2014. A New Model for Refrigeration Condensation Inside a Brazed
                Plate Heat Exchanger(BPHE).Kyoto, Japan, Proceedings of the 15th International Heat Transfer Conference,
                IHTC - 15, August 10– 15.

                """

                x = PropsSI('Q', 'P', p, 'H', h, fluid)
                rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)
                rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)
                mu_v = PropsSI('V', 'P', p, 'Q', 1.0, fluid)
                mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)
                lamda_l = PropsSI('L', 'P', p, 'Q', 0.0, fluid)
                lamda_v = PropsSI('L', 'P', p, 'Q', 1.0, fluid)
                h_evap = PropsSI('H', 'P', p, 'Q', 1.0, fluid) - PropsSI('H', 'P', p, 'Q', 0.0, fluid)
                Re_v = m / S * D_h / mu_v
                cp_h = PropsSI('C', 'P', p, 'H', h, fluid)
                G_eq = m / S * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
                Re_eq = G_eq * D_h / mu_l
                Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0.0, fluid)
                Pr_v = PropsSI('Prandtl', 'P', p, 'Q', 1.0, fluid)
                q = (lamda_l * (1 - x) + lamda_v * x) * np.abs(T - Tw) / t
                h_sat = 1.875 * phi * (lamda_l / D_h) * Re_eq ** 0.445 * Pr_l ** (1 / 3)
                h_l = 0.2267 * (lamda_v / D_h) * Re_v ** 0.631 * Pr_v ** (1 / 3)
                T_sat = PropsSI('T', 'P', p, 'Q', 1.0, fluid)
                if (T_sat - Tw) > 0:
                    F = (T - T_sat) / (T_sat - Tw)
                else:
                    F = 0

                return h_sat + F * (h_l + (cp_h * q) / h_evap)

            elif PhaseSI('H', h, 'P', p, fluid) == "gas":
                return 100

            else:
                return 1000

        else:
            return 1000

    def pressure_loss(self, p, h, T, Tw, m, phase, fluid, A_req):

        S = self.S
        D_h = self.Dh
        A = self.A_h
        L = self.L
        n = self.n

        rho = PropsSI('D', 'H', h, 'P', p, fluid)
        G = m / n / S

        if phase == 'two-phase':

            if fluid == 'R134a':

                """"

                R134a Pressure Drop Correlation by
                Yan, Y.Y., Lin, T.F., 1999. Evaporation heat transfer and pressure drop of refrigerant R-134a
                in a plate heat exchanger. J. Heat Transfer 121

                """

                x = PropsSI('Q', 'P', p, 'H', h, fluid)
                rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)
                rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)
                mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)
                p_crit = PropsSI('PCRIT', fluid)
                h_evap = PropsSI('H', 'P', p, 'Q', 1.0, fluid) - PropsSI('H', 'P', p, 'Q', 0.0, fluid)
                Re = G * D_h / mu_l
                G_eq = G * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
                Re_eq = G_eq * D_h / mu_l
                q = 10.5
                Bo = q / (G * h_evap)
                f = 94.75 * (p / p_crit) ** 0.8 * Bo ** 0.5 * Re ** (-0.4) * Re_eq ** (-0.0467)

            elif fluid == 'R744':

                """""
                Condensation Pressure Drop Correlation by
                Shah, M., 1979. A general correlation for heat transfer during film condensation inside pipes.
                Int.J.Heat Mass Transf. 22(3), 547– 556.

                """

                x = PropsSI('Q', 'P', p, 'H', h, fluid)
                rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)
                rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)
                mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)
                p_crit = PropsSI('PCRIT', fluid)
                h_evap = PropsSI('H', 'P', p, 'Q', 1.0, fluid) - PropsSI('H', 'P', p, 'Q', 0.0, fluid)
                Re = G * D_h / mu_l
                G_eq = G * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
                Re_eq = G_eq * D_h / mu_l
                q = 10.5
                Bo = q / (G * h_evap)
                f = 94.75 * (p / p_crit) ** 0.8 * Bo ** 0.5 * Re ** (-0.4) * Re_eq ** (-0.0467)

            else:
                f = 1.0

            return 2 * f * G ** 2 / rho * L * A_req / A / D_h

        else:
            f = 1.0
            return 2 * f * G ** 2 / rho * L * A_req / A / D_h


def solver(component: [Component]):

    """
    solves heat exchanger object using Finite-Volume-Moving-Boundary-Coupling Algorithm

    :param    component:   heat exchanger component object
    :return:  None:        all port states of the heat exchanger object gets updated by solution values
    """

    A = 1.76

    try:
        for port in component.ports:
            if port.port_id[2] == psd['c']:
                cold_fluid = port.fluid
                p_in_c = port.p.value
                h_in_c = port.h.value
                m_in_c = port.m.value
            elif port.port_id[2] == psd['h']:
                hot_fluid = port.fluid
                p_in_h = port.p.value
                h_in_h = port.h.value
                m_in_h = port.m.value

        if component.linearized:
            x = np.array(component.x0.copy())
            i = 0
            for port in component.ports:
                if port.port_typ == 'in' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    x[i+1] = port.h.value
                    x[i+2] = port.m.value
                    i += 3

        if PropsSI('T', 'P', p_in_h, 'H', h_in_h, hot_fluid) < PropsSI('T', 'P', p_in_c, 'H', h_in_c, cold_fluid):
            HX = FVMBCouplingAlg(cold_fluid, p_in_c, h_in_c, m_in_c, hot_fluid, p_in_h, h_in_h, m_in_h)
            HX.A_h = HX.A_c = A
            HX.run()
            m_out_c = m_in_c
            m_out_h = m_in_h
            h_out_h = HX.hvec_c[-1]
            h_out_c = HX.hvec_h[0]
            p_out_h = HX.p_co
            p_out_c = HX.p_ho

        else:
            HX = FVMBCouplingAlg(hot_fluid, p_in_h, h_in_h, m_in_h, cold_fluid, p_in_c, h_in_c, m_in_c)
            HX.A_h = HX.A_c = A
            HX.run()
            m_out_c = m_in_c
            m_out_h = m_in_h
            h_out_h = HX.hvec_h[0]
            h_out_c = HX.hvec_c[-1]
            p_out_c = HX.p_co
            p_out_h = HX.p_ho

        if component.linearized:
            F = np.dot(component.J, (x - component.x0)) + component.F0
            for port in component.ports:
                if port.port_id[2] == psd['-h']:
                    port.p.set_value(component.lamda * p_out_h + (1 - component.lamda) * F[0])
                    port.h.set_value(component.lamda * h_out_h + (1 - component.lamda) * F[1])
                    port.m.set_value(component.lamda * m_in_h + (1 - component.lamda) * F[2])

                elif port.port_id[2] == psd['-c']:
                    port.p.set_value(component.lamda * p_out_c + (1 - component.lamda) * F[3])
                    port.h.set_value(component.lamda * h_out_c + (1 - component.lamda) * F[4])
                    port.m.set_value(component.lamda * m_in_c + (1 - component.lamda) * F[5])
        else:
            for port in component.ports:
                if port.port_id[2] == psd['-c']:
                    port.p.set_value(p_out_c)
                    port.h.set_value(h_out_c)
                    port.m.set_value(m_out_c)
                elif port.port_id[2] == psd['-h']:
                    port.p.set_value(p_out_h)
                    port.h.set_value(h_out_h)
                    port.m.set_value(m_out_h)

    except(RuntimeError, ValueError):
        print(component.name + ' failed!')
        component.status = 0

    # A = [0]
    # for i, element in enumerate(HX.Areq):
    #     A.append(A[-1] + element)
    # A = A / sum(HX.Areq)
    #
    # fig, ax = plt.subplots()
    # fig.suptitle(component.name, fontsize=16)
    # ax.plot(np.flip(HX.hvec_c), np.flip(HX.Tvec_c), 'b-')
    # ax.plot(HX.hvec_h, HX.Tvec_h, 'r-')
    # ax.set_xlabel('Enthalpy [kJ/kg]')
    # ax.set_ylabel('Temperature [K]')
    # ax.grid(True)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # fig.suptitle(component.name, fontsize=16)
    # ax.plot(A, HX.Tvec_c - 273.15, 'b-')
    # ax.plot(A, HX.Tvec_h - 273.15, 'r-')
    # ax.set_xlabel('Enthalpy [kJ/kg]')
    # ax.set_ylabel('Temperature [K]')
    # ax.grid(True)
    # plt.show()