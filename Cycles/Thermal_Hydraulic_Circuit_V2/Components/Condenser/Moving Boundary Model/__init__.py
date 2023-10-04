from CoolProp.CoolProp import PropsSI, PhaseSI
import matplotlib.pyplot as plt
import numpy as np
from math import log
import scipy.optimize
from system import Component, psd


class HeatExchanger(object):
    """
    Supplemental code for paper:
    I. Bell et al., "A Generalized Moving-Boundary Algorithm to Predict the Heat Transfer Rate of
    Counterflow Heat Exchangers for any Phase Configuration", Applied Thermal Engineering, 2014
    """

    A_c = None # heat transfer area hot side
    A_h = None # heat transfer area cold side
    t = 0.003  # wall thickness [m]
    L = 0.5 # heat excahnge Length [m]
    Dh = 0.000909633 # hydraulic diameter [m]
    n = 22 # number of channels
    S = 0.000227615 # Cross sectional area per channel [m2]
    k = 16.7 # heat conductivity [W/mK]

    fluid_list = ['Water', 'Air', 'R134a', 'R744']

    def __init__(self, Fluid_h, mdot_h, p_hi, h_hi, Fluid_c, mdot_c, p_ci, h_ci):

        # Set variables in the class instance
        self.Fluid_h = Fluid_h
        self.mdot_h = mdot_h
        self.h_hi = h_hi
        self.p_hi = p_hi
        self.Fluid_c = Fluid_c
        self.mdot_c = mdot_c
        self.h_ci = h_ci
        self.p_ci = p_ci

        # Determine the inlet temperatures from the pressure/enthalpy pairs
        self.T_ci = PropsSI('T', 'P', self.p_ci, 'H', self.h_ci, self.Fluid_c)
        self.T_hi = PropsSI('T', 'P', self.p_hi, 'H', self.h_hi, self.Fluid_h)

        # Calculate the bubble and dew enthalpies for each stream
        if self.Fluid_c in self.fluid_list:
            self.T_cbubble = PropsSI('T', 'P', self.p_ci, 'Q', 0, self.Fluid_c)
            self.T_cdew = PropsSI('T', 'P', self.p_ci, 'Q', 1, self.Fluid_c)
            self.h_cbubble = PropsSI('H', 'T', self.T_cbubble, 'Q', 0, self.Fluid_c)
            self.h_cdew = PropsSI('H', 'T', self.T_cdew, 'Q', 1, self.Fluid_c)
        else:
            self.T_cbubble = None
            self.T_cdew = None
            self.h_cbubble = None
            self.h_cdew = None
        if self.Fluid_h in self.fluid_list:
            self.T_hbubble = PropsSI('T', 'P', self.p_hi, 'Q', 0, self.Fluid_h)
            self.T_hdew = PropsSI('T', 'P', self.p_hi, 'Q', 1, self.Fluid_h)
            self.h_hbubble = PropsSI('H', 'T', self.T_hbubble, 'Q', 0, self.Fluid_h)
            self.h_hdew = PropsSI('H', 'T', self.T_hdew, 'Q', 1, self.Fluid_h)
        else:
            self.T_hbubble = None
            self.T_hdew = None
            self.h_hbubble = None
            self.h_hdew = None
        self.T_cmax = PropsSI("TMAX", self.Fluid_c)
        self.T_hmin = PropsSI("TMIN", self.Fluid_h)

    def external_pinching(self):
        """ Determine the maximum heat transfer rate based on the external pinching analysis """

        if self.T_hmin < self.T_ci:
            if abs(self.T_ci - PropsSI('T', 'P', self.p_hi, 'Q', 1.0, self.Fluid_h)) < 1e-4:
                self.h_ho = PropsSI('H', 'P', self.p_hi, 'Q', 1.0, self.Fluid_h)
            else:
                self.h_ho = PropsSI('H', 'T', self.T_ci, 'P', self.p_hi, self.Fluid_h)
        else:
            self.h_ho = PropsSI('H', 'T', self.T_hmin, 'P', self.p_hi, self.Fluid_h)

        Qmaxh = self.mdot_h * (self.h_hi - self.h_ho)

        if self.T_cmax > self.T_hi:
            if abs(self.T_hi - PropsSI('T', 'P', self.p_ci, 'Q', 1.0, self.Fluid_h)) < 1e-4:
                self.h_co = PropsSI('H', 'P', self.p_ci, 'Q', 1.0, self.Fluid_c)
            else:
                self.h_co = PropsSI('H', 'T', self.T_hi, 'P', self.p_ci, self.Fluid_c)
        else:
            self.h_co = PropsSI('H', 'T', self.T_cmax, 'P', self.p_ci, self.Fluid_c)

        Qmaxc = self.mdot_c * (self.h_co - self.h_ci)

        Qmax = min(Qmaxh, Qmaxc)

        self.calculate_cell_boundaries(Qmax)

        return Qmax

    def calculate_cell_boundaries(self, Q):
        """ Calculate the cell boundaries for each fluid """

        # Re-calculate the outlet enthalpies of each stream
        self.h_co = self.h_ci + Q / self.mdot_c
        self.h_ho = self.h_hi - Q / self.mdot_h

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
            Qcell_hk = self.mdot_h * (self.hvec_h[k + 1] - self.hvec_h[k])
            Qcell_ck = self.mdot_c * (self.hvec_c[k + 1] - self.hvec_c[k])

            if abs(Qcell_hk / Qcell_ck - 1) < 1e-6:
                k += 1
                break
            elif Qcell_hk > Qcell_ck:
                # Hot stream needs a complementary cell boundary
                self.hvec_h.insert(k + 1, self.hvec_h[k] + Qcell_ck / self.mdot_h)
            else:
                # Cold stream needs a complementary cell boundary
                self.hvec_c.insert(k + 1, self.hvec_c[k] + Qcell_hk / self.mdot_c)

            Qcell_hk = self.mdot_h * (self.hvec_h[k + 1] - self.hvec_h[k])
            Qcell_ck = self.mdot_c * (self.hvec_c[k + 1] - self.hvec_c[k])
            # assert (abs(Qcell_hk / Qcell_ck - 1) < 1e-6)

            # Increment index
            k += 1

        # Calculate the temperature and entropy at each cell boundary
        self.Tvec_c = PropsSI('T', 'H', self.hvec_c, 'P', self.p_ci, self.Fluid_c)
        self.Tvec_h = PropsSI('T', 'H', self.hvec_h, 'P', self.p_hi, self.Fluid_h)
        self.svec_c = PropsSI('S', 'H', self.hvec_c, 'P', self.p_ci, self.Fluid_c)
        self.svec_h = PropsSI('S', 'H', self.hvec_h, 'P', self.p_hi, self.Fluid_h)

        # Calculate the phase in each cell
        self.phases_h = []
        for i in range(len(self.hvec_h) - 1):
            havg = (self.hvec_h[i] + self.hvec_h[i + 1]) / 2.0
            if self.Fluid_h in self.fluid_list:
                if havg < self.h_hbubble:
                    self.phases_h.append('liquid')
                elif havg > self.h_hdew:
                    self.phases_h.append('vapor')
                else:
                    self.phases_h.append('two-phase')
            else:
                self.phases_h.append(PhaseSI('H', havg, 'P', self.p_ci, self.Fluid_h))

        self.phases_c = []
        for i in range(len(self.hvec_c) - 1):
            havg = (self.hvec_c[i] + self.hvec_c[i + 1]) / 2.0
            if self.Fluid_c in self.fluid_list:
                if havg < self.h_cbubble:
                    self.phases_c.append('liquid')
                elif havg > self.h_cdew:
                    self.phases_c.append('vapor')
                else:
                    self.phases_c.append('two-phase')
            else:
                self.phases_c.append(PhaseSI('H', havg, 'P', self.p_ci, self.Fluid_c))

    def internal_pinching(self, stream):

        """
        Determine the maximum heat transfer rate based on the internal pinching analysis
        """

        if stream == 'hot':

            # Try to find the dew point enthalpy as one of the cell boundaries
            # that is not the inlet or outlet

            if self.h_hdew == None:
                Qmax = self.mdot_h * (self.hvec_h[-1] - self.hvec_h[0])
                return Qmax
            # Check for the hot stream pinch point
            for i in range(1, len(self.hvec_h) - 1):

                # Check if enthalpy is equal to the dewpoint enthalpy of hot
                # stream and hot stream is colder than cold stream (impossible)
                if (abs(self.hvec_h[i] - self.h_hdew) < 1e-6
                        and self.Tvec_c[i] > self.Tvec_h[i]):
                    # Enthalpy of the cold stream at the pinch temperature
                    h_c_pinch = PropsSI('H', 'T', self.T_hdew, 'P', self.p_ci, self.Fluid_c)

                    # Heat transfer in the cell
                    Qright = self.mdot_h * (self.h_hi - self.h_hdew)

                    # New value for the limiting heat transfer rate
                    Qmax = self.mdot_c * (h_c_pinch - self.h_ci) + Qright

                    # Recalculate the cell boundaries
                    self.calculate_cell_boundaries(Qmax)

                    return Qmax

        elif stream == 'cold':
            if self.h_cbubble == None:
                Qmax = self.mdot_c * (self.hvec_c[-1] - self.hvec_c[0])
                return Qmax
            else:
                # Check for the cold stream pinch point
                for i in range(1, len(self.hvec_c) - 1):

                    # Check if enthalpy is equal to the bubblepoint enthalpy of cold
                    # stream and hot stream is colder than cold stream (impossible)
                    if (abs(self.hvec_c[i] - self.h_cbubble) < 1e-6
                            and self.Tvec_c[i] > self.Tvec_h[i]):
                        # Enthalpy of the cold stream at the pinch temperature
                        h_h_pinch = PropsSI('H', 'T', self.T_cbubble, 'P', self.p_hi, self.Fluid_h)

                        # Heat transfer in the cell
                        Qleft = self.mdot_c * (self.h_cbubble - self.h_ci)

                        # New value for the limiting heat transfer rate
                        Qmax = Qleft + self.mdot_h * (self.h_hi - h_h_pinch)

                        # Recalculate the cell boundaries
                        self.calculate_cell_boundaries(Qmax)

                        return Qmax
        else:
            raise ValueError

    def run(self, only_external=False, and_solve=False):
        # Check the external pinching & update cell boundaries
        Qmax_ext = self.external_pinching()
        Qmax = Qmax_ext

        if not only_external:
            # Check the internal pinching
            for stream in ['hot', 'cold']:
                # Check stream internal pinching & update cell boundaries
                Qmax_int = self.internal_pinching(stream)
                if Qmax_int is not None:
                    Qmax = Qmax_int

        self.Qmax = Qmax

        if and_solve and not only_external:
            Q = self.solve()

        Qtotal = self.mdot_c * (self.hvec_c[-1] - self.hvec_c[0])

        # Build the normalized enthalpy vectors
        self.hnorm_h = self.mdot_h * (np.array(self.hvec_h) - self.hvec_h[0]) / Qtotal
        self.hnorm_c = self.mdot_c * (np.array(self.hvec_c) - self.hvec_c[0]) / Qtotal

        if and_solve:
            return Q

    def objective_function(self, Q):

        self.calculate_cell_boundaries(Q)

        w = []
        self.Areq = []
        for k in range(len(self.hvec_c) - 1):
            Thi = self.Tvec_h[k + 1]
            Tci = self.Tvec_c[k]
            Tho = self.Tvec_h[k]
            Tco = self.Tvec_c[k + 1]
            DTA = Thi - Tco
            DTB = Tho - Tci

            if DTA == DTB:
                LMTD = DTA
            else:
                if DTA == 0.0:
                    DTA = 1e-12
                elif DTB == 0.0:
                    DTB = 1e-12
                try:
                    LMTD = (DTA - DTB) / log(abs(DTA / DTB))
                except ValueError as VE:
                    print(Q, DTA, DTB)
                    raise
            UA_req = self.mdot_h * (self.hvec_h[k + 1] - self.hvec_h[k]) / LMTD
            h_h = (self.hvec_h[k + 1] + self.hvec_h[k]) / 2
            T_h = (self.Tvec_h[k+1] + self.Tvec_h[k]) / 2
            h_c = (self.hvec_c[k + 1] + self.hvec_c[k]) / 2
            T_c = (self.Tvec_c[k+1] + self.Tvec_c[k]) / 2
            T_w = (T_h + T_c) / 2
            alpha_h = self.alpha_correlation(self.p_hi, h_h, T_h, T_w, self.mdot_h, self.phases_h[k], self.Fluid_h)
            alpha_c = self.alpha_correlation(self.p_ci, h_c, T_c, T_w, self.mdot_c, self.phases_c[k], self.Fluid_c)

            UA_avail = 1 / (1 / (alpha_h * self.A_h) + 1 / (alpha_c * self.A_c))
            Uj = 1 / (1 / alpha_h + self.A_h/self.A_c * 1 / alpha_c)
            self.Areq.append(UA_req / Uj)
            w.append(UA_req / UA_avail)

        return 1 - sum(w)

    def solve(self):
        """
        Solve the objective function using Brent's method and the maximum heat transfer
        rate calculated from the pinching analysis

        """
        if self.objective_function(self.Qmax - 1e-10) < 0:
            self.Q = scipy.optimize.brentq(self.objective_function, 1e-5, self.Qmax - 1e-10, rtol=1e-14, xtol=1e-10)
        else:
            self.Q = self.Qmax

        self.calculate_cell_boundaries(self.Q)
        return self.Q

    def alpha_correlation(self, p, h, T, Tw, m, phase, fluid):

        # S = self.S
        # D_h = self.Dh
        # n = self.n
        # phi = 1.5
        #
        # if fluid == 'R134a':
        #
        #     if phase == "two-phase":
        #
        #         """
        #         R134aCondensation Heat Transfer Correlation by
        #         Longo, G.A., Righetti, G., Zilio, C., 2014. A New Model for Refrigeration Condensation Inside a Brazed
        #         Plate Heat Exchanger(BPHE).Kyoto, Japan, Proceedings of the 15th International Heat Transfer Conference,
        #         IHTC - 15, August 10– 15.
        #
        #         """
        #
        #         x = PropsSI('Q', 'P', p, 'H', h, fluid)
        #         rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)
        #         rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)
        #         mu_v = PropsSI('V', 'P', p, 'Q', 1.0, fluid)
        #         mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)
        #         lamda_l = PropsSI('L', 'P', p, 'Q', 0.0, fluid)
        #         lamda_v = PropsSI('L', 'P', p, 'Q', 1.0, fluid)
        #         h_evap = PropsSI('H', 'P', p, 'Q', 1.0, fluid) - PropsSI('H', 'P', p, 'Q', 0.0, fluid)
        #         cp_h = PropsSI('C', 'P', p, 'H', h, fluid)
        #         Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0.0, fluid)
        #         Pr_v = PropsSI('Prandtl', 'P', p, 'Q', 1.0, fluid)
        #         G = m / n / S
        #         Re_v = G * D_h / mu_v
        #         G_eq = G * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
        #         Re_eq = G_eq * D_h / mu_l
        #         q = 8.09
        #         h_sat = 1.875 * phi * (lamda_l / D_h) * Re_eq ** 0.445 * Pr_l ** (1 / 3)
        #         h_l = 0.2267 * (lamda_v / D_h) * Re_v ** 0.631 * Pr_v ** (1 / 3)
        #         T_sat = PropsSI('T', 'P', p, 'Q', 1.0, fluid)
        #         if (T_sat - Tw) > 0:
        #             F = (T - T_sat) / (T_sat - Tw)
        #         else:
        #             F = 0
        #         return h_sat + F * (h_l + (cp_h * q) / h_evap)
        #
        #     elif phase == "vapor":
        #
        #         return 10000
        #
        #     else:
        #
        #         return 1000
        #
        # else:
        #     return 4780

        # if fluid == 'R134a':
        #
        #     if phase == 'two-phase':
        #         return 10000
        #     elif phase == 'vapor':
        #         return 10000
        #     else:
        #         return 10000
        # else:
        #     return 1000

        return 10000


def solver(component: [Component]):

    """
    solves heat exchanger object using moving boundary algorithm

    :param    component:   heat exchanger component object
    :return:  None:        all port states of the heat exchanger object gets updated by solution values
    """

    try:

        A = component.parameter['A'].value

        h_in_h = component.ports[psd['h']].h.value
        p_in_h = component.ports[psd['h']].p.value
        m_in_h = component.ports[psd['h']].m.value
        hot_fluid = component.ports[psd['h']].fluid

        h_in_c = component.ports[psd['c']].h.value
        p_in_c = component.ports[psd['c']].p.value
        m_in_c = component.ports[psd['c']].m.value
        cold_fluid = component.ports[psd['c']].fluid

        if component.linearized:
            x = np.array(component.x0.copy())
            i = 0
            for port in component.ports:
                if port.port_type == 'in' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    x[i+1] = port.h.value
                    x[i+2] = port.m.value
                    i += 3

        if PropsSI('T', 'P', p_in_h, 'H', h_in_h, hot_fluid) < PropsSI('T', 'P', p_in_c, 'H', h_in_c, cold_fluid):
            HX = HeatExchanger(cold_fluid, m_in_c, p_in_c, h_in_c, hot_fluid, m_in_h, p_in_h, h_in_h)
            HX.A_h = HX.A_c = A
            HX.run(and_solve=True)
            h_out_h = HX.hvec_c[-1]
            h_out_c = HX.hvec_h[0]
            p_out_c = p_in_c
            p_out_h = p_in_h
            m_out_h = m_in_h
            m_out_c = m_in_c
        else:
            HX = HeatExchanger(hot_fluid, m_in_h, p_in_h, h_in_h, cold_fluid, m_in_c, p_in_c, h_in_c)
            HX.A_h = HX.A_c = A
            HX.run(and_solve=True)
            h_out_h = HX.hvec_h[0]
            h_out_c = HX.hvec_c[-1]
            p_out_c = p_in_c
            p_out_h = p_in_h
            m_out_h = m_in_h
            m_out_c = m_in_c

        if component.linearized:
            F = np.dot(component.J, (x - component.x0)) + component.F0
            for port in component.ports:
                if port.port_id[2] == psd['-h']:
                    p_out_h = component.lamda * p_out_h + (1 - component.lamda) * F[0]
                    h_out_h = component.lamda * h_out_h + (1 - component.lamda) * F[1]
                    m_out_h = component.lamda * m_in_h + (1 - component.lamda) * F[2]

                elif port.port_id[2] == psd['-c']:
                    p_out_c = component.lamda * p_out_c + (1 - component.lamda) * F[3]
                    h_out_c = component.lamda * h_out_c + (1 - component.lamda) * F[4]
                    m_out_c = component.lamda * m_in_c + (1 - component.lamda) * F[5]

        component.ports[psd['-h']].p.set_value(p_out_h)
        component.ports[psd['-h']].h.set_value(h_out_h)
        component.ports[psd['-h']].m.set_value(m_out_h)

        component.ports[psd['-c']].p.set_value(p_out_c)
        component.ports[psd['-c']].h.set_value(h_out_c)
        component.ports[psd['-c']].m.set_value(m_out_c)

        component.outputs['Q'].set_value(HX.Q)

    except:
        print(component.name + ' failed!')
        component.status = 0

    if component.diagramm_plot:

        A = [0]
        for i, element in enumerate(HX.Areq):
            A.append(A[-1] + element)
        A = A / sum(HX.Areq)

        fig, ax = plt.subplots()
        fig.suptitle(f'{component.name} \n {round(HX.Q / 1000, 3)} kW', fontsize=16)
        ax.plot(A, HX.Tvec_c-273.15, 'b-')
        ax.plot(A, HX.Tvec_h-273.15, 'r-')
        ax.set_xlabel('$A / A_{ges} [-]$')
        ax.set_ylabel('$Temperature [\u00b0 C]$')
        ax.grid(True)
        plt.show()
