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
    Dh = 0.000866397 # hydraulic diameter [m]
    n = 22 # number of channels
    S = 0.000216787 # Cross sectional area per channel [m2]
    k = 16.7 # heat conductivity [W/mK]

    fluid_list = ['Water', 'Air', 'R134a', 'R744']

    def __init__(self, Fluid_h, mdot_h, p_hi, T_hi, Fluid_c, mdot_c, p_ci, h_ci):

        # Set variables in the class instance
        self.Fluid_h = Fluid_h
        self.mdot_h = mdot_h
        self.T_hi = T_hi
        self.p_hi = p_hi
        self.Fluid_c = Fluid_c
        self.mdot_c = mdot_c
        self.h_ci = h_ci
        self.p_ci = p_ci

        # Determine the inlet temperatures from the pressure/enthalpy pairs
        if not Fluid_h == 'R134a':
            self.T_hi = T_hi
        self.T_ci = PropsSI('T', 'P', self.p_ci, 'H', self.h_ci, self.Fluid_c)

        # Calculate the bubble and dew enthalpies for each stream
        if self.Fluid_c == 'R134a':
            self.T_cbubble = PropsSI('T', 'P', self.p_ci, 'Q', 0, self.Fluid_c)
            self.T_cdew = PropsSI('T', 'P', self.p_ci, 'Q', 1, self.Fluid_c)
            self.h_cbubble = PropsSI('H', 'T', self.T_cbubble, 'Q', 0, self.Fluid_c)
            self.h_cdew = PropsSI('H', 'T', self.T_cdew, 'Q', 1, self.Fluid_c)
        else:
            self.T_cbubble = None
            self.T_cdew = None
            self.h_cbubble = None
            self.h_cdew = None
        if not self.Fluid_h == 'R134a':
            self.T_hbubble = None
            self.T_hdew = None
            self.h_hbubble = None
            self.h_hdew = None
        self.T_cmax = PropsSI("TMAX", self.Fluid_c)
        self.T_hmin = PropsSI("TMIN", self.Fluid_h)
        # self.T_hmin = 256.243841

    def external_pinching(self):
        """ Determine the maximum heat transfer rate based on the external pinching analysis """


        if self.T_hmin < self.T_ci:
            self.T_ho = self.T_hmin
        else:
            self.T_ho = self.T_ci

        Qmaxh = self.mdot_h * PropsSI('C', 'T', (self.T_hi + self.T_ho) / 2, 'P', self.p_hi, self.Fluid_h) * (self.T_hi - self.T_ho)

        if self.T_cmax > self.T_hi:
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
        self.T_ho = self.T_hi - Q / self.mdot_h / PropsSI('C', 'T', (self.T_hi + self.T_ho) / 2, 'P', self.p_hi, self.Fluid_h)

        # Start with the external boundaries (sorted in increasing enthalpy)
        self.hvec_c = [self.h_ci, self.h_co]
        self.Tvec_h = [self.T_ho, self.T_hi]

        # Add the bubble and dew enthalpies for the cold stream
        if self.h_cdew is not None and self.h_ci + 1e-4 < self.h_cdew < self.h_co - 1e-4:
            self.hvec_c.insert(-1, self.h_cdew)
        if self.h_cbubble is not None and self.h_ci + 1e-4 < self.h_cbubble < self.h_co - 1e-4:
            self.hvec_c.insert(1, self.h_cbubble)

        # Fill in the complementary cell boundaries
        # Start at the first element in the vector
        k = 0
        while k < len(self.hvec_c) - 1:
            if len(self.hvec_c) == 2:
                break

            # Determine which stream is the limiting next cell boundary
            Qcell_hk = self.mdot_h * PropsSI('C', 'T', self.Tvec_h[k], 'P', self.p_hi, self.Fluid_h) * (self.Tvec_h[k + 1] - self.Tvec_h[k])
            Qcell_ck = self.mdot_c * (self.hvec_c[k + 1] - self.hvec_c[k])

            if abs(Qcell_hk / Qcell_ck - 1) < 1e-6:
                k += 1
                break
            Qcell_ck = self.mdot_c * (self.hvec_c[k + 1] - self.hvec_c[k])

            # Hot stream needs a complementary cell boundary
            self.Tvec_h.insert(k + 1, self.Tvec_h[k] + Qcell_ck / self.mdot_h / PropsSI('C', 'T', self.Tvec_h[k], 'P',
                                                                                    self.p_hi, self.Fluid_h))

            # Increment index
            k += 1

        # Calculate the temperature and entropy at each cell boundary
        self.Tvec_c = PropsSI('T', 'H', self.hvec_c, 'P', self.p_ci, self.Fluid_c)

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

        self.phases_h = len(self.Tvec_h) * ['liquid']

    def internal_pinching(self, stream):

        """
        Determine the maximum heat transfer rate based on the internal pinching analysis
        """

        if stream == 'hot':

            Qmax = self.mdot_h * PropsSI('C', 'T', (self.T_hi + self.T_ho) / 2, 'P', self.p_hi, self.Fluid_h) * (self.Tvec_h[-1] - self.Tvec_h[0])
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

                        # Heat transfer in the cell
                        Qleft = self.mdot_c * (self.h_cbubble - self.h_ci)

                        # New value for the limiting heat transfer rate
                        Qmax = Qleft + self.mdot_h * PropsSI ('C', 'T', (self.T_hi + self.T_ho) / 2, 'P', self.p_hi, self.Fluid_h) * (self.T_hi - self.T_cbubble)

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

            UA_req = self.mdot_h * PropsSI('C', 'T', (self.T_hi + self.T_ho) / 2, 'P', self.p_hi, self.Fluid_h) * (self.Tvec_h[k + 1] - self.Tvec_h[k]) / LMTD
            T_h = (self.Tvec_h[k + 1] + self.Tvec_h[k]) / 2
            h_c = (self.hvec_c[k + 1] + self.hvec_c[k]) / 2
            T_c = (self.Tvec_c[k + 1] + self.Tvec_c[k]) / 2
            T_w = (T_h + T_c) / 2
            alpha_h = self.alpha_correlation(self.p_hi, T_h, T_h, T_w, self.mdot_h, self.phases_h[k], self.Fluid_h)
            alpha_c = self.alpha_correlation(self.p_ci, h_c, T_c, T_w, self.mdot_c, self.phases_c[k], self.Fluid_c)

            UA_avail = 1 / (1 / (alpha_h * self.A_h) + 1 / (alpha_c * self.A_c))
            Uj = 1 / (1 / alpha_h + self.A_h / self.A_c * 1 / alpha_c)
            self.Areq.append(UA_req / Uj)
            w.append(UA_req / UA_avail)

        return 1 - sum(w)

    def solve(self):
        """
        Solve the objective function using Brent's method and the maximum heat transfer
        rate calculated from the pinching analysis
        # """
        if self.objective_function(self.Qmax - 1e-10) < 0:
            self.Q = scipy.optimize.brentq(self.objective_function, 1e-5, self.Qmax - 1e-10, rtol=1e-14, xtol=1e-10)
        else:
            self.Q = self.Qmax
        return self.Q

    def alpha_correlation(self, p, h, T, Tw, m, phase, fluid):

        S = self.S
        D_h = self.Dh
        n = self.n

        if fluid == 'R134a':

            if phase == "two-phase":

                """
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
                G = m / n / S
                Re = G * D_h / mu_l
                G_eq = G * ((1 - x) + x * (rho_l / rho_v)) ** (1 / 2)
                Re_eq = G_eq * D_h / mu_l
                q = 10.5 * 1e3
                Bo_eq = q / (G_eq * h_evap)
                Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0.0, fluid)

                return 1.926 * (lamda_l / D_h) * Re_eq * Pr_l ** (1 / 3) * Bo_eq ** (0.3) * Re ** (-0.5)

            elif phase == "vapor":
                return 1000

            else:
                return 1000

        else:
            return 12200


def solver(component: [Component]):

    """
    solves heat exchanger object using moving boundary algorithm

    :param    component:   heat exchanger component object
    :return:  None:        all port states of the heat exchanger object gets updated by solution values
    """

    try:

        A = component.parameter['A'].value

        T_in_h = component.ports[psd['h']].h.value
        p_in_h = component.ports[psd['h']].p.value
        m_in_h = component.ports[psd['h']].m.value
        hot_fluid = component.ports[psd['h']].fluid

        h_in_c = component.ports[psd['c']].h.value
        p_in_c = component.ports[psd['c']].p.value
        m_in_c = component.ports[psd['c']].m.value
        cold_fluid = component.ports[psd['c']].fluid

        if T_in_h < PropsSI('T', 'P', p_in_c, 'H', h_in_c, cold_fluid):
            raise RuntimeError('Secondary side fluid is hoter than primary side fluid!')
        else:
            HX = HeatExchanger(hot_fluid, m_in_h, p_in_h, T_in_h, cold_fluid, m_in_c, p_in_c, h_in_c)
            HX.A_h = HX.A_c = A
            HX.run(and_solve=True)
            h_out_c = HX.hvec_c[-1]
            T_out_h = HX.Tvec_h[0]
            p_out_c = p_in_c
            p_out_h = p_in_h
            m_out_h = m_in_h
            m_out_c = m_in_c

        component.ports[psd['-h']].p.set_value(p_out_h)
        component.ports[psd['-h']].h.set_value(T_out_h)
        component.ports[psd['-h']].m.set_value(m_out_h)

        component.ports[psd['-c']].p.set_value(p_out_c)
        component.ports[psd['-c']].h.set_value(h_out_c)
        component.ports[psd['-c']].m.set_value(m_out_c)

        component.outputs['Q'] = HX.Q

    except:
        print('Evaporator ' + str(component.number) + ' ' + 'failed!')
        component.status = 0

    if component.diagramm_plot:
        A = [0]
        for i, element in enumerate(HX.Areq):
            A.append((sum(A) + element))
        A = A / sum(HX.Areq)

        fig, ax = plt.subplots()
        fig.suptitle(f'{component.name} \n {round(HX.Q / 1000, 3)} kW', fontsize=16)
        ax.plot(A, HX.Tvec_c - 273.15, 'b-')
        ax.plot(A, np.array(HX.Tvec_h) - 273.15, 'r-')
        ax.set_xlabel('$A / A_{ges} [-]$')
        ax.set_ylabel('$Temperature [\u00b0 C]$')
        ax.grid(True)
        plt.show()
