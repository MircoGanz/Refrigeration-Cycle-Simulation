

from system import *


import os
from pathlib import Path
import pandas as pd


def comp_poly_map_analytical(pi, po, hi, dpi, dpo, dhi, f, fluid):
    """
    Compute performance map using polynomial coefficients.

    Args:
    - p_evap (float): Evaporator pressure in bar.
    - p_cond (float): Condenser pressure in bar.
    - p1 (float): Pressure at suction side of compressor in bar.
    - T1 (float): Temperature at suction side of compressor in °C.
    - f (float): Compressor frequency in Hz.
    - fluid (str): Fluid name.

    Returns:
    - tuple: Output mass flow rate, and total derivatives of the mass flow rate with respect to p_evap, p_cond, p1 amd T1.
    """
    SH_ref = 10
    if not f % 5 == 0 and 30 <= f <= 70:
        raise RuntimeError(f'Frequency must be divisible by 5 and between 30 and 70 Hz, but is {f}.')

    # Load polynomial coefficients from file
    directory = os.getcwd()
    source_dir = Path(directory + '/Cycles/OneStageDX/Components/Compressor/Polynomial based Model/Polynomials_4TES-12Y_R134a')
    coefficients = pd.read_csv(
        source_dir / f'10K-UH_{str(f)}Hz.csv',
        sep=';', encoding='latin', skiprows=26, nrows=3, header=0, index_col=0)

    def evalPoly(to, tc, constants):
        """
        Evaluate compressor polynomial function.

        Args:
        - to (float): Evaporator temperature in °C.
        - tc (float): Condenser temperature in °C.
        - constants (pd.Series): Coefficients of the polynomial.

        Returns:
        - float: Result of the polynomial evaluation.
        """
        Phi = constants.c1 + constants.c2 * to + constants.c3 * tc + constants.c4 * to ** 2 + constants.c5 * to * tc + \
              constants.c6 * tc ** 2 + constants.c7 * to ** 3 + constants.c8 * tc * to ** 2 + constants.c9 * to * tc ** 2 \
              + constants.c10 * tc ** 3
        return Phi

    def der_evalPoly_dT_evap(to, tc, constants):
        """
        Evaluate the derivative of compressor polynomial with respect to evaporator temperature.

        Args:
        - to (float): Evaporator temperature in Celsius.
        - tc (float): Condenser temperature in Celsius.
        - constants (pd.Series): Coefficients of the polynomial.

        Returns:
        - float: Result of the derivative evaluation.
        """
        Phi = constants.c2 + 2 * constants.c4 * to + constants.c5 * tc + 3 * constants.c7 * to ** 2 + \
              2 * constants.c8 * tc * to + constants.c9 * tc ** 2
        return Phi

    def der_evalPoly_dT_cond(to, tc, constants):
        """
        Evaluate the derivative of compressor polynomial with respect to condenser temperature.

        Args:
        - to (float): Evaporator temperature in Celsius.
        - tc (float): Condenser temperature in Celsius.
        - constants (pd.Series): Coefficients of the polynomial.

        Returns:
        - float: Result of the derivative evaluation.
        """
        Phi = constants.c3 + constants.c5 * to + 2 * constants.c6 * tc + constants.c8 * to ** 2 + \
              2 * constants.c9 * to * tc + 3 * constants.c10 * tc ** 2
        return Phi

    # Check if state at T1 is in the gas or supercritical gas phase
    assert PhaseSI('H', hi, 'P', pi, fluid) in ['gas', 'supercritical_gas']

    # Calculate properties at key points
    T_evap = PropsSI('T', 'P', pi, 'Q', 1.0, fluid)
    dT_evap_dpi = PropsSI('d(T)/d(P)|H', 'P', pi, 'Q', 1, fluid)

    T_cond = PropsSI('T', 'P', po, 'Q', 1.0, fluid)
    dT_cond_dpo = PropsSI('d(T)/d(P)|H', 'P', po, 'Q', 1, fluid)

    rho = PropsSI("D", "H", hi, "P", pi, fluid)
    drho_dhi = PropsSI("d(D)/d(H)|P", "H", hi, "P", pi, fluid)
    drho_dpi = PropsSI("d(D)/d(P)|H", "H", hi, "P", pi, fluid)

    rho_ref = PropsSI("D", "P", pi, "T", T_evap + SH_ref, fluid)
    drho_ref_dpi = PropsSI("d(D)/d(P)|T", "P", pi, "T", T_evap + SH_ref, fluid)
    drho_ref_dT_evap = PropsSI("d(D)/d(T)|P", "P", pi, "T", T_evap + SH_ref, fluid)

    # Evaluate polynomial coefficients for mass flow rate
    m_ref = evalPoly(T_evap - 273.15, T_cond - 273.15, coefficients.loc['m [kg/h]', :]) / 3600
    dm_ref_dT_evap = der_evalPoly_dT_evap(T_evap - 273.15, T_cond - 273.15, coefficients.loc['m [kg/h]', :]) / 3600
    dm_ref_dT_cond = der_evalPoly_dT_cond(T_evap - 273.15, T_cond - 273.15, coefficients.loc['m [kg/h]', :]) / 3600

    Pel = evalPoly(T_evap - 273.15, T_cond - 273.15, coefficients.loc['P [W]', :])
    dPel_dT_evap = der_evalPoly_dT_evap(T_evap - 273.15, T_cond - 273.15, coefficients.loc['P [W]', :])
    dPel_dT_cond = der_evalPoly_dT_cond(T_evap - 273.15, T_cond - 273.15, coefficients.loc['P [W]', :])

    # Compute output mass flow rate and derivatives
    m = m_ref * rho / rho_ref
    dm_dm_ref = rho / rho_ref
    dm_drho = m_ref / rho_ref
    dm_drho_ref = -m_ref * rho / (rho_ref ** 2)

    ho = hi + Pel / m
    dho_dhi = 1.0
    dho_dm = -Pel/(m**2)
    dho_dPel = 1.0/m

    Dm_Dpi = (dm_drho * drho_dpi
             + dm_drho_ref * (drho_ref_dpi + drho_ref_dT_evap * dT_evap_dpi)
             + dm_dm_ref * dm_ref_dT_evap * dT_evap_dpi) * dpi

    Dm_Dpo = (dm_dm_ref * dm_ref_dT_cond * dT_cond_dpo) * dpo
    Dm_Dhi = dm_drho * drho_dhi * dhi

    Dho_Dpi = (dho_dPel * dPel_dT_evap * dT_evap_dpi
              + dho_dm * dm_drho * drho_dpi
              + dho_dm * dm_dm_ref * dm_ref_dT_evap * dT_evap_dpi
              + dho_dm * dm_drho_ref * drho_ref_dpi
              + dho_dm * dm_drho_ref * drho_ref_dT_evap * dT_evap_dpi) * dpi

    Dho_Dpo = (dho_dPel * dPel_dT_cond * dT_cond_dpo
              + dho_dm * dm_dm_ref * dm_ref_dT_cond * dT_cond_dpo) * dpo

    Dho_Dhi = (dho_dhi + dho_dm * dm_drho * drho_dhi) * dhi

    return m, ho, Dm_Dpi, Dm_Dpo, Dm_Dhi, Dho_Dpi, Dho_Dpo, Dho_Dhi


def comp_polynomial_map(pi, po, hi, f, fluid):

    t_evap = coolpropsTPQ(pi, 1, fluid)
    t_cond = coolpropsTPQ(po, 1, fluid)

    SH_ref = 10

    if not f % 5 == 0 and 30 <= f <= 70:
        raise RuntimeError(f'Frequency must be divisible by 5 and between 30 and 70 Hz, but is {f}.')
    directory = os.getcwd()
    source_dir = Path(directory + '/Cycles/OneStageDX/Components/Compressor/Polynomial based Model/Polynomials_4TES-12Y_R134a')
    coefficients = pd.read_csv(
        source_dir / f'10K-UH_{str(f)}Hz.csv',
        sep=';', encoding='latin', skiprows=26, nrows=3, header=0, index_col=0)

    def evalPoly(to, tc, constants):
        Phi = constants.c1 + constants.c2 * to + constants.c3 * tc + constants.c4 * to ** 2 + constants.c5 * to * tc + \
              constants.c6 * tc ** 2 + constants.c7 * to ** 3 + constants.c8 * tc * to ** 2 + constants.c9 * to * tc ** 2 \
              + constants.c10 * tc ** 3

        return Phi

    rho = coolpropsDPH(pi, hi, fluid)
    if abs(coolpropsTPQ(pi, 1, fluid) - (t_evap + SH_ref)) < 1e-4:
        rho_ref = coolpropsDPQ(pi, 1, fluid)
    else:
        rho_ref = coolpropsDPT(pi, t_evap + SH_ref, fluid)

    m_ref = evalPoly(t_evap-273.15, t_cond-273.15, coefficients.loc['m [kg/h]', :]) / 3600

    m = m_ref * rho / rho_ref

    Pel = evalPoly(t_evap-273.15, t_cond-273.15, coefficients.loc['P [W]', :])

    ho = hi + Pel / m

    return m, ho, m

def IHX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, hot_fluid, cold_fluid):
    A = 0.76
    k = 130

    C_h = mi_h * coolpropsCPH(pi_h, hi_h, hot_fluid)
    C_c = mi_c * coolpropsCPH(pi_c, hi_c, cold_fluid)
    T_in_h = coolpropsTPH(pi_h, hi_h, hot_fluid)
    T_in_c = coolpropsTPH(pi_c, hi_c, cold_fluid)

    R = C_h / C_c
    NTU = k * A / C_h
    n = (-1) * NTU * (1 + R)
    if isinstance(n, DualNumber):
        epsilon = (1 - n.exp()) / (1 + R)
    else:
        epsilon = (1 - np.exp(n)) / (1 + R)
    Q = epsilon * C_h * (T_in_h - T_in_c)

    p_out_c = pi_c
    p_out_h = pi_h
    h_out_h = (hi_h - Q / mi_h)
    h_out_c = (hi_c + Q / mi_c)
    m_out_h = mi_h
    m_out_c = mi_c

    return p_out_h, h_out_h, m_out_h, p_out_c, h_out_c, m_out_c


from CoolProp.CoolProp import PropsSI, PhaseSI
import matplotlib.pyplot as plt
import numpy as np
from math import log
import scipy.optimize
from system import *


class HeatExchanger(object):
    """
    Supplemental code for paper:
    I. Bell et al., "A Generalized Moving-Boundary Algorithm to Predict the Heat Transfer Rate of
    Counterflow Heat Exchangers for any Phase Configuration", Applied Thermal Engineering, 2014
    """

    A_c = None  # heat transfer area hot side
    A_h = None  # heat transfer area cold side
    t = 0.003  # wall thickness [m]
    L = 0.5  # heat excahnge Length [m]
    Dh = 0.000866397  # hydraulic diameter [m]
    n = 22  # number of channels
    S = 0.000216787  # Cross sectional area per channel [m2]
    k = 16.7  # heat conductivity [W/mK]

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
        self.T_hi = coolpropsTPH(self.p_hi, self.h_hi, self.Fluid_h)
        self.T_ci = coolpropsTPH(self.p_ci, self.h_ci, self.Fluid_c)

        # Calculate the bubble and dew enthalpies for each stream
        self.T_cbubble = coolpropsTPQ(self.p_ci, 0, self.Fluid_c)
        self.T_hbubble = coolpropsTPQ(self.p_hi, 0, self.Fluid_h)
        self.T_cdew = coolpropsTPQ(self.p_ci, 1, self.Fluid_c)
        self.T_hdew = coolpropsTPQ(self.p_hi, 1, self.Fluid_h)
        self.h_cbubble = coolpropsHTQ(self.T_cbubble, 0, self.Fluid_c)
        self.h_cdew = coolpropsHTQ(self.T_cdew, 1, self.Fluid_c)
        self.h_hbubble = coolpropsHTQ(self.T_hbubble, 0, self.Fluid_h)
        self.h_hdew = coolpropsHTQ(self.T_hdew, 1, self.Fluid_h)
        self.T_cmax = PropsSI("TMAX", self.Fluid_c)
        self.T_hmin = PropsSI("TMIN", self.Fluid_h)

    def external_pinching(self):
        """ Determine the maximum heat transfer rate based on the external pinching analysis """

        if self.T_hmin < self.T_ci:
            self.h_ho = coolpropsHTP(self.T_ci, self.p_hi, self.Fluid_h)
        else:
            self.h_ho = coolpropsHTP(self.T_hmin, self.p_hi, self.Fluid_h)

        if self.T_cmax > self.T_hi:
            self.h_co = coolpropsHTP(self.T_hi, self.p_ci, self.Fluid_c)
        else:
            self.h_co = coolpropsHTP(self.T_cmax, self.p_ci, self.Fluid_c)

        Qmaxh = self.mdot_h * (self.h_hi - self.h_ho)
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
        if self.h_hdew is not None and self.h_hi - 1e-4 > self.h_hdew > self.h_ho + 1e-4:
            self.hvec_h.insert(-1, self.h_hdew)
        if self.h_hbubble is not None and self.h_hi > self.h_hbubble > self.h_ho:
            self.hvec_h.insert(1, self.h_hbubble)

        # Add the bubble and dew enthalpies for the cold stream
        if self.h_cdew is not None and self.h_ci + 1e-4 < self.h_cdew < self.h_co - 1e-4:
            self.hvec_c.insert(-1, self.h_cdew)
        if self.h_cbubble is not None and self.h_ci + 1e-4 < self.h_cbubble < self.h_co - 1e-4:
            self.hvec_c.insert(1, self.h_cbubble)

        # Fill in the complementary cell boundaries
        # Start at the first element in the vector
        k = 0
        while k < len(self.hvec_c) - 1 or k < len(self.hvec_h) - 1:
            if len(self.hvec_c) == 2 and len(self.hvec_h) == 2:
                break

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
            assert (abs(Qcell_hk / Qcell_ck - 1) < 1e-6)

            # Increment index
            k += 1

        # Calculate the temperature at each cell boundary
        self.Tvec_c = [coolpropsTPH(self.p_ci, h, self.Fluid_c) for h in self.hvec_c]

        # Calculate the temperature and entropy at each cell boundary
        self.Tvec_h = [coolpropsTPH(self.p_hi, h, self.Fluid_h) for h in self.hvec_h]

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
                    h_c_pinch = coolpropsHTP(self.T_hdew, self.p_ci, self.Fluid_c)

                    Qright = self.mdot_h * (self.h_hi - self.h_hdew)

                    Qmax = self.mdot_c * (h_c_pinch - self.h_ci) + Qright

                    self.calculate_cell_boundaries(Qmax)

                    return Qmax

        elif stream == 'cold':
            # Check for the cold stream pinch point
            for i in range(1, len(self.hvec_c) - 1):

                # Check if enthalpy is equal to the bubblepoint enthalpy of cold
                # stream and hot stream is colder than cold stream (impossible)
                if (abs(self.hvec_c[i] - self.h_cbubble) < 1e-6
                        and self.Tvec_c[i] > self.Tvec_h[i]):

                    h_h_pinch = coolpropsHTP(self.T_cbubble, self.p_hi, self.Fluid_h)

                    Qleft = self.mdot_c * (self.h_cbubble - self.h_ci)

                    Qmax = Qleft + self.mdot_h * (self.h_hi - h_h_pinch)

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
                    n = abs(DTA / DTB)
                    if isinstance(n, DualNumber):
                        LMTD = (DTA - DTB) / n.log()
                    else:
                        LMTD = (DTA - DTB) / log(n)
                except ValueError as VE:
                    print(Q, DTA, DTB)
                    raise

            UA_req = self.mdot_h * (self.h_hi - self.h_ho) / LMTD
            T_h = (self.Tvec_h[k + 1] + self.Tvec_h[k]) / 2
            h_c = (self.hvec_c[k + 1] + self.hvec_c[k]) / 2
            T_c = (self.Tvec_c[k + 1] + self.Tvec_c[k]) / 2
            T_w = (T_h + T_c) / 2
            alpha_h = 500.0
            alpha_c = 500.0

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


def HX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, hot_fluid, cold_fluid):

    def mapping_func(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, Q):

        po_h = pi_h
        ho_h = hi_h - Q / mi_h
        mo_h = mi_h
        po_c = pi_c
        ho_c = hi_c + Q / mi_c
        mo_c = mi_c

        return np.array([po_h.der, ho_h.der, mo_h.der, po_c.der, ho_c.der, mo_c.der])

    def dF(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, Q):
        Fu = np.array([])
        u = [DualNumber(pi_h.no, 0.0), DualNumber(hi_h.no, 0.0), DualNumber(mi_h.no, 0.0),
             DualNumber(pi_c.no, 0.0),
             DualNumber(hi_c.no, 0.0), DualNumber(mi_c.no, 0.0)]
        for i in range(len(u)):
            u[i].der = 1.0
            if i == 0:
                Fu = mapping_func(u[0], u[1], u[2], u[3], u[4], u[5], Q)
            else:
                Fu = np.column_stack([Fu, mapping_func(u[0], u[1], u[2], u[3], u[4], u[5], Q)])
            u[i].der = 0.0
        Q = DualNumber(Q, 1.0)
        Fx = mapping_func(u[0], u[1], u[2], u[3], u[4], u[5], Q)
        return Fu, Fx

    def dr(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, Q):
        u = [DualNumber(pi_h.no, 0.0), DualNumber(hi_h.no, 0.0), DualNumber(mi_h.no, 0.0),
             DualNumber(pi_c.no, 0.0),
             DualNumber(hi_c.no, 0.0), DualNumber(mi_c.no, 0.0)]
        ru = np.zeros(len(u))
        for i in range(len(u)):
            u[i].der = 1.0
            HX = HeatExchanger(hot_fluid, u[2], u[0], u[1], cold_fluid, u[5], u[3], u[4])
            HX.A_h = HX.A_c = A
            ru[i] = HX.objective_function(Q).der
            u[i].der = 0.0
        HX = HeatExchanger(hot_fluid, u[2], u[0], u[1], cold_fluid, u[5], u[3], u[4])
        HX.A_h = HX.A_c = A
        rx = HX.objective_function(DualNumber(Q, 1.0)).der
        return ru, rx


    A = 3.65

    T_in_h = coolpropsTPH(pi_h, hi_h, hot_fluid)
    T_in_c = coolpropsTPH(pi_c, hi_c, cold_fluid)
    if T_in_h < T_in_c:
        raise RuntimeError('Secondary side fluid is hotter than primary side fluid!')
    else:

        HX = HeatExchanger(hot_fluid, mi_h.no, pi_h.no, hi_h.no, cold_fluid, mi_c.no, pi_c.no, hi_c.no)
        HX.A_h = HX.A_c = A
        HX.run(and_solve=True)
        Q = HX.Q
        ru, rx = dr(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, Q)
        phi = ru / rx
        Fu, Fx = dF(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, Q)
        Df = Fu - np.outer(Fx, phi)

        du = np.array([pi_h.der, hi_h.der, mi_h.der, pi_c.der, hi_c.der, mi_c.der])
        po_h = DualNumber(pi_h.no, np.dot(Df[0, :], du))
        ho_h = DualNumber(hi_h.no - Q / mi_h.no, np.dot(Df[1, :], du))
        mo_h = DualNumber(mi_h.no, np.dot(Df[2, :], du))
        po_c = DualNumber(pi_c.no, np.dot(Df[3, :], du))
        ho_c = DualNumber(hi_c.no + Q / mi_c.no, np.dot(Df[4, :], du))
        mo_c = DualNumber(mi_c.no, np.dot(Df[5, :], du))

    return po_h, ho_h, mo_h, po_c, ho_c, mo_c


# Ti_h_list = np.linspace(20.0, 50.0, 100)
# D = np.zeros([len(Ti_h_list), 2])
# for i, Ti_h in enumerate(Ti_h_list):
#     print(i)
#     hot_fluid = 'Water'
#     cold_fluid = 'R134a'
#     pi_c = DualNumber(2e5, 0.0)
#     mi_c = DualNumber(0.5, 1.0)
#     pi_h = DualNumber(1e5, 0.0)
#     Ti_h = DualNumber(Ti_h+273.15, 0.0)
#     hi_c = DualNumber(PropsSI('H', 'P', pi_c.no, 'Q', 0.5, cold_fluid), 0.0)
#     hi_h = DualNumber(PropsSI('H', 'T', Ti_h.no, 'P', pi_h.no, hot_fluid), 0.0)
#     mi_h = DualNumber(1.0, 0.0)
#
#     ε = 1e-6 * max(abs(hi_c.no), 1.0)
#     po_h, ho_h, mo_h, po_c, ho_c, mo_c = IHX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, hot_fluid, cold_fluid)
#     po_h_fw, ho_h_fw, mo_h_fw, po_c_fw, ho_c_fw, mo_c_fw = IHX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c+ε, hot_fluid, cold_fluid)
#     D[i, 0] = ho_c.der
#     D[i, 1] = (ho_c_fw.no - ho_c.no) / ε


Ti_h_list = np.linspace(20.0, 50.0, 100)
D = np.zeros([len(Ti_h_list), 2])
for i, Ti_h in enumerate(Ti_h_list):
    print(i)
    hot_fluid = 'Water'
    cold_fluid = 'R134a'
    pi_c = DualNumber(2e5, 0.0)
    mi_c = DualNumber(0.5, 1.0)
    pi_h = DualNumber(1e5, 0.0)
    Ti_h = DualNumber(Ti_h+273.15, 0.0)
    hi_c = DualNumber(PropsSI('H', 'P', pi_c.no, 'Q', 0.5, cold_fluid), 0.0)
    hi_h = DualNumber(PropsSI('H', 'T', Ti_h.no, 'P', pi_h.no, hot_fluid), 0.0)
    mi_h = DualNumber(1.0, 0.0)

    ε = 1e-6 * max(abs(hi_c.no), 1.0)
    po_h, ho_h, mo_h, po_c, ho_c, mo_c = HX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c, hot_fluid, cold_fluid)
    po_h_fw, ho_h_fw, mo_h_fw, po_c_fw, ho_c_fw, mo_c_fw = HX(pi_h, hi_h, mi_h, pi_c, hi_c, mi_c+ε, hot_fluid, cold_fluid)
    D[i, 0] = ho_c.der
    D[i, 1] = (ho_c_fw.no - ho_c.no) / ε

error = np.abs(D[:, 0] - D[:, 1])

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the derivatives
ax1.plot(Ti_h_list, D[:, 0], 'r-x', label='Automatic Differentiation')
ax1.plot(Ti_h_list, D[:, 1], 'b-o', label='Finite Difference')
ax1.set_xlabel('pi_c')
ax1.set_ylabel('Derivatives')
ax1.legend()

# Plot the error
ax2.plot(Ti_h_list, error, 'g-^', label='Error (|AutoDiff - FiniteDiff|)')
ax2.set_xlabel('pi_c')
ax2.set_ylabel('Absolute Error')
ax2.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

pi = 2e5
po = 15e5
hi = 4.5e5
f = 50
fluid = 'R134a'
ε = 1e-6
pi_list = np.linspace(1e5, 10e5, 1000)
D = np.zeros([len(pi_list), 2])
for i, pi in enumerate(pi_list):
    print(i)
    ε = 1e-6 * (max(abs(pi), 1.0))
    mi, ho, mo = comp_polynomial_map(DualNumber(pi, 1.0), DualNumber(po, 0.0), DualNumber(hi, 0.0), f, fluid)
    mi_fw, ho_fw, mo_fw = comp_polynomial_map(pi+ε, po, hi, f, fluid)
    D[i, 0] = ho.der
    D[i, 1] = (ho_fw - ho.no) / ε

fig, ax = plt.subplots()
ax.plot(pi_list, D[:, 0], 'r-x', label='automatic differentiation')
ax.plot(pi_list, D[:, 1], 'b-o', label='finite difference')
plt.legend()
plt.show()