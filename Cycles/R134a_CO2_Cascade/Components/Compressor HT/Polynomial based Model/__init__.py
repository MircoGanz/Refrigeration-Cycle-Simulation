from system import Component
from CoolProp.CoolProp import PropsSI
from pathlib import Path
import pandas as pd
import numpy as np


def solver(component: [Component]):
    """
    solves compressor object using a polynomial map

    :param    component:   compressor object
    :return:  None:        all port states of the compressor object gets updated by solution values
    """
    try:

        f = component.parameter['f']

        for port in component.ports:
            if port.port_type == "in":
                p_in = port.p.value
                h_in = port.h.value
            elif port.port_type == "out":
                p_out = port.p.value

        if component.linearized:
            x = np.array(component.x0.copy())
            i = 0
            for port in component.ports:
                if port.port_type == 'in' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    x[i+1] = port.h.value
                    i += 2
                elif port.port_type == 'out' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    i += 1

        t_evap = PropsSI("T", "P", p_in, "Q", 1, port.fluid) - 273.15
        t_cond = PropsSI("T", "P", p_out, "Q", 0, port.fluid) - 273.15
        SH_ref = 10

        if not f % 5 == 0 and 30 <= f <= 70:
            raise RuntimeError(f'Frequency must be divisible by 5 and between 30 and 70 Hz, but is {f}.')
        directory = component.solver_path
        source_dir = Path(directory + '/Polynomials_4TES-12Y_R744')
        coefficients = pd.read_csv(
            source_dir / f'10K-UH_{str(f)}Hz.csv',
            sep=';', encoding='latin', skiprows=26, nrows=3, header=0, index_col=0)

        def evalPoly(to, tc, constants):
            # evaluates the polynomial for given evaporation T, condensation T, set of constants

            Phi = constants.c1 + constants.c2 * to + constants.c3 * tc + constants.c4 * to ** 2 + constants.c5 * to * tc + \
                  constants.c6 * tc ** 2 + constants.c7 * to ** 3 + constants.c8 * tc * to ** 2 + constants.c9 * to * tc ** 2 \
                  + constants.c10 * tc ** 3

            return Phi

        Z = PropsSI("D", "P", p_in, "H", h_in, port.fluid)
        if abs(PropsSI("T", "P", p_in, "Q", 1.0, port.fluid) - (t_evap + SH_ref + 273.15)) < 1e-4:
            N = PropsSI("D", "P", p_in, "Q", 1.0, port.fluid)
        else:
            N = PropsSI("D", "P", p_in, "T", t_evap + SH_ref + 273.15, port.fluid)

        m = evalPoly(t_evap, t_cond, coefficients.loc['m [kg/h]', :]) / 3600

        m *= Z / N

        P_el = evalPoly(t_evap, t_cond, coefficients.loc['P [W]', :])

        h_out = h_in + P_el / m

        if component.linearized:
            F = np.dot(component.J, (x - component.x0)) + component.F0
            m_in = component.lamda * m + (1 - component.lamda) * F[0]
            h_out = component.lamda * h_out + (1 - component.lamda) * F[1]
            m_out = component.lamda * m + (1 - component.lamda) * F[2]
        else:
            m_in = m
            h_out = h_out
            m_out = m

        for port in component.ports:
            if port.port_type == "in":
                port.m.set_value(m_in)
        for port in component.ports:
            if port.port_type == "out":
                port.m.set_value(m_out)
                port.h.set_value(h_out)

    except:
        print(component.name + ' failed!')
        component.status = 0
