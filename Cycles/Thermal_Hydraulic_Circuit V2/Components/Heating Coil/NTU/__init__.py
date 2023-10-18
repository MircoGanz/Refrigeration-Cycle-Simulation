import math

from system import Component, psd
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import numpy as np


def solver(component: [Component]):

    """
    solves heat exchanger object using ε-NTU Method

    :param    component:   heat exchanger component object
    :return:  None:        all port states of the heat exchanger object gets updated by solution values
    """

    try:

        UA = component.parameter['UA'].value

        h_in_h = component.ports[psd['h']].h.value
        p_in_h = component.ports[psd['h']].p.value
        m_in_h = component.ports[psd['h']].m.value
        hot_fluid = component.ports[psd['h']].fluid

        h_in_c = component.ports[psd['c']].h.value
        p_in_c = component.ports[psd['c']].p.value
        m_in_c = component.ports[psd['c']].m.value
        cold_fluid = component.ports[psd['c']].fluid

        C_h = m_in_h * PropsSI("C", "P", p_in_h, "H", h_in_h, hot_fluid)
        C_c = m_in_c * PropsSI("C", "P", p_in_c, "H", h_in_c, cold_fluid)
        if C_c < 1e-16 or C_h < 1e-16:
            print()
        T_in_h = PropsSI('T', 'H', h_in_h, 'P', p_in_h, hot_fluid)
        T_in_c = PropsSI('T', 'H', h_in_c, 'P', p_in_c, cold_fluid)

        R = C_h / C_c
        NTU = UA / C_h
        epsilon = 1 - np.exp((np.exp(-R * NTU) - 1) / R)
        Q = epsilon * C_h * (T_in_h - T_in_c)
        p_out_h = p_in_h
        p_out_c = p_in_c
        h_out_h = h_in_h - Q / m_in_h
        h_out_c = h_in_c + Q / m_in_c
        m_out_c = m_in_c
        m_out_h = m_in_h

        component.ports[psd['-h']].p.set_value(p_out_h)
        component.ports[psd['-h']].h.set_value(h_out_h)
        component.ports[psd['-h']].m.set_value(m_out_h)

        component.ports[psd['-c']].p.set_value(p_out_c)
        component.ports[psd['-c']].h.set_value(h_out_c)
        component.ports[psd['-c']].m.set_value(m_out_c)

        component.outputs['Q'].set_value(Q)

    except:
        print(component.name + ' failed!')
        component.status = 0

    if component.diagramm_plot:
        T_out_h = T_in_h - Q / C_h
        T_out_c = T_in_c + Q / C_c
        fig, ax = plt.subplots()
        fig.suptitle(f'{component.name} \n {round(Q / 1000, 3)} kW', fontsize=16)
        ax.plot([T_in_c - 273.15, T_out_c - 273.15], 'b-')
        ax.plot([T_out_h - 273.15, T_in_h - 273.15], 'r-')
        ax.set_xlabel('$A / A_{ges} [-]$')
        ax.set_ylabel('$Temperature [\u00b0 C]$')
        ax.grid(True)
        plt.show()
