from system import *
from CoolProp.CoolProp import PropsSI
import numpy as np


def solver(component: [Component]):
    # function calculates Evaporator internal and external outputs
    # input:    Evaporator Object
    # returns:  None

    try:

        A = component.parameter['A'].value
        k = component.parameter['k'].value

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
        T_in_h = PropsSI('T', 'H', h_in_h, 'P', p_in_h, hot_fluid)
        T_in_c = PropsSI('T', 'H', h_in_c, 'P', p_in_c, cold_fluid)

        R = C_h / C_c
        NTU = k * A / C_h
        epsilon = (1 - np.exp(-NTU * (1 + R))) / (1 + R)
        Q = epsilon * C_h * (T_in_h - T_in_c)

        p_out_c = p_in_c
        p_out_h = p_in_h
        h_out_h = (h_in_h - Q / m_in_h)
        h_out_c = (h_in_c + Q / m_in_c)
        m_out_h = m_in_h
        m_out_c = m_in_c

        component.ports[psd['-h']].p.set_value(p_out_h)
        component.ports[psd['-h']].h.set_value(h_out_h)
        component.ports[psd['-h']].m.set_value(m_out_h)

        component.ports[psd['-c']].p.set_value(p_out_c)
        component.ports[psd['-c']].h.set_value(h_out_c)
        component.ports[psd['-c']].m.set_value(m_out_c)

        component.outputs['Q'] = Q

    except (RuntimeError, ValueError):
        print(component.name + 'failed!')
        component.status = 0
