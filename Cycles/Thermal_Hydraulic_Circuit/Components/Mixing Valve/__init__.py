from system import *
from CoolProp.CoolProp import PropsSI
import numpy as np
    

def solver(component: [Component]):
    """
    solves compressor object using a polynomial map

    :param    component:   compressor object
    :return:  None:        all port states of the compressor object gets updated by solution values
    """
    try:

        CA = component.parameter['CA'].value
        ε = component.parameter['eps'].value

        p_in = 2 * []
        h_in = 2 * []
        for port in component.ports:
            if port.port_type == "in":
                p_in.append(port.p.value)
                h_in.append(port.h.value)
            elif port.port_type == "out":
                p_out = port.p.value

        rho_A = PropsSI('D', 'H', h_in[0], 'P', p_in[0], port.fluid)
        rho_B = PropsSI('D', 'H', h_in[1], 'P', p_in[1], port.fluid)
        A = 1.0
        m_A = ε * A * CA * np.sqrt((p_in[0] - p_out) / rho_A)
        m_B = (1 - ε) * A * CA * np.sqrt((p_in[1] - p_out) / rho_B)
        h_out = m_A / (m_A + m_B) * h_in[0] + m_B / (m_A + m_B) * h_in[1]
        m_out = m_A + m_B

        i = 0
        for port in component.ports:
            if port.port_type == "in":
                if i == 0:
                    port.m.set_value(m_A)
                else:
                    port.m.set_value(m_B)
                i += 1
        for port in component.ports:
            if port.port_type == "out":
                port.m.set_value(m_out)
                port.h.set_value(h_out)

    except:
        print(component.name + ' ' + ' failed!')
        component.status = 0
