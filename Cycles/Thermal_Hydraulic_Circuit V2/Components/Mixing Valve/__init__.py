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

        Kv = component.parameter['Kv'].value
        U = component.parameter['U'].value

        p_in_A = component.ports[psd['A']].p.value
        h_in_A = component.ports[psd['A']].h.value
        fluid_A = component.ports[psd['A']].fluid
        p_in_B = component.ports[psd['B']].p.value
        h_in_B = component.ports[psd['B']].h.value
        fluid_B = component.ports[psd['B']].fluid
        p_out = component.ports[psd['-C']].p.value

        rho_A = PropsSI('D', 'H', h_in_A, 'P', p_in_A, fluid_A)
        rho_B = PropsSI('D', 'H', h_in_A, 'P', p_in_B, fluid_B)
        A = 1.0
        # m_A = eps * A * CA * np.sqrt((p_in_A - p_out) / rho_A)
        m_A = Kv * U * np.sqrt((p_in_A - p_out) * 1e-5 * 1000 * rho_A) / 3600
        # m_B = (1 - eps) * A * CA * np.sqrt((p_in_B - p_out) / rho_B)
        m_B = Kv * U * np.sqrt((p_in_B - p_out) * 1e-5 * 1000 * rho_B) / 3600
        h_out = m_A / (m_A + m_B) * h_in_A + m_B / (m_A + m_B) * h_in_B
        m_out = m_A + m_B

        component.ports[psd['-C']].h.set_value(h_out)
        component.ports[psd['-C']].m.set_value(m_out)
        component.ports[psd['A']].m.set_value(m_A)
        component.ports[psd['B']].m.set_value(m_B)

    except:
        print(component.name + ' ' + ' failed!')
        component.status = 0
