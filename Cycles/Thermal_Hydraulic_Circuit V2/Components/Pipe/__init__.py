from system import *
from CoolProp.CoolProp import PropsSI
import numpy as np
    

def solver(component: [Component]):
    """
    solves valve object

    :param    component:   pipe object
    :return:  None:        all port states of the valve object gets updated by solution values
    """
    try:

        CA = component.parameter['CA'].value

        p_in = component.ports[psd['p']].p.value
        h_in = component.ports[psd['p']].h.value
        m_in = component.ports[psd['p']].m.value

        p_out = p_in - CA * abs(m_in)
        h_out = h_in
        m_out = m_in

        component.ports[psd['-p']].p.set_value(p_out)
        component.ports[psd['-p']].h.set_value(h_out)
        component.ports[psd['-p']].m.set_value(m_out)

    except:
        print(component.name + ' ' + ' failed!')
        component.status = 0
