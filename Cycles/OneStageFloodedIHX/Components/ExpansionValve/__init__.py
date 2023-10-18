from system import Component, psd
import numpy as np
from CoolProp.CoolProp import PropsSI


def solver(component: [Component]):

    """
    solves expansion valve object using Fixed Orifice Model

    :param    component:   expansion valve object
    :return:  None:        all port states of the expansion valve object gets updated by solution values
    """

    try:

        U = component.parameter['U'].value
        Kv = component.parameter['Kv'].value

        h_in = component.ports[psd['p']].h.value
        p_in = component.ports[psd['p']].p.value
        p_out = component.ports[psd['-p']].p.value
        fluid = component.ports[psd['p']].fluid

        rho = PropsSI('D', 'P', p_in, 'H', h_in, fluid)
        m = Kv * U * np.sqrt((p_in - p_out) * 1e-5 * 1000 * rho) / 3600
        h_out = h_in
        m_in = m_out = m

        component.ports[psd['p']].m.set_value(m_in)
        component.ports[psd['-p']].h.set_value(h_out)
        component.ports[psd['-p']].m.set_value(m_out)

        component.ports[psd['p']].m.set_value(m_in)
        component.ports[psd['-p']].h.set_value(h_out)
        component.ports[psd['-p']].m.set_value(m_out)

    except:
        print(component.name + ' failed!')
        component.status = 0
