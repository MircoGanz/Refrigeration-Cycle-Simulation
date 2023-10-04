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

        CA = component.parameter['CA'].value

        h_in = component.ports[psd['p']].h.value
        p_in = component.ports[psd['p']].p.value
        p_out = component.ports[psd['-p']].p.value
        fluid = component.ports[psd['p']].fluid

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

        rho = PropsSI('D', 'P', p_in, 'H', h_in, fluid)
        if component.linearized:
            F = component.lamda * CA * np.sqrt((p_in - p_out) / rho) + \
            (1 - component.lamda) * (np.dot(component.J, x - component.x0) + component.F0)
            m = F[0]
        else:
            m = CA * np.sqrt((p_in - p_out) / rho)
        h_out = h_in

        m_in = m
        h_out = h_out
        m_out = m
        if m < 0:
            print()

        component.ports[psd['p']].m.set_value(m_in)
        component.ports[psd['-p']].h.set_value(h_out)
        component.ports[psd['-p']].m.set_value(m_out)

    except:
        print(component.name + ' failed!')
        component.status = 0
