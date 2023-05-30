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

        CA = 1.0e-6

        for port in component.ports:
            if port.port_id[2] == psd['p']:
                p_in = port.p.value
                h_in = port.h.value
            elif port.port_id[2] == psd['-p']:
                p_out = port.p.value

        if component.linearized:
            x = np.array(component.x0.copy())
            i = 0
            for port in component.ports:
                if port.port_typ == 'in' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    x[i+1] = port.h.value
                    i += 2
                elif port.port_typ == 'out' and port.port_id[-1] == 0:
                    x[i] = port.p.value
                    i += 1

        rho = PropsSI('D', 'P', p_in, 'H', h_in, port.fluid)
        if component.linearized:
            F = component.lamda * CA * np.sqrt(rho * (p_in - p_out)) + \
            (1 - component.lamda) * (np.dot(component.J, x - component.x0) + component.F0)
            m = F[0]
        else:
            m = CA * np.sqrt(rho * (p_in - p_out))
        h_out = h_in

        m_in = m
        h_out = h_out
        m_out = m

        for port in component.ports:
            if port.port_id[2] == psd['p']:
                port.m.set_value(m_in)
            elif port.port_id[2] == psd['-p']:
                port.h.set_value(h_out)
                port.m.set_value(m_out)

    except(RuntimeError, ValueError):
        print(component.name + ' failed!')
        component.status = 0
