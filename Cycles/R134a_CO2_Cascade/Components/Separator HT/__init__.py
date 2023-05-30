from system import Component, psd
from CoolProp.CoolProp import PropsSI
import numpy as np


def solver(component: [Component]):

    """
    solves separator object

    :param    component:   separator component object
    :return:  None:        all port states of the separator object gets updated by solution values
    """

    try:
        for port in component.ports:
            if port.port_id[2] == psd['p']:
                fluid = port.fluid
                h_in = port.h.value
                p_in = port.p.value
                m_in = port.m.value

        h_l = PropsSI('H', 'P', p_in, 'Q', 0.0, fluid)
        if h_in < h_l - 1e-1:
            h_out = h_in
        else:
            h_out = h_l

        if component.linearized:
            h_out = component.lamda * h_out + (1 - component.lamda) * h_in
        m_out = m_in

        for port in component.ports:
            if port.port_id[2] == psd['-p']:
                port.p.set_value(p_in)
                port.h.set_value(h_out)
                port.m.set_value(m_out)

    except (RuntimeError, ValueError):
        print(component.component_typ + 'failed!')
        component.status = 0
