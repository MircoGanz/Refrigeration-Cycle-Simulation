from system import Component
from CoolProp.CoolProp import PropsSI
import numpy as np
    

def solver(component: [Component]):
    """
    solves compressor object using a polynomial map

    :param    component:   compressor object
    :return:  None:        all port states of the compressor object gets updated by solution values
    """
    try:

        k = component.parameter['k'].value
        m0 = component.parameter['m0'].value

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

        rho = PropsSI('D', 'H', h_in, 'P', p_in, port.fluid)
        # m = max(m0 - k * (p_out - p_in), 0.00001)
        m = k * np.sqrt((p_out - p_in) / rho)
        h_out = h_in

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
        print(component.name + ' ' + ' failed!')
        component.status = 0
