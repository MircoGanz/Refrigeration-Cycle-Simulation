from system import Component


def solver(component: [Component]):

    """
    solves expansion valve object using moving boundary algorithm

    :param    component:   expansion valve object
    :return:  None:        all port states of the expansion valve object gets updated by solution values
    """

    for port in component.ports:
        if port.port_typ == "in":
            h_in = port.h.value
            m_in = port.m.value

    m_out = m_in
    h_out = h_in

    for port in component.ports:
        if port.port_typ == "in":
            port.m.set_value(m_in)
        elif port.port_typ == "out":
            port.m.set_value(m_out)
            port.h.set_value(h_out)
