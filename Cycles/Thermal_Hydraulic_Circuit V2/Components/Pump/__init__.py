from system import *
    

def solver(component: [Component]):
    """
    solves compressor object using a polynomial map

    :param    component:   compressor object
    :return:  None:        all port states of the compressor object gets updated by solution values
    """
    try:

        k = component.parameter['k'].value
        m0 = component.parameter['m0'].value

        h_in = component.ports[psd['p']].h.value
        p_in = component.ports[psd['p']].p.value
        p_out = component.ports[psd['-p']].p.value
        fluid = component.ports[psd['p']].fluid

        # rho = PropsSI('D', 'H', h_in, 'P', p_in, fluid)
        # m = k * np.sqrt((p_out - p_in) / rho)
        h_out = h_in
        m = m0 - k * (p_out - p_in) * 1e-5

        m_in = m_out = m
        h_out = h_out

        component.ports[psd['p']].m.set_value(m_in)
        component.ports[psd['-p']].m.set_value(m_out)
        component.ports[psd['-p']].h.set_value(h_out)

    except:
        print(component.name + ' ' + ' failed!')
        component.status = 0
