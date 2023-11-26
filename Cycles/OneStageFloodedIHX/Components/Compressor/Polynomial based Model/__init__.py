from system import *
from pathlib import Path
import pandas as pd
    

def solver(component: [Component]):
    """
    solves compressor object using a polynomial map

    :param    component:   compressor object
    :return:  None:        all port states of the compressor object gets updated by solution values
    """
    try:

        f = component.parameter['f'].value

        h_in = component.ports[psd['p']].h.value
        p_in = component.ports[psd['p']].p.value
        p_out = component.ports[psd['-p']].p.value
        fluid = component.ports[psd['-p']].fluid

        t_evap = coolpropsTPQ(p_in, 1, fluid) - 273.15
        t_cond =  coolpropsTPQ(p_out, 1, fluid) - 273.15

        # t_evap = PropsSI("T", "P", p_in, "Q", 1, fluid) - 273.15
        # t_cond = PropsSI("T", "P", p_out, "Q", 0, fluid) - 273.15
        SH_ref = 10

        if not f % 5 == 0 and 30 <= f <= 70:
            raise RuntimeError(f'Frequency must be divisible by 5 and between 30 and 70 Hz, but is {f}.')
        directory = component.solver_path
        source_dir = Path(directory + '/Polynomials_4TES-12Y_R134a')
        coefficients = pd.read_csv(
            source_dir / f'10K-UH_{str(f)}Hz.csv',
            sep=';', encoding='latin', skiprows=26, nrows=3, header=0, index_col=0)

        def evalPoly(to, tc, constants):

            Phi = constants.c1 + constants.c2 * to + constants.c3 * tc + constants.c4 * to ** 2 + constants.c5 * to * tc + \
                  constants.c6 * tc ** 2 + constants.c7 * to ** 3 + constants.c8 * tc * to ** 2 + constants.c9 * to * tc ** 2 \
                  + constants.c10 * tc ** 3

            return Phi

        Z = coolpropsDPH(p_in, h_in, fluid) - 273.15
        # Z = PropsSI("D", "P", p_in, "H", h_in, fluid)
        if abs(coolpropsTPQ(p_in, 1, fluid) - t_evap + SH_ref + 273.15) < 1e-4:
        # if abs(PropsSI("T", "P", p_in, "Q", 1.0, fluid) - (t_evap + SH_ref + 273.15)) < 1e-4:
            N = coolpropsDPQ(p_in, 1, fluid)
            # N = PropsSI("D", "P", p_in, "Q", 1.0, fluid)
        else:
            N = coolpropsDPT(p_in, t_evap + SH_ref + 273.15, fluid)
            # N = PropsSI("D", "P", p_in, "T", t_evap + SH_ref + 273.15, fluid)

        m = evalPoly(t_evap, t_cond, coefficients.loc['m [kg/h]', :]) / 3600

        m *= Z / N

        P_el = evalPoly(t_evap, t_cond, coefficients.loc['P [W]', :])

        h_out = h_in + P_el / m

        component.ports[psd['p']].m.set_value(m)
        component.ports[psd['-p']].h.set_value(h_out)
        component.ports[psd['-p']].m.set_value(m)


    except:
        print(component.name + ' ' + ' failed!')
        component.status = 0
