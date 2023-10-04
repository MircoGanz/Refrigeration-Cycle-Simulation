from multiprocessing.dummy import freeze_support
from system import *
import os
from CoolProp.CoolProp import PropsSI
import pandas as pd


def main():
    # generates JPCM from .csv-file
    directory = os.getcwd()
    data = pd.read_csv(directory + '/Cycles/OneStageDX_V2/JPCM.csv', sep=';', encoding='utf-8-sig',
                       header=None)
    component_typ_list = data.loc[0, 1:len(data.loc[0, 1:]) - 2].to_numpy()
    jpcm = data.loc[1:, 1:].to_numpy().astype(int)

    # defines component types of the cycle
    component_type_list = ['Compressor',
                           'Condenser',
                           'Expansion Valve',
                           'Evaporator',
                           'Source',
                           'Source',
                           'Sink',
                           'Sink']

    component_name_list = ['Compressor',
                           'Condenser',
                           'Expansion Valve',
                           'Evaporator',
                           'Evaporator Source',
                           'Condenser Source',
                           'Evaporator Sink',
                           'Condenser SInk']

    # defines the solver path of each component
    solver_path_list = [directory + '/Cycles/OneStageDX_V2/Components/Compressor/Polynomial based Model',
                        directory + '/Cycles/OneStageDX_V2/Components/Condenser/Moving Boundary Model',
                        directory + '/Cycles/OneStageDX_V2/Components/ExpansionValve',
                        directory + '/Cycles/OneStageDX_V2/Components/Evaporator/Moving Boundary Model']

    # defines boundary type of each component
    component_modeling_type_list = ['PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'Source',
                                    'Source',
                                    'Sink',
                                    'Sink']

    # defines fluids of each fluid loop
    fluid_list = ['R134a', 'Water', 'INCOMP::TY20']

    circuit = Circuit(jpcm=jpcm,
                      component_type_list=component_type_list,
                      component_name_list=component_name_list,
                      component_modeling_type_list=component_modeling_type_list,
                      solver_path_list=solver_path_list,
                      fluid_list=fluid_list)

    circuit.components['Compressor'].add_component_parameter(name='f', value=70)
    circuit.components['Compressor'].add_component_output(name='P')

    circuit.components['Condenser'].add_component_parameter(name='A', value=5.0)
    circuit.components['Condenser'].add_component_output(name='Q')

    circuit.components['Expansion Valve'].add_component_parameter(name='CA', value=5.0e-6, scale_factor=1e5, is_input=True, bounds=(1e-9, 1e3))

    circuit.components['Evaporator'].add_component_parameter(name='A', value=5.0)
    circuit.components['Evaporator'].add_component_output(name='Q')

    pi_v_sec = 2.0
    Ti_v_sec = -2.0
    mi_v_sec = 10.0

    pi_c_sec = 3.0
    Ti_c_sec = 50.0
    mi_c_sec = 10.0

    circuit.components['Evaporator Source'].parameter['p_source'].set_value(pi_v_sec * 1e5)
    circuit.components['Evaporator Source'].parameter['T_source'].set_value(Ti_v_sec + 273.15)
    circuit.components['Evaporator Source'].parameter['m_source'].set_value(mi_v_sec)
    circuit.components['Evaporator Source'].parameter['m_source'].is_input = False
    circuit.components['Evaporator Source'].parameter['m_source'].initial_value = 10.0
    circuit.components['Evaporator Source'].parameter['m_source'].bounds = (0.01, 1000.0)

    circuit.components['Condenser Source'].parameter['p_source'].set_value(pi_c_sec * 1e5)
    circuit.components['Condenser Source'].parameter['T_source'].set_value(Ti_c_sec + 273.15)
    circuit.components['Condenser Source'].parameter['m_source'].set_value(mi_c_sec)
    circuit.components['Condenser Source'].parameter['m_source'].is_input = False
    circuit.components['Condenser Source'].parameter['m_source'].initial_value = 10.0
    circuit.components['Condenser Source'].parameter['m_source'].bounds = (0.01, 10000.0)

    circuit.add_inputs()

    # gets design criteria value from widget input
    SH = 10.0
    SC = 2.0
    pi_co = 2.0
    To_c_sp = 60.0

    # adds design equations to circuit
    circuit.add_design_equa(SuperheatEquation(circuit.components['Evaporator'], SH, 'out', 'h', psd['-c'], relaxed=False))
    circuit.add_design_equa(SubcoolingEquation(circuit.components['Condenser'], SC, 'out', 'h', psd['-h'], relaxed=False))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Compressor'], pi_co * 1e5, 'in', 'p', psd['p'], relaxed=True))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Condenser'], To_c_sp + 273.15, 'out', 'T', psd['-c'], relaxed=False))

    # solver initial values
    p1 = PropsSI('P', 'T', Ti_v_sec + 273.15 - 5.0, 'Q', 1.0, fluid_list[0])
    h1 = PropsSI('H', 'P', p1, 'Q', 1.0, fluid_list[0])
    p2 = PropsSI('P', 'T', Ti_c_sec + 273.15 + 5.0, 'Q', 1.0, fluid_list[0])
    p4 = p1

    init = [p1, h1, p2, p4]
    Vt_bnds = [(0.1e5, 5e5),
               (3e5, 6e5),
               (5.1e5, 30e5),
               (0.1e5, 7e5)]

    # with open('init.pkl', 'rb') as load_data:
    #     init = pickle.load(load_data)

    i = 0
    for var in circuit.Vt:
        var.initial_value = init[i]
        var.bounds = Vt_bnds[i]
        i += 1

    # resets all compontent ports
    [(circuit.components[key].reset(), setattr(circuit.components[key], 'linearized', False)) for key in circuit.components]

    # runs the system solver to solve the system of equations of th cycle
    sol = system_solver(circuit)

    # solution vector of tearing variables
    x = sol['x']

    if not sol['converged']:
        print(sol['message'])

    else:

        circuit.solve(x[0:len(circuit.Vt)])

        # junction values corresponding to the solution
        pi_co, hi_co, mi_co = circuit.components['Compressor'].ports[psd['p']].p.value, circuit.components['Compressor'].ports[psd['p']].h.value, circuit.components['Compressor'].ports[psd['p']].m.value
        pi_c, hi_c, mi_c = circuit.components['Condenser'].ports[psd['h']].p.value, circuit.components['Condenser'].ports[psd['h']].h.value, circuit.components['Condenser'].ports[psd['h']].m.value
        pi_ev, hi_ev, mi_ev = circuit.components['Expansion Valve'].ports[psd['p']].p.value, circuit.components['Expansion Valve'].ports[psd['p']].h.value, circuit.components['Expansion Valve'].ports[psd['p']].m.value
        pi_v, hi_v, mi_v = circuit.components['Evaporator'].ports[psd['c']].p.value, circuit.components['Evaporator'].ports[psd['c']].h.value, circuit.components['Evaporator'].ports[psd['c']].m.value

        Ti_co = PropsSI('T', 'P', pi_v, 'H', hi_v, circuit.components['Compressor'].ports[psd['p']].fluid) - 273.15
        SH = Ti_co - (PropsSI('T', 'P', pi_v, 'Q', 1.0, circuit.components['Compressor'].ports[psd['p']].fluid) - 273.15)

        for key in circuit.components:
            for port in circuit.components[key].ports:
                T = PropsSI('T', 'H', circuit.components[key].ports[port].h.value, 'P', circuit.components[key].ports[port].p.value, circuit.components[key].ports[port].fluid) - 273.15
                print(f'{circuit.components[key].name} Port ID {circuit.components[key].ports[port].port_id}: '
                      f'p = {circuit.components[key].ports[port].p.value * 1e-5} bar, '
                      f'h = {circuit.components[key].ports[port].h.value * 1e-3} kJ/kg,'
                      f' m = {circuit.components[key].ports[port].m.value} kg/s, T = {T} °C \n')

        # plots log ph diagramm
        logph([[hi_co * 1e-3, hi_c * 1e-3, hi_ev * 1e-3, hi_v * 1e-3, hi_co * 1e-3]],
              [[pi_co * 1e-5, pi_c * 1e-5, pi_ev * 1e-5, pi_v * 1e-5, pi_co * 1e-5]],
              [[1, 2, 3, 4, 1]],
              [fluid_list[0]])


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
