from multiprocessing.dummy import freeze_support
from system import *
import os
from CoolProp.CoolProp import PropsSI
import pandas as pd


def main():

    # generates JPCM from .csv-file
    directory = os.getcwd()
    data = pd.read_csv(directory + '/JPCM.csv', sep=';', encoding='utf-8-sig',
                       header=None)
    jpcm = data.loc[1:, 1:].to_numpy().astype(int)

    # defines component types of the cycle
    component_type_list = ['Compressor',
                           'Condenser',
                           'Receiver',
                           'Expansion Valve',
                           'Evaporator',
                           'IHX',
                           'Source',
                           'Source',
                           'Sink',
                           'Sink']

    component_name_list = ['Compressor',
                           'Condenser',
                           'Receiver',
                           'Expansion Valve',
                           'Evaporator',
                           'IHX',
                           'Evaporator Source',
                           'Condenser Source',
                           'Evaporator Sink',
                           'Condenser Sink']

    # defines the solver path of each component
    solver_path_list = [directory + '/Components/Compressor/Polynomial based Model',
                        directory + '/Components/Condenser/Moving Boundary Model V2',
                        directory + " ",
                        directory + '/Components/ExpansionValve',
                        directory + '/Components/Evaporator/Moving Boundary Model',
                        directory + '/Components/IHX/NTU',
                        directory + " ",
                        directory + " ",
                        directory + " ",
                        directory + " "]

    # defines boundary type of each component
    component_modeling_type_list = ['PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'SeparatorComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'MassFlowBasedComponent',
                                    'Source',
                                    'Source',
                                    'Sink',
                                    'Sink']

    # defines fluids of each fluid loop
    fluid_list = ['R134a', 'Water', 'Water']

    circuit = Circuit(jpcm=jpcm,
                      component_type_list=component_type_list,
                      component_name_list=component_name_list,
                      component_modeling_type_list=component_modeling_type_list,
                      solver_path_list=solver_path_list,
                      fluid_list=fluid_list)

    circuit.add_parameter(component_name='Compressor', parameter_name='f', value=30)
    circuit.add_output(component_name='Compressor', output_name='P')

    circuit.add_parameter(component_name='Condenser', parameter_name='A', value=3.65)
    circuit.add_output(component_name='Condenser', output_name='Q')

    circuit.add_parameter(component_name='Expansion Valve', parameter_name='Kv', value=0.25, scale_factor=1e1, bounds=(1e-3, 1e3))
    circuit.add_parameter(component_name='Expansion Valve', parameter_name='U', value=0.5, scale_factor=1e1, is_input=True, bounds=(0.1, 1.0))

    circuit.add_parameter(component_name='Evaporator', parameter_name='A', value=2.65)
    circuit.add_output(component_name='Evaporator', output_name='Q')

    circuit.add_parameter(component_name='IHX', parameter_name='A', value=0.784)
    circuit.add_parameter(component_name='IHX', parameter_name='k', value=123.0)

    pi_v_sec = 2.0
    Ti_v_sec = 10.0
    mi_v_sec = 50 / 60 * 1

    pi_c_sec = 2.0
    Ti_c_sec = 45.0
    mi_c_sec = 70 / 60 * 1

    circuit.set_parameter('Evaporator Source', 'p_source', value=pi_v_sec * 1e5)
    # circuit.set_port_value('Evaporator', port_name='h', var_type='p', value=pi_v_sec * 1e5)
    circuit.set_parameter('Evaporator Source', 'T_source', value=Ti_v_sec + 273.15)
    # circuit.set_port_value('Evaporator', port_name='h', var_type='h', value=Ti_v_sec - 273.15)
    circuit.set_parameter('Evaporator Source', 'm_source', value=mi_v_sec, is_input=False, initial_value=mi_v_sec, bounds=(0.0001, 10000.0))
    # circuit.set_port_value('Evaporator', port_name='h', var_type='m', value=mi_v_sec)

    circuit.set_parameter('Condenser Source', 'p_source', value=pi_c_sec * 1e5)
    circuit.set_parameter('Condenser Source', 'T_source', value=Ti_c_sec + 273.15)
    circuit.set_parameter('Condenser Source', 'm_source', value=mi_c_sec, is_input=False, initial_value=mi_c_sec, bounds=(0.0001, 10000.0))

    # gets design criteria value from widget input
    SH = 10.0
    SC = 2.0
    po_co = 11.0
    To_c_sp = 50.0

    # adds design equations to circuit
    circuit.add_design_equa(name='Superheat Equation', design_equa=SuperheatEquation(circuit.components['Evaporator'], SH, 'out', 'h', psd['-c'], relaxed=False))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Compressor'], po_co * 1e5, 'out', 'p', psd['-p'], relaxed=False))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Condenser'], To_c_sp + 273.15, 'out', 'T', psd['-c'], relaxed=False))

    # solver initial values
    p1 = PropsSI('P', 'T', Ti_v_sec + 273.15 - 5.0, 'Q', 1.0, fluid_list[0])
    h1 = PropsSI('H', 'P', p1, 'Q', 1.0, fluid_list[0])
    p2 = PropsSI('P', 'T', Ti_c_sec + 273.15 + 5.0, 'Q', 1.0, fluid_list[0])

    # solver initial values
    p1 = PropsSI('P', 'T', Ti_v_sec + 273.15 - 5.0, 'Q', 1.0, fluid_list[0])
    h2 = PropsSI('H', 'P', p2, 'Q', 0.0, fluid_list[0])
    p2 = PropsSI('P', 'T', Ti_c_sec + 273.15 + 5.0, 'Q', 1.0, fluid_list[0])

    init = [p1, h1, p2, p2, h2, p1]
    Vt_bnds = [(0.1e5, 5e5),
               (3e5, 6e5),
               (5.1e5, 30e5),
               (5.1e5, 30e5),
               (1e5, 3e5),
               (0.1e5, 5e5)]

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
        pi_co, hi_co, mi_co = circuit.components['Compressor'].ports[psd['p']].p.value, \
                              circuit.components['Compressor'].ports[psd['p']].h.value, \
                              circuit.components['Compressor'].ports[psd['p']].m.value

        pi_c, hi_c, mi_c = circuit.components['Condenser'].ports[psd['h']].p.value, \
                           circuit.components['Condenser'].ports[psd['h']].h.value, \
                           circuit.components['Condenser'].ports[psd['h']].m.value

        pi_r, hi_r, mi_r = circuit.components['Receiver'].ports[psd['tp']].p.value, \
                           circuit.components['Receiver'].ports[psd['tp']].h.value, \
                           circuit.components['Receiver'].ports[psd['tp']].m.value

        pi_ihx_h, hi_ihx_h, mi_ihx_h = circuit.components['IHX'].ports[psd['h']].p.value, \
                                       circuit.components['IHX'].ports[psd['h']].h.value, \
                                       circuit.components['IHX'].ports[psd['h']].m.value

        pi_ev, hi_ev, mi_ev = circuit.components['Expansion Valve'].ports[psd['p']].p.value, \
                              circuit.components['Expansion Valve'].ports[psd['p']].h.value, \
                              circuit.components['Expansion Valve'].ports[psd['p']].m.value

        pi_v, hi_v, mi_v = circuit.components['Evaporator'].ports[psd['c']].p.value, \
                           circuit.components['Evaporator'].ports[psd['c']].h.value, \
                           circuit.components['Evaporator'].ports[psd['c']].m.value

        pi_ihx_c, hi_ihx_c, mi_ihx_c = circuit.components['IHX'].ports[psd['c']].p.value, \
                                       circuit.components['IHX'].ports[psd['c']].h.value, \
                                       circuit.components['IHX'].ports[psd['c']].m.value

        for key in circuit.components:
            print(f'{circuit.components[key].name}:')
            for port in circuit.components[key].ports:
                T = PropsSI('T', 'H', circuit.components[key].ports[port].h.value, 'P', circuit.components[key].ports[port].p.value, circuit.components[key].ports[port].fluid) - 273.15
                print(f'Port ID {circuit.components[key].ports[port].port_id}: '
                      f'p = {round(circuit.components[key].ports[port].p.value * 1e-5, 3)} bar, '
                      f'T = {round(T, 3)} °C, '
                      f'h = {round(circuit.components[key].ports[port].h.value * 1e-3, 3)} kJ/kg, '
                      f'm = {round(circuit.components[key].ports[port].m.value, 3)} kg/s')
            for param in circuit.components[key].parameter:
                print(f'{param}: {round(circuit.components[key].parameter[param].value, 3)}')
            print('\n')

        # plots log ph diagramm
        logph([[hi_co * 1e-3, hi_c * 1e-3, hi_r * 1e-3, hi_ihx_h * 1e-3, hi_ev * 1e-3, hi_v * 1e-3, hi_ihx_c * 1e-3, hi_co * 1e-3]],
              [[pi_co * 1e-5, pi_c * 1e-5, pi_r * 1e-5, pi_ihx_h * 1e-5, pi_ev * 1e-5, pi_v * 1e-5, pi_ihx_c * 1e-5, pi_co * 1e-5]],
              [[1, 2, 3, 4, 5, 6, 7, 1]],
              [fluid_list[0]])


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
