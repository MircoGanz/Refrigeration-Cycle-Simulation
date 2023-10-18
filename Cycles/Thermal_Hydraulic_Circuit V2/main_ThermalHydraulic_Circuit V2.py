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
                           'Separator',
                           'Expansion Valve',
                           'Evaporator',
                           'Pump',
                           'Heat Exchanger',
                           'Mixing Valve',
                           'Pump',
                           'Valve',
                           'Pipe',
                           'Valve',
                           'Heat Exchanger',
                           'Heat Exchanger',
                           'Mixing Valve',
                           'Source',
                           'Sink',
                           'Source',
                           'Sink',
                           'Source',
                           'Sink']

    component_name_list = ['Compressor',
                           'Condenser',
                           'Separator',
                           'Expansion Valve',
                           'Evaporator',
                           'Pump 1',
                           'Cooling Coil',
                           'Mixing Valve 1',
                           'Pump 2',
                           'Valve 1',
                           'Pipe',
                           'Valve 2',
                           'Heating Coil 1',
                           'Heating Coil 2',
                           'Mixing Valve 2',
                           'Source Cooling Coil',
                           'Sink Cooling Coil',
                           'Source Heating Coil 1',
                           'Sink Heating Coil 1',
                           'Source Heating Coil 2',
                           'Sink Heating Coil 2']

    # defines the solver path of each component
    solver_path_list = [directory + '/Components/Compressor/Polynomial based Model',
                        directory + '/Components/Condenser/Moving Boundary Model V2',
                        directory + ' ',
                        directory + '/Components/ExpansionValve',
                        directory + '/Components/Evaporator/Moving Boundary Model',
                        directory + '/Components/Pump',
                        directory + '/Components/Cooling Coil/NTU',
                        directory + '/Components/Mixing Valve',
                        directory + '/Components/Pump',
                        directory + '/Components/Valve',
                        directory + '/Components/Pipe',
                        directory + '/Components/Valve',
                        directory + '/Components/Heating Coil/NTU',
                        directory + '/Components/Heating Coil/NTU',
                        directory + '/Components/Mixing Valve',
                        directory + ' ',
                        directory + ' ',
                        directory + ' ',
                        directory + ' ',
                        directory + ' ',
                        directory + ' ']

    # defines boundary type of each component
    component_modeling_type_list = ['PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'SeparatorComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'PressureBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'Source',
                                    'Sink',
                                    'Source',
                                    'Sink',
                                    'Source',
                                    'Sink']

    # defines fluids of each fluid loop
    fluid_list = ['R134a', 'INCOMP::TY20', 'Water', 'INCOMP::TY20', 'Water', 'Water']

    circuit = Circuit(jpcm=jpcm,
                      component_type_list=component_type_list,
                      component_name_list=component_name_list,
                      component_modeling_type_list=component_modeling_type_list,
                      solver_path_list=solver_path_list,
                      fluid_list=fluid_list)

    circuit.add_parameter(component_name='Compressor', parameter_name='f', value=70)
    circuit.add_output(component_name='Compressor', output_name='P')

    circuit.add_parameter(component_name='Condenser', parameter_name='A', value=3.65)
    circuit.add_output(component_name='Condenser', output_name='Q')

    circuit.add_parameter(component_name='Expansion Valve', parameter_name='Kv', value=0.25)
    circuit.add_parameter(component_name='Expansion Valve', parameter_name='U', value=0.5, scale_factor=1e1, is_input=True, bounds=(0.1, 1.0))

    circuit.add_parameter(component_name='Evaporator', parameter_name='A', value=2.65)
    circuit.add_output(component_name='Evaporator', output_name='Q')

    circuit.add_parameter(component_name='Pump 1', parameter_name='k', value=0.01, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e1))
    circuit.add_parameter(component_name='Pump 1', parameter_name='m0', value=10.0)

    circuit.add_parameter(component_name='Cooling Coil', parameter_name='UA', value=10000.0)
    circuit.add_output(component_name='Cooling Coil', output_name='Q')

    circuit.add_parameter(component_name='Pump 2', parameter_name='k', value=0.01, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e1))
    circuit.add_parameter(component_name='Pump 2', parameter_name='m0', value=10.0)

    circuit.add_parameter(component_name='Heating Coil 1', parameter_name='UA', value=10000.0)
    circuit.add_output(component_name='Heating Coil 1', output_name='Q')

    circuit.add_parameter(component_name='Heating Coil 2', parameter_name='UA', value=10000.0)
    circuit.add_output(component_name='Heating Coil 2', output_name='Q')

    circuit.add_parameter(component_name='Mixing Valve 1', parameter_name='Kv', value=25.0)
    circuit.add_parameter(component_name='Mixing Valve 1', parameter_name='U', value=0.5, scale_factor=1e3, is_input=True, bounds=(0.1, 1.0))

    circuit.add_parameter(component_name='Mixing Valve 2', parameter_name='Kv', value=25.0)
    circuit.add_parameter(component_name='Mixing Valve 2', parameter_name='U', value=0.9, scale_factor=1.0, is_input=True, bounds=(0.1, 1.0))

    circuit.add_parameter(component_name='Valve 1', parameter_name='Kv', value=25.0)
    circuit.add_parameter(component_name='Valve 1', parameter_name='U', value=0.9, scale_factor=1.0, is_input=True, bounds=(0.001, 1.0))

    circuit.add_parameter(component_name='Valve 2', parameter_name='Kv', value=25.0)
    circuit.add_parameter(component_name='Valve 2', parameter_name='U', value=0.9, scale_factor=1.0, is_input=True, bounds=(0.001, 1.0))

    circuit.add_parameter(component_name='Pipe', parameter_name='CA', value=5000.0)

    pi_cc_sec = 2.0
    Ti_cc_sec = 5.0
    mi_cc_sec = 18.0

    pi_hc1_sec = 1.0
    Ti_hc1_sec = 40.0
    mi_hc1_sec = 1.0

    pi_hc2_sec = 1.0
    Ti_hc2_sec = 40.0
    mi_hc2_sec = 1.0

    circuit.set_parameter('Source Cooling Coil', 'p_source', value=pi_cc_sec * 1e5)
    circuit.set_parameter('Source Cooling Coil', 'T_source', value=Ti_cc_sec + 273.15)
    circuit.set_parameter('Source Cooling Coil', 'm_source', value=mi_cc_sec,
                          is_input=True, initial_value=mi_cc_sec, bounds=(0.01, 1000.0))

    circuit.set_parameter('Source Heating Coil 1', 'p_source', value=pi_hc1_sec * 1e5)
    circuit.set_parameter('Source Heating Coil 1', 'T_source', value=Ti_hc1_sec + 273.15)
    circuit.set_parameter('Source Heating Coil 1', 'm_source', value=mi_hc1_sec,
                          is_input=True, initial_value=mi_hc1_sec, bounds=(0.01, 1000.0))

    circuit.set_parameter('Source Heating Coil 2', 'p_source', value=pi_hc2_sec * 1e5)
    circuit.set_parameter('Source Heating Coil 2', 'T_source', value=Ti_hc2_sec + 273.15)
    circuit.set_parameter('Source Heating Coil 2', 'm_source', value=mi_hc2_sec,
                          is_input=True, initial_value=mi_hc2_sec, bounds=(0.01, 1000.0))

    # gets design criteria value from widget input
    SH = 5.0
    SC = 2.0
    T6_6sp = -5.0
    T10_9sp = 55.0

    TVL_HC1 = 59.0
    TVL_HC2 = 59.0

    # adds design equations to circuit
    circuit.add_design_equa(name='Superheat Equation',
                            design_equa=SuperheatEquation(circuit.components['Evaporator'],
                                                          DC_value=SH,
                                                          port_type='out',
                                                          var_type='h',
                                                          port_id=psd['-c'],
                                                          relaxed=False))

    circuit.add_design_equa(name='TVL_HC1 Equation',
                            design_equa=DesignParameterEquation(circuit.components['Heating Coil 1'],
                                                                DC_value=TVL_HC1 + 273.15,
                                                                port_type='in',
                                                                var_type='T',
                                                                port_id=psd['-c'],
                                                                relaxed=False))
    circuit.add_design_equa(name='TVL_HC2 Equation',
                            design_equa=DesignParameterEquation(circuit.components['Heating Coil 2'],
                                                                DC_value=TVL_HC2 + 273.15,
                                                                port_type='in',
                                                                var_type='T',
                                                                port_id=psd['-c'],
                                                                relaxed=False))
    circuit.add_design_equa(name='HC1 Q Equation',
                            design_equa=OutputDesignEquation(circuit.components['Heating Coil 1'],
                                                             DC_value=10000,
                                                             output_name='Q',
                                                             scale_factor=1e-5,
                                                             relaxed=False))
    circuit.add_design_equa(name='HC2 Q Equation',
                            design_equa=OutputDesignEquation(circuit.components['Heating Coil 2'],
                                                             DC_value=10000,
                                                             output_name='Q',
                                                             scale_factor=1e-5,
                                                             relaxed=False))
    # solver initial values
    p1_1 = PropsSI('P', 'T', T6_6sp + 273.15 - 5.0, 'Q', 1.0, fluid_list[0])
    h1_1 = 4.0e5
    p2_1 = PropsSI('P', 'T', T10_9sp + 273.15 + 5.0, 'Q', 1.0, fluid_list[0])
    p6_6 = 1.0e5
    h6_6 = PropsSI('H', 'T', T6_6sp + 273.15, 'P', p6_6, fluid_list[1])
    p7_6 = 1.1e5
    p10_9 = 1.0e5
    h10_9 = PropsSI('H', 'T', T10_9sp + 273.15, 'P', p10_9, fluid_list[2])
    p11_9 = 1.15e5
    p5_4 = p1_1
    p14_10 = p11_9 - 1e3
    h16_15 = h10_9
    p15_12 = p11_9 - 1e3
    m8_7 = 10.0

    init = [p1_1, h1_1, p2_1, p6_6, h6_6, p7_6, p10_9, h10_9, p11_9, p5_4, p14_10, h16_15, p15_12, m8_7]

    Vt_bnds = [(0.1e5, 5e5),
               (3e5, 6e5),
               (5.1e5, 30e5),
               (0.1e5, 1e5),
               (-1e5, -1e4),
               (5.5e5, 10e5),
               (0.1e5, 1e5),
               (1e5, 5e6),
               (5.5e5, 10e5),
               (0.1e5, 5.0e5),
               (2.5e5, 5.0e5),
               (1e4, 1e6),
               (2.5e5, 5.0e5),
               (0.01, 1000.0)]

    with open('init.pkl', 'rb') as load_data:
        init = pickle.load(load_data)

    i = 0
    for var in circuit.Vt:
        var.initial_value = init[i]
        var.bounds = Vt_bnds[i]
        i += 1
    for var in circuit.U:
        var.initial_value = init[i]
        i += 1
    for var in circuit.S:
        var.initial_value = init[i]
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
        pi_se, hi_se, mi_se = circuit.components['Separator'].ports[psd['tp']].p.value, circuit.components['Separator'].ports[psd['tp']].h.value, circuit.components['Separator'].ports[psd['tp']].m.value
        pi_ev, hi_ev, mi_ev = circuit.components['Expansion Valve'].ports[psd['p']].p.value, circuit.components['Expansion Valve'].ports[psd['p']].h.value, circuit.components['Expansion Valve'].ports[psd['p']].m.value
        pi_v, hi_v, mi_v = circuit.components['Evaporator'].ports[psd['c']].p.value, circuit.components['Evaporator'].ports[psd['c']].h.value, circuit.components['Evaporator'].ports[psd['c']].m.value

        Ti_co = PropsSI('T', 'P', pi_co, 'H', hi_co, fluid_list[0]) - 273.15
        SH = Ti_co - (PropsSI('T', 'P', pi_v, 'Q', 1.0, fluid_list[0]) - 273.15)

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
        logph([[hi_co * 1e-3, hi_c * 1e-3, hi_se * 1e-3, hi_ev * 1e-3, hi_v * 1e-3, hi_co * 1e-3]],
              [[pi_co * 1e-5, pi_c * 1e-5, pi_se * 1e-5, pi_ev * 1e-5, pi_v * 1e-5, pi_co * 1e-5]],
              [[1, 2, 3, 4, 5, 1]],
              [fluid_list[0]])


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
