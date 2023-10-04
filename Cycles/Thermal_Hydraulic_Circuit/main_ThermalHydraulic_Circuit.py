from multiprocessing.dummy import freeze_support
from system import *
import os
from CoolProp.CoolProp import PropsSI
import pandas as pd


def main():
    # generates JPCM from .csv-file
    directory = os.getcwd()
    data = pd.read_csv(directory + '/Cycles/Thermal_Hydraulic_Circuit/JPCM.csv', sep=';', encoding='utf-8-sig',
                       header=None)
    component_typ_list = data.loc[0, 1:len(data.loc[0, 1:]) - 2].to_numpy()
    jpcm = data.loc[1:, 1:].to_numpy().astype(int)

    # defines component types of the cycle
    component_type_list = ['Compressor',
                           'Condenser',
                           'Expansion Valve',
                           'Evaporator',
                           'Pump',
                           'Heat Exchanger',
                           'Pump',
                           'Heat Exchanger',
                           'Heat Exchanger',
                           'Mixing Valve',
                           'Mixing Valve',
                           'Mixing Valve',
                           'Mixing Valve']

    component_name_list = ['Compressor',
                           'Condenser',
                           'Expansion Valve',
                           'Evaporator',
                           'Pump 1',
                           'Cooling Coil',
                           'Pump 2',
                           'Heating Coil 1',
                           'Heating Coil 2',
                           'Mixing Valve 1',
                           'Mixing Valve 2',
                           'Mixing Valve 3',
                           'Mixing Valve 4']

    # defines the solver path of each component
    solver_path_list = [directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Compressor/Polynomial based Model',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Condenser/Moving Boundary Model',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/ExpansionValve',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Evaporator/Moving Boundary Model',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Pump 1',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Cooling Coil/NTU',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Pump 2',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Heating Coil 1/NTU',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Heating Coil 2/NTU',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Mixing Valve',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Mixing Valve',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Mixing Valve',
                        directory + '/Cycles/Thermal_Hydraulic_Circuit V2/Components/Mixing Valve']

    # defines boundary type of each component
    component_modeling_type_list = ['PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'MassFlowBasedComponent',
                                    'MassFlowBasedComponent',
                                    'PressureBasedComponent',
                                    'PressureBasedComponent',
                                    'PressureBasedComponent',
                                    'PressureBasedComponent']

    # defines fluids of each fluid loop
    fluid_list = ['R134a', 'INCOMP::TY20', 'Water', 'INCOMP::TY20', 'Water', 'Water']

    circuit = Circuit(jpcm=jpcm,
                      component_type_list=component_type_list,
                      component_name_list=component_name_list,
                      component_modeling_type_list=component_modeling_type_list,
                      solver_path_list=solver_path_list,
                      fluid_list=fluid_list)

    circuit.components['Compressor'].add_component_parameter(name='f', value=50)
    circuit.components['Compressor'].add_component_output(name='P')

    circuit.components['Condenser'].add_component_parameter(name='A', value=5.0)
    circuit.components['Condenser'].add_component_output(name='Q')

    circuit.components['Expansion Valve'].add_component_parameter(name='CA', value=1.0e-5, scale_factor=1e5, is_input=False, bounds=(1e-9, 1e3))

    circuit.components['Evaporator'].add_component_parameter(name='A', value=5.0)
    circuit.components['Evaporator'].add_component_output(name='Q')

    circuit.components['Pump 1'].add_component_parameter(name='k', value=1e-1, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Pump 1'].add_component_parameter(name='m0', value=100.0)

    circuit.components['Cooling Coil'].add_component_parameter(name='UA', value=10000.0)
    circuit.components['Cooling Coil'].add_component_output(name='Q')

    circuit.components['Pump 2'].add_component_parameter(name='k', value=1e-1, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Pump 2'].add_component_parameter(name='m0', value=100.0)

    circuit.components['Heating Coil 1'].add_component_parameter(name='UA', value=10000.0)
    circuit.components['Heating Coil 1'].add_component_output(name='Q')

    circuit.components['Heating Coil 2'].add_component_parameter(name='UA', value=10000.0)
    circuit.components['Heating Coil 2'].add_component_output(name='Q')

    circuit.components['Mixing Valve 1'].add_component_parameter(name='CA', value=1e-1, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Mixing Valve 1'].add_component_parameter(name='eps', value=0.5, scale_factor=1.0, is_input=True, bounds=(0.0, 0.9))

    circuit.components['Mixing Valve 2'].add_component_parameter(name='CA', value=1e-1/2, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Mixing Valve 2'].add_component_parameter(name='eps', value=0.5, scale_factor=1.0, is_input=True, bounds=(0.0, 0.9))

    circuit.components['Mixing Valve 3'].add_component_parameter(name='CA', value=1e-1/4, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Mixing Valve 3'].add_component_parameter(name='eps', value=0.5, scale_factor=1.0, is_input=True, bounds=(0.0, 0.9))

    circuit.components['Mixing Valve 4'].add_component_parameter(name='CA', value=1e-1/4, scale_factor=1e1, is_input=True, bounds=(1e-5, 1e5))
    circuit.components['Mixing Valve 4'].add_component_parameter(name='eps', value=0.5, scale_factor=1.0, is_input=True, bounds=(0.0, 0.9))

    for port in circuit.components['Cooling Coil'].ports:
        if port.port_id[2] == psd['h']:
            port.m.is_input = False
            port.m.initial_value = 10.0
            port.m.bounds = (0.00001, 100.0)
        else:
            pass

    for port in circuit.components['Heating Coil 1'].ports:
        if port.port_id[2] == psd['c']:
            port.m.is_input = False
            port.m.initial_value = 10.0
            port.m.bounds = (0.00001, 100.0)
        else:
            pass

    for port in circuit.components['Heating Coil 2'].ports:
        if port.port_id[2] == psd['c']:
            port.m.is_input = False
            port.m.initial_value = 10.0
            port.m.bounds = (0.00001, 100.0)
        else:
            pass

    circuit.add_inputs()

    ps1 = 1.0
    Ts1 = -2.0
    ms1 = 1.0

    ps3 = 1.0
    Ts3 = 35.0
    ms3 = 1.0

    ps5 = 1.0
    Ts5 = 35.0
    ms5 = 1.0

    # gets design criteria value from widget input
    SH = 4.5
    SC = 2.0
    p7sp = 1.0
    p9sp = 1.0
    T8sp = -5.0
    T10sp = 60.0

    # # sets boundary condition values to corresponding components
    circuit.components['Cooling Coil'].set_boundary_port_values(ps1 * 1e5, PropsSI('H', 'P', ps1 * 1e5, 'T', Ts1 + 273.15, fluid_list[3]), ms1)
    circuit.components['Heating Coil 1'].set_boundary_port_values(ps3 * 1e5, PropsSI('H', 'P', ps3 * 1e5, 'T', Ts3 + 273.15, fluid_list[4]), ms3)
    circuit.components['Heating Coil 2'].set_boundary_port_values(ps5 * 1e5, PropsSI('H', 'P', ps5 * 1e5, 'T', Ts5 + 273.15, fluid_list[5]), ms5)

    # adds design equations to circuit
    circuit.add_design_equa(SuperheatEquation(circuit.components['Evaporator'], SH, 'out', 'h', psd['-c'], relaxed=True))
    circuit.add_design_equa(SubcoolingEquation(circuit.components['Condenser'], SC, 'out', 'h', psd['-h'], relaxed=True))
    circuit.add_design_equa(DesignParameterEquation(circuit.components['Pump 1'], p7sp * 1e5, 'in', 'p', psd['p'], relaxed=True))
    circuit.add_design_equa(DesignParameterEquation(circuit.components['Pump 2'], p9sp * 1e5, 'in', 'p', psd['p'], relaxed=True))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Evaporator'], T8sp, 'in', 'T', psd['h'], relaxed=True))
    # circuit.add_design_equa(DesignParameterEquation(circuit.components['Condenser'], T10sp, 'in', 'T', psd['c'], relaxed=True))

    # solver initial values
    p1 = PropsSI('P', 'T', T8sp + 273.15 - 5.0, 'Q', 1.0, fluid_list[0])
    h1 = 4.0e5
    p2 = PropsSI('P', 'T', T10sp + 273.15 + 5.0, 'Q', 1.0, fluid_list[0])
    p7 = 1.0e5
    h7 = PropsSI('H', 'T', T8sp + 273.15, 'P', p7, fluid_list[1])
    p8 = 1.1e5
    p9 = 1.0e5
    h9 = PropsSI('H', 'T', T10sp + 273.15, 'P', p9, fluid_list[2])
    p10 = 1.15e5
    p4 = p1
    m5_10 = 0.1
    m11_8 = 0.1
    m11_9 = 0.1

    init = [p1, h1, p2, p7, h7, p8, p9, h9, p10, p4, m5_10, m11_8, m11_9, p9]
    Vt_bnds = [(0.1e5, 7e5),
               (3e5, 6e5),
               (7.1e5, 30e5),
               (0.1e5, 1e5),
               (-1.5e6, -1e4),
               (1.01e5, 10e5),
               (0.1e5, 1e5),
               (1e4, 1e6),
               (1.01e5, 10e5),
               (0.1e5, 5.0e5),
               (0.000001, 100.0),
               (0.000001, 100.0),
               (0.000001, 100.0),
               (1.01e5, 1.001e5)]

    # with open('init.pkl', 'rb') as load_data:
    #     init = pickle.load(load_data)

    i = 0
    for var in circuit.Vt:
        var.initial_value = init[i]
        var.bounds = Vt_bnds[i]
        i += 1
    # for var in circuit.U:
    #     var.initial_value = init[i]
    #     i += 1
    # for var in circuit.S:
    #     var.initial_value = init[i]
    #     i += 1

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
        pi_co, hi_co, mi_co = circuit.components['Compressor'].ports[0].p.value, circuit.components['Compressor'].ports[0].h.value, circuit.components['Compressor'].ports[0].m.value
        pi_c, hi_c, mi_c = circuit.components['Condenser'].ports[0].p.value, circuit.components['Condenser'].ports[0].h.value, circuit.components['Condenser'].ports[0].m.value
        pi_ev, hi_ev, mi_ev = circuit.components['Expansion Valve'].ports[0].p.value, circuit.components['Expansion Valve'].ports[0].h.value, circuit.components['Expansion Valve'].ports[0].m.value
        pi_v, hi_v, mi_v = circuit.components['Evaporator'].ports[1].p.value, circuit.components['Evaporator'].ports[1].h.value, circuit.components['Evaporator'].ports[1].m.value

        Ti_co = PropsSI('T', 'P', pi_co, 'H', hi_co, fluid_list[0]) - 273.15
        SH = Ti_co - (PropsSI('T', 'P', pi_v, 'Q', 1.0, fluid_list[0]) - 273.15)

        for key in circuit.components:
            for port in circuit.components[key].ports:
                T = PropsSI('T', 'H', port.h.value, 'P', port.p.value, port.fluid) - 273.15
                print(f'{circuit.components[key].name} Port ID {port.port_id}: p = {port.p.value * 1e-5} bar, h = {port.h.value * 1e-3} kJ/kg, m = {port.m.value} kg/s, T = {T} °C \n')

        # plots log ph diagramm
        logph([[hi_co * 1e-3, hi_c * 1e-3, hi_ev * 1e-3, hi_v * 1e-3, hi_co * 1e-3]],
              [[pi_co * 1e-5, pi_c * 1e-5, pi_ev * 1e-5, pi_v * 1e-5, pi_co * 1e-5]],
              [[1, 2, 3, 4, 1]],
              [fluid_list[0]])


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
