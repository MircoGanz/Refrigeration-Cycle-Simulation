"""
Created on Thu Jan 19 07:14:57 2023

@author: Mirco Ganz

Accompany the Master Thesis:

M. Ganz (2023) Numerical Modeling and Analysis of an Adaptive Refrigeration Cycle Simulator
"""

import csv
import multiprocessing
from copy import deepcopy
import pickle
from functools import partial
from typing import List

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from CoolProp.CoolProp import PropsSI, PhaseSI
from labellines import labelLines
import math

# from pyoptsparse import OPT, Optimization


# dictionary to convert junction port connectivity matrix string values to integer values
from scipy.optimize import NonlinearConstraint, BFGS, minimize, Bounds

psd = {'0': 0,
       'p': 1, '-p': -1,
       'h': 2, '-h': -2,
       'c': 3, '-c': -3,
       'tp': 4, '-tp': -4,
       'g': 5, '-g': -5,
       'l': 6, '-l': -6,
       'A': 7, 'B': 8, '-C': -9
       }


# class Variable:
#     """
#     Represents a variable in the system.
#
#     Attributes:
#         name (str): Name of the variable.
#         port_type (str): Port type.
#         port_id (list): Port ID.
#         var_typ (str): Variable type.
#         value (float): Value of the variable.
#         known (bool): Indicates if the variable value is known.
#         bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
#     """
#
#     def __init__(self, name: str, port_type: str, port_id: list, var_type: str, scale_factor: float, value=None,
#                  known=False, is_input=False, bounds=(-np.inf, np.inf)):
#         """
#         Initialize a Variable object.
#
#         Args:
#             name: Name of the variable.
#             port_type: Port type.
#             port_id: Port ID.
#             var_type: Variable type.
#             value: Value of the variable.
#             bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
#         """
#         self.name = name
#         self.port_type = port_type
#         self.port_id = port_id
#         self.var_type = var_type
#         self.value = value
#         self.initial_value = value
#         self.known = known
#         self.is_input = is_input
#         self.scale_factor = scale_factor
#         self.bounds = bounds
#
#     def set_value(self, value: float):
#         """
#         Set the value of the variable.
#
#         Args:
#             value: Value to set.
#         """
#         self.value = value
#         self.known = True
#
#     def reset(self):
#         """
#         Reset the variable to its initial state.
#         """
#         self.known = False
#         self.value = None


class Variable:
    """
    Represents a variable in the system.

    Attributes:
        name (str): Name of the variable.
        port_type (str): Port type.
        port_id (list): Port ID.
        var_typ (str): Variable type.
        value (float): Value of the variable.
        known (bool): Indicates if the variable value is known.
        bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
    """

    def __init__(self, name: str, port_type: str, port_id: list, var_type: str, scale_factor: float, value=None,
                 known=False, is_input=False, bounds=(-np.inf, np.inf)):
        """
        Initialize a Variable object.

        Args:
            name: Name of the variable.
            port_type: Port type.
            port_id: Port ID.
            var_type: Variable type.
            value: Value of the variable.
            bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
        """
        self.name = name
        self.port_type = port_type
        self.port_id = port_id
        self.var_type = var_type
        self.value = value
        self.initial_value = value
        self.known = known
        self.is_input = is_input
        self.scale_factor = scale_factor
        self.bounds = bounds

    def set_value(self, value):
        """
        Set the value of the variable.

        Args:
            value: Value to set.
        """
        if isinstance(value, DualNumber):
            self.value = DualNumber(value.no, value.der)
        else:
            self.value = value
        self.known = True

    def reset(self):
        """
        Reset the variable to its initial state.
        """
        self.known = False
        self.value = None


class DualNumber:
    def __init__(self, value, der):
        self.no = value
        self.der = der

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no + other.no, self.der + other.der)
        elif isinstance(other, (float, int)):
            return DualNumber(self.no + float(other), self.der)

    def __radd__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no + other.no, self.der + other.der)
        elif isinstance(other, (float, int)):
            return DualNumber(self.no + float(other), self.der)

    def __iadd__(self, other):
        if isinstance(other, DualNumber):
            self.no += other.no
            self.der += other.der
        elif isinstance(other, (float, int)):
            self.no += float(other)
        return self

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no - other.no, self.der - other.der)
        elif isinstance(other, (float, int)):
            return DualNumber(self.no - float(other), self.der)

    def __rsub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.no - self.no, other.der - self.der)
        elif isinstance(other, (float, int)):
            return DualNumber(float(other) - self.no, -self.der)

    def __isub__(self, other):
        if isinstance(other, DualNumber):
            self.no -= other.no
            self.der -= other.der
        elif isinstance(other, (float, int)):
            self.no -= float(other)
        return self

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no * other.no, self.der * other.no + self.no * other.der)
        elif isinstance(other, (float, int)):
            return DualNumber(self.no * float(other), self.der * float(other))
        elif isinstance(other, np.ndarray):
            if all(isinstance(item, DualNumber) for item in other):
                result_values = self.no * np.array([item.no for item in other])
                result_derivatives = self.der * np.array([item.no for item in other]) + self.no * np.array([item.der for item in other])
                return np.array([DualNumber(val, der) for val, der in zip(result_values, result_derivatives)])
            elif all(isinstance(item, (float, int)) for item in other):
                result_values = self.no * np.array([item for item in other])
                result_derivatives = self.der * np.array([item for item in other])
                return np.array([DualNumber(val, der) for val, der in zip(result_values, result_derivatives)])
            else:
                raise ValueError("Unsupported operand types for multiplication.")
        else:
            raise ValueError("Unsupported operand types for multiplication.")

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return DualNumber(self.no * float(other), self.der * float(other))
        elif isinstance(other, np.ndarray):
            if all(isinstance(item, DualNumber) for item in other):
                result_values = self.no * np.array([item.no for item in other])
                result_derivatives = self.der * np.array([item.no for item in other]) + self.no * np.array([item.der for item in other])
                return np.array([DualNumber(val, der) for val, der in zip(result_values, result_derivatives)])
            elif all(isinstance(item, (float, int)) for item in other):
                return DualNumber(self.no * np.prod(other), self.der * np.prod(other))
            else:
                raise ValueError("Unsupported operand types for multiplication.")
        else:
            raise ValueError("Unsupported operand types for multiplication.")

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no / other.no, (self.der * other.no - self.no * other.der) / (other.no ** 2))
        elif isinstance(other, (float, int)):
            return DualNumber(self.no / float(other), self.der / float(other))

    def __rtruediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.no / self.no, (other.der * self.no - other.no * self.der) / (self.no ** 2))
        elif isinstance(other, (float, int)):
            return DualNumber(float(other) / self.no, -float(other) * self.der / (self.no ** 2))

    def log(self):
        return DualNumber(math.log(self.no), 1 / self.no * self.der)

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.no ** other.no, (other.no * self.no ** (other.no - 1) * self.der))
        elif isinstance(other, (float, int)):
            return DualNumber(self.no ** float(other), float(other) * self.no ** (float(other) - 1) * self.der)

    def sqrt(self):
        return DualNumber(math.sqrt(self.no), 1 / (2 * math.sqrt(self.no)) * self.der)

    def exp(self):
        return DualNumber(math.exp(self.no), math.exp(self.no) * self.der)

    def __abs__(self):
        if self.no > 0:
            return DualNumber(abs(self.no), 1.0 * self.der)
        elif self.no < 0:
            return DualNumber(abs(self.no), -1.0 * self.der)
        else:
            Warning("Derivative of abs() function not defined at 0.0 for a Dual Variable!")
            return DualNumber(abs(self.no), 0.0)

    def __lt__(self, other):
        if isinstance(other, DualNumber):
            return self.no < other.no
        elif isinstance(other, (float, int)):
            return self.no < other

    def __gt__(self, other):
        if isinstance(other, DualNumber):
            return self.no > other.no
        elif isinstance(other, (float, int)):
            return self.no > other


class SlackVariable:
    """
    Represents a Slack Variable used for relaxing the optimization problem for inverse modeling.

    Attributes:
        value: Current value of the slack variable.
        scale_factor (float): Scaling factor for the slack variable.
        bounds (tuple): Bounds for the slack variable.
    """

    def __init__(self, value: float, initial_value=0.0, scale_factor=1.0, bounds=(-np.inf, np.inf)):
        """
        Initialize a Parameter object.

        Args:
            value (float):Current value of the slack variable.
            scale_factor (float): Scaling factor for the slack variable.
            bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
        """

        self.value = value
        self.initial_value = initial_value
        self.scale_factor = scale_factor
        self.bounds = bounds

    def set_value(self, value: float):
        self.value = value


class Parameter:
    """
    Represents a Parameter of a Thermal-Hydraulic Circuit Component.

    Attributes:
        name (str): Name of the parameter.
        value: Current value of the parameter.
        scale_factor (float): Scaling factor for the parameter.
        is_input (bool): Indicates if the parameter is an input.
        bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
    """

    def __init__(self, name: str, value: float, scale_factor=1.0, initial_value=None, is_input=False, bounds=(-np.inf, np.inf)):
        """
        Initialize a Parameter object.

        Args:
            name (str): Name of the input.
            value (optional): Initial value of the input.
            scale_factor (float): Scaling factor of the input.
            is_input (bool): Whether the parameter is an input.
            bounds (tuple): Bounds for the parameter (default: (-inf, inf)).
        """

        self.name = name
        self.value = value
        if initial_value is None:
            self.initial_value = value
        else:
            self.initial_value = initial_value
        self.scale_factor = scale_factor
        self.is_input = is_input
        self.bounds = bounds

    def set_value(self, value: float):
        self.value = value


class Output:

    def __init__(self, name: str):
        self.name = name
        self.value = None

    def set_value(self, value: float):
        self.value = value


class Port:
    """
    Represents a port of a component.

    Attributes:
        port_id (list): Port ID.
        port_typ (str): Port type.
        fluid: Fluid associated with the port.
        p (Variable): Pressure variable of the port.
        h (Variable): Enthalpy variable of the port.
        m (Variable): Mass flow variable of the port.
    """

    def __init__(self, port_id: list, port_type: str):
        """
        Initialize a Port object.

        Args:
            port_id: Port ID.
            port_type: Port type.
        """
        self.port_id = port_id
        self.port_type = port_type
        self.fluid = None
        self.p = Variable(f"p({self.port_type})", self.port_type, self.port_id, "p", 1e-5)
        self.h = Variable(f"h({self.port_type})", self.port_type, self.port_id, "h", 1e-5)
        self.m = Variable(f"m({self.port_type})", self.port_type, self.port_id, "m", 1e1)


class Component:
    """
    A base class representing a component in a thermal hydraulic system.

    Attributes:
        modeling_type (str): The modeling type of the component. Default value is "undefined".
        number (int): The component number.
        component_type (str): The component type.
        name (str): The component name.
        source_component (bool): Indicates if the component is a source component.
        fluid_loop_list (list): list of fluids for which at least one component port is connected.
        executed (bool): Indicates if the component has been executed.
        ports (list): The list of ports associated with the component.
        specifications (dict): A dictionary of specifications for the component.
        parameters (dict): A dictionary of parameters for the component.
        inputs (dict): A dictionary of inputs of the component.
        outputs (dict): A dictionary of outputs of the component.
        solver_path (str): The path to the solver for the component.
        status (int): The status of the component (1: solving component successful, 0: solving component unsuccessful).
        lamda (float): Homotopy parameter of the solver
        linearized(bool): Indicates if linearization of the component is used in solver
        x0 (list): The values at which the component is linearized.
        J (list): The Jacobian at x0 of the component.
        F0 (list): The output values at the input values x0 of the component.
        no_in_ports (int): The number of input ports.
        no_out_ports (int): The number of output ports.
    """

    modeling_type = "undefined"

    def __init__(self, number: int, component_type: str, name: str, jpcm: list):
        """
        Initialize a Component object.

        Args:
            number: Component number.
            component_typ: Component type (Compressor, Pump, Heat Exchanger, Expansion Valve, Mixing Valve, Separator).
            name: Component name.
            jpcm: List of jpcm values.
        """
        self.number = number
        self.component_type = component_type
        self.name = name
        self.executed = False
        self.solved = False
        self.diagramm_plot = False
        self.calculate_derivatives = False
        self.ports = {}
        self.specifications = {}
        self.parameter = {}
        self.inputs = {}
        self.outputs = {}
        self.solver_path = ""
        self.status = 1
        self.lamda = 1.0
        self.linearized = False
        self.x0 = []
        self.J = []
        self.F0 = []
        self.no_in_ports = 0
        self.no_out_ports = 0

        for j, line in enumerate(jpcm):
            if line[self.number - 1] > 0:
                self.ports[line[self.number - 1]] = Port(
                    [j + 1, self.number, line[self.number - 1], line[-2], line[-1]], "in")
                self.no_in_ports += 1
            elif line[self.number - 1] < 0:
                self.ports[line[self.number - 1]] = Port(
                    [j + 1, self.number, line[self.number - 1], line[-2], line[-1]], "out")
                self.no_out_ports += 1

    def reset(self):
        """
        Reset the Component object.
        """
        self.executed = False
        self.status = 1
        for key in self.ports:
            self.ports[key].p.reset()
            self.ports[key].h.reset()
            self.ports[key].m.reset()

    def solve(self, calculate_derivatives=False):
        pass


class PressureBasedComponent(Component):
    modeling_type = "Pressure Based"

    def solve(self):
        """
        Solve the PressureBasedComponent using the solver specified by the solver_path.
        """
        MODULE_PATH = self.solver_path + "/__init__.py"
        with open(MODULE_PATH) as f:
            code = compile(f.read(), MODULE_PATH, 'exec')
        namespace = {}
        exec(code, namespace)
        namespace['solver'](self)

    def jacobian(self):
        """
        Calculate the Jacobian matrix for the component.

        This method calculates the Jacobian matrix of the pressure based component using finite differences.
        The Jacobian matrix represents the partial derivatives of the component's output variables
        with respect to its input variables.

        Returns:
            None
        """
        i = 0
        self.J = np.zeros([2 * self.no_out_ports + self.no_in_ports, 2 * self.no_in_ports + self.no_out_ports])
        self.F0 = np.zeros([2 * self.no_out_ports + self.no_in_ports])
        for key in self.ports:
            if self.ports[key].port_type == 'in':
                self.ports[key].p.set_value(self.x0[i])
                self.ports[key].h.set_value(self.x0[i + 1])
                i += 2
            elif self.ports[key].port_type == 'out':
                self.ports[key].h.set_value(self.x0[i])
                i += 1
        self.solve()
        i = 0
        for key in self.ports:
            if self.ports[key].port_type == 'in':
                self.F0[i] = self.ports[key].m.value
                i += 1
            elif self.ports[key].port_type == 'out':
                self.F0[i] = self.ports[key].h.value
                self.F0[i + 1] = self.ports[key].m.value
                i += 2

        i = 0
        for key in self.ports:
            if self.ports[key].port_type == 'in':
                j = 0
                self.ports[key].p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                if port_inside.port_type == 'in':
                    self.J[j, i] = (port_inside.m.value - self.F0[i]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 1
                elif port_inside.port_type == 'out' and not port_inside.port_id[-1] == 1:
                    self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    self.J[j + 1, i] = (port_inside.m.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 2
                self.ports[key].p.set_value(self.x0[i])
                i += 1

                j = 0
                self.ports[key].h.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                if port_inside.port_type == 'in':
                    self.J[j, i] = (port_inside.m.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 1
                elif port_inside.port_type == 'out':
                    self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    self.J[j + 1, i] = (port_inside.m.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 2
                self.ports[key].h.set_value(self.x0[i])
                i += 1

            if self.ports[key].port_type == 'out':
                j = 0
                self.ports[key].p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                if port_inside.port_type == 'in':
                    self.J[j, i] = (port_inside.m.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 1
                elif port_inside.port_type == 'out' and not port_inside.port_id[-1] == 1:
                    self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    self.J[j + 1, i] = (port_inside.m.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                    j += 2
                self.ports[key].p.set_value(self.x0[i])
                i += 1

        self.reset()


class MassFlowBasedComponent(Component):
    modeling_type = "Mass Flow Based"

    def solve(self):
        MODULE_PATH = self.solver_path + "/__init__.py"
        with open(MODULE_PATH) as f:
            code = compile(f.read(), MODULE_PATH, 'exec')
        namespace = {}
        exec(code, namespace)
        namespace['solver'](self)

    def jacobian(self):

        """
        Calculate the Jacobian matrix for the component.

        This method calculates the Jacobian matrix of the mass flow based component using finite differences.
        The Jacobian matrix represents the partial derivatives of the component's output variables
        with respect to its input variables.

        Returns:
            None
        """

        i = 0
        self.J = np.zeros([3 * self.no_out_ports, 3 * self.no_in_ports])
        self.F0 = np.zeros([3 * self.no_out_ports])
        for port in self.ports:
            if port.port_type == 'in' and port.port_id[-1] == 0:
                port.p.set_value(self.x0[i])
                port.h.set_value(self.x0[i + 1])
                port.m.set_value(self.x0[i + 2])
                i += 3
        self.solve()
        i = 0
        for port in self.ports:
            if port.port_type == 'out' and not port.port_id[-1] == 1:
                self.F0[i] = port.p.value
                self.F0[i + 1] = port.h.value
                self.F0[i + 2] = port.m.value
                i += 3

        i = 0
        for port in self.ports:
            if port.port_type == 'in' and port.port_id[-1] == 0:
                j = 0
                port.p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_type == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 1, i] = (port_inside.h.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 2, i] = (port_inside.m.value - self.F0[j + 2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.p.set_value(self.x0[i])
                i += 1

                j = 0
                port.h.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_type == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 1, i] = (port_inside.h.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 2, i] = (port_inside.m.value - self.F0[j + 2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.h.set_value(self.x0[i])
                i += 1

                j = 0
                port.m.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_type == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 1, i] = (port_inside.h.value - self.F0[j + 1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j + 2, i] = (port_inside.m.value - self.F0[j + 2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.m.set_value(self.x0[i])
                i += 1

        self.reset()


class SeparatorComponent(Component):
    modeling_type = "Separator"

    def __init__(self, number: int, component_type: str, name: str, jpcm: list):
        super().__init__(number, component_type, name, jpcm)

    def solve(self):

        p_in = self.ports[psd['tp']].p.value
        h_in = self.ports[psd['tp']].h.value
        m_in = self.ports[psd['tp']].m.value
        fluid = self.ports[psd['tp']].fluid

        p_out = p_in
        # phase = PhaseSI('H', h_in, 'P', p_in, fluid)
        # if phase == 'twophase':
        #     h_out = coolpropsHPQ(p_in, 0.0, fluid)
        # else:
        #     h_out = h_in
        m_out = m_in
        h_out = h_in

        self.ports[psd['-l']].p.set_value(p_out)
        self.ports[psd['-l']].h.set_value(h_out)
        self.ports[psd['-l']].m.set_value(m_out)


class BypassComponent(Component):
    """
    Represents a Bypass Component.
    """

    modeling_type = "Bypass"

    def solve(self):
        """
        Solve the bypass component.

        This method solves the bypass component by executing the solver function specified in the solver path.
        """
        # MODULE_PATH = self.solver_path + "/__init__.py"
        # with open(MODULE_PATH) as f:
        #     code = compile(f.read(), MODULE_PATH, 'exec')
        # namespace = {}
        # exec(code, namespace)
        # namespace['solver'](self)

        h_in = self.ports[psd['p']].h.value
        m_in = self.ports[psd['p']].m.value

        self.ports[psd['-p']].h.set_value(h_in)
        self.ports[psd['-p']].m.set_value(m_in)


class Source(Component):
    """
    Represents a Source Component.
    """

    modeling_type = "Source"

    def __init__(self, number: int, component_type: str, name: str, jpcm: list):
        super().__init__(number, component_type, name, jpcm)

    def solve(self):
        """
        Solve the source component.

        This method solves the source component.
        """
        self.ports[psd['-p']].p.set_value(self.parameter['p_source'].value)
        self.ports[psd['-p']].h.set_value(
            PropsSI('H', 'P', self.parameter['p_source'].value, 'T', self.parameter['T_source'].value,
                    self.ports[psd['-p']].fluid))
        # self.ports[psd['-p']].h.set_value(self.parameter['T_source'].value)

        self.ports[psd['-p']].m.set_value(self.parameter['m_source'].value)


class Sink(Component):
    """
    Represents a Source Component.
    """

    modeling_type = "Source"

    def __init__(self, number: int, component_type: str, name: str, jpcm: list):
        super().__init__(number, component_type, name, jpcm)

    def solve(self):
        """
        Solve the source component.

        This method solves the source component.
        """
        self.parameter['p_sink'].set_value(self.ports[psd['p']].p.value)
        self.parameter['T_sink'].set_value(self.ports[psd['p']].h.value)
        self.parameter['m_sink'].set_value(self.ports[psd['p']].m.value)


class BalanceEquation:
    """
    Represents a balance equation.

    Attributes:
        name (str): The name of the equation.
        variables (list): List of variables involved in the balance equation.
        fluid_loop: Identifier of the fluid loop.
    """

    name = ""

    def __init__(self, variables, fluid_loop):
        """
        Initialize a balance equation.

        Args:
            variables (list): List of variables involved in the balance equation.
            fluid_loop: Identifier of the fluid loop.
        """
        self.variables = variables
        self.fluid_loop = fluid_loop
        self.solved = False

    def is_solvable(self):
        """
        Check if the balance equation is solvable.

        Returns:
            bool: True if the equation is solvable, False otherwise.
        """
        n_unknown = sum(not variable.known for variable in self.variables)
        return n_unknown == 1


class MassFlowBalance(BalanceEquation):
    """
    Represents a mass flow balance equation.

    Attributes:
        name (str): The name of the equation ("Mass Flow Balance").
        variables (List[Variable]): List of variables involved in the equation.
        fluid_loop (int): Identifier of the fluid loop.
        solved (bool): Indicates whether the equation has been solved or not.
    """

    name = "Mass Flow Balance"
    scale_factor = 1e1

    def solve(self):
        """
        Solve the mass flow balance equation.

        This method calculates the unknown variable by summing the known variables and applying mass conservation.
        """
        if any([isinstance(var, DualNumber) for var in self.variables]):
            m_total = DualNumber(0.0, 0.0)
        else:
            m_total = 0
        unknown_variable = None
        for variable in self.variables:
            if variable.known:
                if variable.port_type == 'out':
                    m_total += variable.value
                elif variable.port_type == 'in':
                    m_total -= variable.value
            else:
                unknown_variable = variable

        if not unknown_variable:
            raise RuntimeError("Tried to solve equation: " + self.name + ", but no unknown Variable was found.")

        unknown_variable.set_value(m_total)
        self.solved = True


    def residual(self):
        """
        Calculate the residual of the mass flow balance equation.

        Returns:
            float: Residual value of the equation.
        """
        res = sum(variable.value for variable in self.variables if variable.port_type == "out") \
              - sum(variable.value for variable in self.variables if variable.port_type == "in")
        return res * self.scale_factor


class PressureEquality(BalanceEquation):
    """
    Represents a pressure equality equation.

    Attributes:
        name (str): The name of the equation ("Pressure Equality").
        variables (List[Variable]): List of variables involved in the equation.
        fluid_loop (int): Identifier of the fluid loop.
        solved (bool): Indicates whether the equation has been solved or not.
    """

    name = "Pressure Equality"
    scale_factor = 1e-5

    def solve(self):
        """
        Solve the pressure equality equation.

        This method sets the unknown variable to the known pressure value.
        """
        if any([isinstance(var, DualNumber) for var in self.variables]):
            p = DualNumber(0.0, 0.0)
        else:
            p = 0
        unknown_variable = None
        for variable in self.variables:
            if variable.known:
                p = variable.value
            else:
                unknown_variable = variable

        if not unknown_variable:
            raise RuntimeError("Tried to solve equation: " + self.name + ", but no unknown Variable was found.")

        unknown_variable.set_value(p)
        self.solved = True

    def residual(self):
        """
        Calculate the residual of the pressure equality equation.

        Returns:
            float: Residual value of the equation.
        """
        return (self.variables[0].value - self.variables[1].value) * self.scale_factor


class EnthalpyEquality(BalanceEquation):
    """
    Represents an enthalpy equality equation.

    Attributes:
        name (str): The name of the equation ("Enthalpy Equality").
        variables (List[Variable]): List of variables involved in the equation.
        fluid_loop (int): Identifier of the fluid loop.
        solved (bool): Indicates whether the equation has been solved or not.
    """

    name = "Enthalpy Equality"
    scale_factor = 1e-5

    def solve(self):
        """
        Solve the enthalpy equality equation.

        This method sets the unknown variable to the known enthalpy value.
        """
        if not self.is_solvable():
            raise RuntimeError("Tried to solve equation: " + self.name + ", but it is not solvable yet.")

        if any([isinstance(var, DualNumber) for var in self.variables]):
            h = DualNumber(0.0, 0.0)
        else:
            h = 0
        unknown_variable = None
        for variable in self.variables:
            if variable.known:
                h = variable.value
            else:
                unknown_variable = variable

        if not unknown_variable:
            raise RuntimeError("Tried to solve equation: " + self.name + ", but no unknown Variable was found.")

        unknown_variable.set_value(h)
        self.solved = True

    def residual(self):
        """
        Calculate the residual of the enthalpy equality equation.

        Returns:
            float: Residual value of the equation.
        """
        return (self.variables[0].value - self.variables[1].value) * self.scale_factor


class EnthalpyFlowBalance:
    """
    Represents an enthalpy flow balance equation.

    Attributes:
        name (str): The name of the equation ("Enthalpy Flow Balance").
        port_typ_multiplier (Dict[str, int]): A dictionary mapping port types to their corresponding multipliers.
        variables (List[List[Variable]]): A list of variable pairs involved in the equation.
            Each variable pair is represented as a list [variable1, variable2].
        fluid_loop (int): The identifier of the fluid loop.
        solved (bool): Indicates whether the equation has been solved or not.
    """

    name = "Enthalpy Flow Balance"
    scale_factor = 1e-4
    port_typ_multiplier = {"out": 1, "in": -1}

    def __init__(self, variables: List[List[Variable]], fluid_loop: int):
        """
        Initialize an enthalpy flow balance equation.

        Args:
            variables: List of variable pairs involved in the equation.
            fluid_loop (int): Identifier of the fluid loop.
        """
        self.variables = variables
        self.fluid_loop = fluid_loop
        self.solved = False

    def is_solvable(self) -> bool:
        """
        Check if the enthalpy flow balance equation is solvable.

        Returns:
            bool: True if the equation is solvable, False otherwise.
        """
        n_unknown = sum(not item.known for variable in self.variables for item in variable)
        return n_unknown == 1

    def solve(self):
        """
        Solve the enthalpy flow balance equation.

        This method calculates the total enthalpy flow and sets the value of the unknown variable in each variable pair.
        """
        if not self.is_solvable():
            raise RuntimeError("Tried to solve equation: " + self.name + ", but it is not solvable yet.")

        H_total = sum(variable[0].value * variable[1].value * self.port_typ_multiplier[variable[0].port_type]
                      for variable in self.variables if variable[0].known and variable[1].known)

        for variable in self.variables:
            if not variable[0].known:
                variable[0].set_value(H_total / variable[1].value)
            elif not variable[1].known:
                variable[1].set_value(H_total / variable[0].value)

    def residual(self):
        """
        Calculate the residual of the enthalpy flow balance equation.

        Returns:
            float: Residual value of the equation.
        """
        res = sum(variable[0].value * variable[1].value * self.port_typ_multiplier[variable[0].port_type]
                  for variable in self.variables)
        return res * self.scale_factor


class LoopBreakerEquation:

    def __init__(self, component: Component, parameter_name: str):
        self.component = component
        self.parameter_name = parameter_name

    def solve(self):
        return self.component.parameter[self.parameter_name].value * self.component.parameter[
            self.parameter_name].scale_factor


class DesignEquation:
    """
    Represents a design equation for a component.

    Args:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.
        relaxed (boolean): defines if design equation is treated as relaxed in the optimization

    Attributes:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.
        relaxed (boolean): defines if design equation is treated as relaxed in the optimization
        res (float): The calculated result of the equation.
    """

    def __init__(self, component: Component, DC_value: float, relaxed=False):
        self.component = component
        self.DC_value = DC_value
        self.relaxed = relaxed
        self.res = float()
        self.S = SlackVariable


class SuperheatEquation(DesignEquation):
    """
    Represents a superheat equation for a component.

    Args:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.

    Attributes:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.
        res (float): The calculated result of the equation.
    """
    name = 'Superheat Equation'
    scale_factor = 1e-5

    def __init__(self, component: Component, DC_value: float, port_type: str, var_type: str, port_id: int,
                 relaxed=False):

        super().__init__(component, DC_value, relaxed)

        self.DC_port_type = port_type
        self.DC_var_type = var_type
        self.port_id = port_id

    def solve(self):
        """
        Solve the superheat equation.

        Returns:
            float: The residual value of the equation.
        """
        T_sat = coolpropsTPQ(self.component.ports[psd['-c']].p.value, 1.0, self.component.ports[psd['-c']].fluid)
        if self.DC_value < 1e-4:
            h_SH = coolpropsHPQ(self.component.ports[psd['-c']].p.value, 1.0, self.component.ports[psd['-c']].fluid)
        else:
            h_SH = coolpropsHTP(T_sat + self.DC_value, self.component.ports[psd['-c']].p.value, self.component.ports[psd['-c']].fluid)
        self.res = (self.component.ports[psd['-c']].h.value - h_SH)
        return self.res


class SubcoolingEquation(DesignEquation):
    """
    Represents a subcooling equation for a component.
    """
    name = 'Subcooling Equation'
    scale_factor = 1e-5

    def __init__(self, component: Component, DC_value: float, port_type: str, var_type: str, port_id: int,
                 relaxed=False):

        super().__init__(component, DC_value, relaxed)

        self.DC_port_type = port_type
        self.DC_var_type = var_type
        self.port_id = port_id
        self.scale_factor = 1e-5

    def solve(self):
        """
        Solve the subcooling equation.

        Returns:
            float: The residual value of the equation.
        """

        T_sat = coolpropsTPQ(self.component.ports[psd['-c']].p.value, 0.0, self.component.ports[psd['-c']].fluid)
        if self.DC_value < 1e-4:
            h_SC = coolpropsHPQ(self.component.ports[psd['-c']].p.value, 0.0, self.component.ports[psd['-c']].fluid)
        else:
            h_SC = coolpropsHTP(T_sat + self.DC_value, self.component.ports[psd['-c']].p.value, self.component.ports[psd['-c']].fluid)
        self.res = (self.component.ports[psd['-h']].h.value - h_SC)
        return self.res


class DesignParameterEquation(DesignEquation):
    """
    Represents a design parameter equation.

    Attributes:
        DC_var_typ (str): The variable type string.
        DC_port_typ (str): The port type string.
        res (float): The equation residual.
    """
    name = 'Design Parameter Equation'

    def __init__(self, component: Component, DC_value: float, port_type: str, var_type: str, port_id: int,
                 relaxed=False):

        super().__init__(component, DC_value, relaxed)

        self.DC_port_type = port_type
        self.DC_var_type = var_type
        self.port_id = port_id
        if var_type in ['p', 'h']:
            self.scale_factor = 1e-5
        elif var_type == 'T':
            self.scale_factor = 1e-2
        elif var_type == 'm':
            self.scale_factor = 1e1
        else:
            RuntimeError(f'Design equation for variable type "{var_type}" is not defined!')

    def solve(self):
        """
        Solve the design parameter equation.

        Returns:
            float: The residual value of the equation.
        """

        for key in self.component.ports:
            if self.component.ports[key].p.var_type == self.DC_var_type and self.component.ports[key].p.port_type == self.DC_port_type:
                self.res = (self.component.ports[key].p.value - self.DC_value)
                break
            elif self.component.ports[key].h.var_type == self.DC_var_type and self.component.ports[key].h.port_id[
                2] == self.port_id:
                self.res = (self.component.ports[key].h.value - self.DC_value)
                break
            elif self.DC_var_type == 'T' and self.component.ports[key].port_id[2] == self.port_id:
                T = coolpropsTPH(self.component.ports[key].p.value, self.component.ports[key].h.value, self.component.ports[key].fluid)
                self.res = (T - self.DC_value)
        return self.res


class OutputDesignEquation(DesignEquation):

    def __init__(self, component: Component, DC_value: float, output_name: str, scale_factor=1.0, relaxed=False):
        super().__init__(component, DC_value, relaxed)

        self.output_name = output_name
        self.scale_factor = scale_factor

        if not output_name in [key for key in component.outputs]:
            RuntimeError(f'Output name "{output_name}" not in components output dictionary!')
        else:
            pass

    def solve(self):
        return self.component.outputs[self.output_name].value - self.DC_value


class Junction:
    """
    Represents a junction.

    Attributes:
        equations (List[List[Equation]]): List of equations associated with the junction.
        in_comp (List[Component]): List of input components connected to the junction.
        out_comp (List[Component]): List of output components connected to the junction.
        number (int): The junction number.
        fluid_loop (int): Identifier of the fluid loop.

    """

    def __init__(self, number: int, inp_comp: List[Component], out_comp: List[Component],
                 fluid_loop: int):
        self.equations = None
        self.in_comp = inp_comp
        self.out_comp = out_comp
        self.number = number
        self.fluid_loop = fluid_loop

    def create_equations(self):

        """
        Create the equations associated with the junction.
        """

        pressure_variables = []
        enthalpy_variables = []
        mass_flow_variables = []
        enthalpy_flow_variables = []
        for ic in self.in_comp:
            for key in ic.ports:
                if ic.ports[key].port_type == 'out' and ic.ports[key].port_id[0] == self.number:
                    pressure_variables.append(ic.ports[key].p)
                    mass_flow_variables.append(ic.ports[key].m)
                    if len(self.in_comp) == 1:
                        enthalpy_variables.append(ic.ports[key].h)
                    enthalpy_flow_variables.append([ic.ports[key].m, ic.ports[key].h])

        for oc in self.out_comp:
            for key in oc.ports:
                if oc.ports[key].port_type == 'in' and oc.ports[key].port_id[0] == self.number:
                    pressure_variables.append(oc.ports[key].p)
                    mass_flow_variables.append(oc.ports[key].m)
                    enthalpy_variables.append(oc.ports[key].h)
                    enthalpy_flow_variables.append([oc.ports[key].m, oc.ports[key].h])

        pressure_equations = []
        id = [i for i, var in enumerate(pressure_variables) if var.port_type == 'in']
        ref_pressure = pressure_variables[id[0]]
        for pressure_variable in pressure_variables:
            if pressure_variable != ref_pressure:
                pressure_equations.append(PressureEquality([ref_pressure, pressure_variable], self.fluid_loop))

        enthalpy_equations = []
        id = [i for i, var in enumerate(enthalpy_variables) if var.port_type == 'in']
        ref_enthalpy = enthalpy_variables[id[0]]
        for enthalpy_variable in enthalpy_variables:
            if enthalpy_variable != ref_enthalpy:
                enthalpy_equations.append(EnthalpyEquality([ref_enthalpy, enthalpy_variable], self.fluid_loop))

        mass_flow_equation = MassFlowBalance(mass_flow_variables, self.fluid_loop)
        if len(self.in_comp) > 1:
            enthalpy_flow_equation = EnthalpyFlowBalance(enthalpy_flow_variables, self.fluid_loop)

            if enthalpy_equations:
                self.equations = [pressure_equations, enthalpy_equations, [enthalpy_flow_equation],
                                  [mass_flow_equation]]
            else:
                self.equations = [pressure_equations, [enthalpy_flow_equation], [mass_flow_equation]]
        else:
            if enthalpy_equations:
                self.equations = [pressure_equations, enthalpy_equations, [mass_flow_equation]]
            else:
                self.equations = [pressure_equations, [mass_flow_equation]]


class TripartiteGraph:
    """
    Represents a tripartite graph.

    Attributes:
        V (List[Variable]): List of variables in the graph.
        U (List[Equation]): List of equations in the graph.
        C (List[Component]): List of components in the graph.
        E (List[Tuple[Variable, Equation]]): List of undirected edges between variables and equations.
        Ed (List[Tuple[Variable, Component]]): List of directed edges between variables and components.

    """

    def __init__(self, jpcm, components):

        junctions = []
        equation_list = []
        components = [components[key] for key in components]

        for j, line in enumerate(jpcm):
            input_comp = [components[c] for c, i in enumerate(line[:-2]) if i < 0]
            output_comp = [components[c] for c, i in enumerate(line[:-2]) if i > 0]
            junction = Junction(j + 1, input_comp, output_comp, line[-2])
            junctions.append(junction)
            equation_list.append(junction.create_equations())

        self.V = []
        for junction in junctions:
            for equations in junction.equations:
                for equation in equations:
                    for variable in equation.variables:
                        if not isinstance(variable, list):
                            self.V.append(variable)
                        else:
                            self.V.append(variable[1])

        self.U = [equation for junction in junctions
                  for equations in junction.equations
                  for equation in equations]

        index_set = set()
        for element in jpcm[:, -2]:
            if element not in index_set:
                index_set.add(element)
            else:
                pass
        for fluid_no in index_set:
            check_list = []
            for row in jpcm:
                if row[-2] == fluid_no:
                    check_list.append(row)
                else:
                    pass
            if any(abs(row[-1]) == 1 for row in check_list):
                index_set = index_set.difference({fluid_no})

        for i, equation in enumerate(self.U):
            if equation.fluid_loop in index_set and isinstance(equation, MassFlowBalance):
                for var in equation.variables:
                    if components[var.port_id[1] - 1].component_type in ['Compressor', 'Pump']:
                        self.U.pop(i)
                        index_set = index_set.difference({equation.fluid_loop})
                        break
                    else:
                        pass

        self.C = components
        self.E = [(v, u) for v in self.V for u in self.U if v in u.variables]
        self.Ed = [(v, c) for c in self.C for v in self.V
                   if v.port_id[1] == c.number and ((isinstance(c, PressureBasedComponent) and (
                    (v.port_type == 'in' and v.var_type in ['p', 'h']) or (
                    v.port_type == 'out' and v.var_type == 'p')))
                                                    or ((isinstance(c, MassFlowBasedComponent) or isinstance(c,
                                                                                                             SeparatorComponent)) and v.port_type == 'in')
                                                    or (isinstance(c, BypassComponent) and (
                            (v.port_type == 'in' and v.var_type in ['p', 'h', 'm']) or (
                            v.port_type == 'out' and v.var_type == 'p')))
                                                    or (isinstance(c, Sink) and v.port_type == 'in'))]
        self.Ed.extend([(c, v) for c in self.C for v in self.V
                        if v.port_id[1] == c.number and ((isinstance(c, PressureBasedComponent) and (
                    v.var_type == 'm' or (v.var_type == 'h' and v.port_type == 'out')))
                                                         or ((isinstance(c, MassFlowBasedComponent) or isinstance(c,
                                                                                                                  SeparatorComponent)) and v.port_type == 'out')
                                                         or (isinstance(c, BypassComponent) and (
                            v.port_type == 'out' and v.var_type in ['h', 'm']))
                                                         or (isinstance(c, Source) and (
                            v.port_type == 'out' and v.var_type in ['p', 'h', 'm'])))])


class Circuit:

    def __init__(self, jpcm, component_type_list: list, component_name_list: list, component_modeling_type_list: list,
                 solver_path_list: list, fluid_list: list):

        self.jpcm = jpcm
        self.component_type_list = component_type_list
        self.component_name_list = component_name_list
        self.component_modeling_type_list = component_modeling_type_list
        self.solver_path_list = solver_path_list
        self.fluid_list = fluid_list
        self.design_equa = {}
        self.loop_breaker_equa = []
        self.U = []
        self.S = []
        self.xlast = []
        self.flast = []
        self.reslast = []
        self.grad_objfun = []

        # generates components and sets corresponding fluid to each port
        self.components = {}
        for i, item in enumerate(self.component_type_list):
            if component_modeling_type_list[i] == 'PressureBasedComponent':
                self.components[component_name_list[i]] = PressureBasedComponent(i + 1, item, self.component_name_list[i], jpcm)
            elif component_modeling_type_list[i] == 'MassFlowBasedComponent':
                self.components[component_name_list[i]] = MassFlowBasedComponent(i + 1, item, self.component_name_list[i], jpcm)
            elif component_modeling_type_list[i] == 'BypassComponent':
                self.components[component_name_list[i]] = BypassComponent(i + 1, item, self.component_name_list[i], jpcm)
            elif component_modeling_type_list[i] == 'SeparatorComponent':
                self.components[component_name_list[i]] = SeparatorComponent(i + 1, item, self.component_name_list[i], jpcm)
                # self.add_parameter(self.component_name_list[i], 'hLP', value=0.0, scale_factor=1e-5, is_input=True)
                # self.add_loop_breaker_equa(LoopBreakerEquation(self.components[component_name_list[i]], 'hLP'))
            elif component_modeling_type_list[i] == 'Source':
                self.components[component_name_list[i]] = Source(i + 1, item, self.component_name_list[i], jpcm)
                self.add_parameter(self.component_name_list[i], 'p_source', value=np.nan, scale_factor=1e-5)
                self.add_parameter(self.component_name_list[i], 'T_source', value=np.nan, scale_factor=1e-2)
                self.add_parameter(self.component_name_list[i], 'm_source', value=np.nan, scale_factor=1e1)
            elif component_modeling_type_list[i] == 'Sink':
                self.components[component_name_list[i]] = Sink(i + 1, item, self.component_name_list[i], jpcm)
                self.add_parameter(self.component_name_list[i], 'p_sink', value=np.nan, scale_factor=1e-5)
                self.add_parameter(self.component_name_list[i], 'T_sink', value=np.nan, scale_factor=1e-2)
                self.add_parameter(self.component_name_list[i], 'm_sink', value=np.nan, scale_factor=1e1)
            else:
                RuntimeError(
                    f'Component Modeling Type of Component {item.name} not defined as PressureBasedComponent, MassFlowBasedComponent, BypassComponent or SourceComponent')
            self.components[component_name_list[i]].solver_path = solver_path_list[i]
            for key in self.components[component_name_list[i]].ports:
                self.components[component_name_list[i]].ports[key].fluid = self.fluid_list[self.components[component_name_list[i]].ports[key].port_id[-2] - 1]

        # generates the systems Tripartite Graph from JPCM
        self.tpg = TripartiteGraph(self.jpcm, self.components)

        # runs tearing algorithm to identify tearing variable, execution list, residual equation and the necessity and number of
        # design equations
        self.Vt, self.exec_list, self.res_equa, self.no_design_equa = tearing_alg(self.tpg)

        print(f'Thermal-Hydraulic Circuit has been successfully initialized \n')
        print('Tearing Variables:')
        for var in self.Vt:
            print(
                f'{var.var_type} at Junction {var.port_id[0]} and {self.component_name_list[var.port_id[1] - 1]}')
        print(f'\n'
              f'Residual Equations:')
        for equa in self.res_equa:
            if isinstance(equa.variables[0], list):
                junction_no = equa.variables[0][0].port_id[0]
            else:
                junction_no = equa.variables[0].port_id[0]
            print(f'{equa.name} at Junction {junction_no}')
        print(f'\n'
              f'Number of design equations necessary to close equation system: {self.no_design_equa[1]}'
              f'\n')

    def add_design_equa(self, name, design_equa):
        self.design_equa[name] = design_equa
        # if design_equa.relaxed:
        #     self.S += [SlackVariable(value=0.0, scale_factor=design_equa.scale_factor)]
        #     design_equa.S = self.S[-1]
        # else:
        #     pass

    def add_loop_breaker_equa(self, loop_breaker_equa):
        self.loop_breaker_equa += [loop_breaker_equa]

    def add_inputs(self):
        for comp in self.components:
            for port in self.components[comp].ports:
                if self.components[comp].ports[port].p.is_input:
                    self.U += [port.p]
                elif self.components[comp].ports[port].h.is_input:
                    self.U += [port.h]
                elif self.components[comp].ports[port].m.is_input:
                    self.U += [port.m]
                else:
                    pass
            for param in self.components[comp].parameter:
                if self.components[comp].parameter[param].is_input:
                    self.U += [self.components[comp].parameter[param]]
                else:
                    pass

    def set_port_value(self, component_name: str, port_name: str, var_type: str, value: float):
        if component_name not in self.components.keys():
            raise RuntimeError(f'Component "{component_name}" not in Circuit!')
        else:
            if not psd[port_name] in self.components[component_name].ports.keys():
                raise RuntimeError(f'"{component_name}" has no Port "{port_name}"!')
            else:
                if var_type == 'p':
                    self.components[component_name].ports[psd[port_name]].p.set_value(value)
                elif var_type == 'h':
                    self.components[component_name].ports[psd[port_name]].h.set_value(value)
                elif var_type == 'm':
                    self.components[component_name].ports[psd[port_name]].m.set_value(value)
                else:
                    raise RuntimeError(f'"{var_type}" is not a valid variable type ! Must be p, h or m')

    def add_parameter(self, component_name: str, parameter_name: str, value: float, scale_factor=1.0, initial_value=None, is_input=False, bounds=(-np.inf, np.inf)):
        if component_name not in self.components.keys():
            raise RuntimeError(f'Component "{component_name}" not in Circuit!')
        else:
            self.components[component_name].parameter[parameter_name] = Parameter(name=parameter_name, value=value, scale_factor=scale_factor, initial_value=initial_value, is_input=is_input, bounds=bounds)

    def add_output(self, component_name: str, output_name: str):
        if component_name not in self.components.keys():
            raise RuntimeError(f'Component "{component_name}" not in Circuit!')
        else:
            self.components[component_name].outputs[output_name] = Output(output_name)

    def set_parameter(self, component_name: str, parameter_name: str, value: float, scale_factor: float = None, initial_value: float = None, is_input: bool = None, bounds: tuple = None):
        if component_name not in self.components.keys():
            raise RuntimeError(f'Component "{component_name}" not in Circuit!')
        else:
            if parameter_name not in self.components[component_name].parameter.keys():
                raise RuntimeError(f'Parameter "{parameter_name}" is not a parameter of {component_name}!')
            else:
                self.components[component_name].parameter[parameter_name].set_value(value)
                if scale_factor is not None:
                    self.components[component_name].parameter[parameter_name].scale_factor = scale_factor
                else:
                    pass
                if initial_value is not None:
                    self.components[component_name].parameter[parameter_name].initial_value = initial_value
                if is_input is not None:
                    self.components[component_name].parameter[parameter_name].is_input = is_input
                else:
                    pass
                if bounds is not None:
                    self.components[component_name].parameter[parameter_name].bounds = bounds

    def set_design_equa_DC_value(self, design_equa: DesignEquation, DC_value: float):
        if design_equa not in self.design_equa:
            raise RuntimeError(f'Design equation not in Circuit!')
        else:
            design_equa.DC_value = DC_value

    def objfun(self, x):

        # y = 0.5 * sum((x[len(self.Vt) + len(self.U):]) ** 2)

        i = 0
        # sets current iteration value to tearing variables
        for var in self.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to input variables
        for var in self.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to slack variables
        for var in self.S:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # solves and executes equations and components to solve the circuit
        try:
            for item in self.exec_list:
                if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                    item.solve()
                elif isinstance(item, Component):
                    item.lamda = 1.0
                    item.solve()
            f = sum([(self.design_equa[key].solve() * self.design_equa[key].scale_factor) ** 2
                    if self.design_equa[key].relaxed else 0.0 for key in self.design_equa])
        except Exception as e:
            print(e)
            [self.components[key].reset() for key in self.components]
        [self.components[key].reset() for key in self.components]
        return f

    def res_objfun(self, x):

        i = 0
        # sets current iteration value to tearing variables
        for var in self.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to input variables
        for var in self.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to slack variables
        for var in self.S:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # solves and executes equations and components to solve the circuit
        for item in self.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
        # evaluates the resiudals at the current iteration
        res = [equa.residual() for equa in self.res_equa]
        res += [equa.solve() for equa in self.loop_breaker_equa]
        try:
            # res += [((self.design_equa[key].solve() + self.design_equa[key].S.value) * self.design_equa[key].scale_factor)
            #     if self.design_equa[key].relaxed
            #     else (self.design_equa[key].solve() * self.design_equa[key].scale_factor)
            #     for key in self.design_equa]
            # res += [0.5 * sum((x[len(self.Vt) + len(self.U):]) ** 2)]

            for key in self.design_equa:
                if not self.design_equa[key].relaxed:
                    res += [(self.design_equa[key].solve() * self.design_equa[key].scale_factor)]
                else:
                    pass

            res += [sum([(self.design_equa[key].solve() * self.design_equa[key].scale_factor) ** 2
                    if self.design_equa[key].relaxed else 0.0 for key in self.design_equa])]

        except Exception as e:
            print(e)
            [self.components[key].reset() for key in self.components]
        [self.components[key].reset() for key in self.components]
        return res

    def econ(self, x):

        i = 0
        # sets current iteration value to tearing variables
        for var in self.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to input variables
        for var in self.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to slack variables
        for var in self.S:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # solves and executes equations and components to solve the circuit
        for item in self.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()

        # evaluates the resiudals at the current iteration
        res = [equa.residual() for equa in self.res_equa]
        res += [equa.solve() for equa in self.loop_breaker_equa]
        try:
            # res += [((self.design_equa[key].solve() + self.design_equa[key].S.value) * self.design_equa[key].scale_factor)
            #     if self.design_equa[key].relaxed
            #     else (self.design_equa[key].solve() * self.design_equa[key].scale_factor)
            #     for key in self.design_equa]
            for key in self.design_equa:
                if not self.design_equa[key].relaxed:
                    res += [(self.design_equa[key].solve() * self.design_equa[key].scale_factor)]
                else:
                    pass
        except Exception as e:
            print(e)
            [self.components[key].reset() for key in self.components]
        # resets all components
        [self.components[key].reset() for key in self.components]
        return np.array(res)

    def sim_fun(self, x):

        """
        Calculate the residuals of the system equations
        :param x: iteration variables
        :return:  residuals
        """

        # sets current iteration value to tearing variables
        for i, var in enumerate(self.Vt):
            var.set_value(x[i] / var.scale_factor)

        for item in self.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
                if item.status == 0:
                    [self.components[key].reset() for key in self.components]
                    return [], 0

        res = [equa.residual() for equa in self.res_equa]
        res += [equa.solve() for equa in self.loop_breaker_equa]
        try:
            for key in self.design_equa:
                if not self.design_equa[key].relaxed:
                    res += [(self.design_equa[key].solve() * self.design_equa[key].scale_factor)]
                else:
                    pass
        except Exception as e:
            print(f'Error in design equation calculation: {e}')
            [self.components[key].reset() for key in self.components]
            return [], 0

        [self.components[key].reset() for key in self.components]

        return np.array(res)

    def iecon(self, x):

        i = 0
        # sets current iteration value to tearing variables
        for var in self.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to input variables
        for var in self.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # sets current iteration values to slack variables
        for var in self.S:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # solves and executes equations and components to solve the circuit
        for item in self.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()

        con = [self.components[key].ports[port].m.value for key in self.components for port in
               self.components[key].ports]

        [self.components[key].reset() for key in self.components]

        return np.array(con)

    def solve(self, x):
        i = 0
        # sets current iteration value to tearing variables
        for var in self.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        # solves and executes equations and components to solve the circuit
        for item in self.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.diagramm_plot = False
                item.solve()

    def jacobian(self, x):
        for k in range(len(self.Vt) + len(self.U)):
            i = 0
            for var in self.Vt:
                if i == k:
                    var.set_value(DualNumber(x[i] / var.scale_factor, 1.0 / var.scale_factor))
                else:
                    var.set_value(DualNumber(x[i] / var.scale_factor, 0.0))
                i += 1

            for var in self.U:
                if i == k:
                    var.set_value(DualNumber(x[i] / var.scale_factor, 1.0 / var.scale_factor))
                else:
                    var.set_value(DualNumber(x[i] / var.scale_factor, 0.0))
                i += 1

            for item in self.exec_list:
                if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                    item.solve()
                elif isinstance(item, Component):
                    item.lamda = 1.0
                    item.diagramm_plot = False
                    item.calculate_derivatives = True
                    item.solve()
                    item.calculate_derivatives = False
            dr = [equa.residual().der for equa in self.res_equa]
            try:
                for key in self.design_equa:
                    if not self.design_equa[key].relaxed:
                        dr += [(self.design_equa[key].solve() * self.design_equa[key].scale_factor).der]
                    else:
                        pass
            except Exception as e:
                print(e)
                [self.components[key].reset() for key in self.components]
            # resets all components
            [self.components[key].reset() for key in self.components]
            if k == 0:
                J = np.array(dr)
            else:
                J = np.column_stack([J, np.array(dr)])
        return J


def tearing_alg(tpg: TripartiteGraph):
    """
    Applies a tearing algorithm to identify a set of independent tearing variables and residual equations
    required to solve a system of equations of a thermal hydraulic system, and determines the order in which
    the components and equations should be executed and solved.

    :param tpg  : TripartiteGraph which describes the connections between system variables, components and equations.
    :return     : A tuple containing the following information:
            - Vt: The set of independent system tearing variables.
            - exec_list: A list of system components to be executed in the induced order.
            - res_equa: A list of system residual equations.
            - design_equa: The number of design equations required to close the system of equations.
    """

    Vt, V, Vnew = [], [], []
    comp_exec = []
    comp_not_exec = tpg.C.copy()
    equa_solved = []
    exec_list = []
    res_equa = []

    for comp in tpg.C:
        if isinstance(comp, Source):
            comp_exec += [comp]
            exec_list += [comp]
            for key in comp.ports:
                V += [comp.ports[key].p, comp.ports[key].h, comp.ports[key].m]

    while len(comp_exec) != len(tpg.C):
        if any(c.component_type in ['Compressor', 'Pump'] for c in comp_not_exec):
            indices = [i for i, c in enumerate(comp_not_exec)
                       if c.component_type in ['Compressor', 'Pump']]
            for i, index in enumerate(indices):
                c = comp_not_exec[index]
                if i == 0:
                    id = index
                    n = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
                else:
                    if len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V]) < n:
                        id = index
                        n = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
        else:
            for i, c in enumerate(comp_not_exec):
                if not isinstance(c, Sink):
                    n_new = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
                    if i == 0:
                        c_old = c
                        n = n_new
                        id = i
                    else:
                        if n_new < n:
                            c_old = c
                            n = n_new
                            id = i
                        elif n_new == n:
                            if isinstance(c, PressureBasedComponent) and isinstance(c_old, MassFlowBasedComponent):
                                c_old = c
                                n = n_new
                                id = i
                            else:
                                pass
                else:
                    pass

        Vt += [e[0] for e in tpg.Ed if e[1] == comp_not_exec[id] and e[0] not in V]
        V += [e[0] for e in tpg.Ed if e[1] == comp_not_exec[id] and e[0] not in V]
        n_solved = len(equa_solved)
        n_executed = len(comp_exec)
        while True:
            for u in list(set(tpg.U).difference(set(equa_solved))):
                if isinstance(u.variables[0], list):
                    if len(set([item for sublist in u.variables
                                for item in sublist]).difference(set([item for sublist in u.variables
                                                                      for item in sublist]).intersection(set(V)))) == 1:
                        equa_solved.append(u)
                        exec_list.append(u)
                        V.extend(list(set([item for sublist in u.variables
                                           for item in sublist]).difference(set([item for sublist in u.variables
                                                                                 for item in sublist]).intersection(
                            set(V))).difference(set(V))))
                else:
                    if len(set(u.variables).difference(set(u.variables).intersection(set(V)))) == 1:
                        equa_solved.append(u)
                        exec_list.append(u)
                        V.extend(list(set(u.variables).difference(set(V))))

            comp_new = [c for c in comp_not_exec
                        if c not in comp_exec
                        and all(v in V for v in [e[0] for e in tpg.Ed if e[1] == c])]
            Vnew.extend([e[1] for e in tpg.Ed if e[0] in comp_new])
            for u in tpg.U:
                v_list = []
                for variable in u.variables:
                    if isinstance(variable, list):
                        for v in variable:
                            v_list.append(v)
                    else:
                        v_list.append(variable)
                if any(v in V and v in Vnew for v in v_list) or (any(v in Vnew for v in v_list)
                                                                 and all(v in V or v in Vnew for v in v_list)
                                                                 and any(v in Vt for v in v_list)):
                    res_equa.append(u)

            V.extend(Vnew)
            comp_exec.extend(comp_new)
            comp_not_exec = [c for c in comp_not_exec if c not in comp_exec]
            exec_list.extend(comp_new)
            Vnew = []
            if len(comp_exec) == n_executed and len(equa_solved) == n_solved:
                break
            else:
                n_executed = len(comp_exec)
                n_solved = len(equa_solved)
    if len(res_equa) < len(Vt):
        design_equa = [True, len(Vt) - len(res_equa)]
    else:
        design_equa = [False, 0]
    return Vt, exec_list, res_equa, design_equa


def source_search(junc_no: int, var: Variable, jpcm, components: List[Component]):
    """
    Search for sources in the junction matrix based on specified conditions.

    Parameters:
        junc_no (int): Junction number.
        var: Variable object.
        jpcm (array-like): junction port connectivity matrix.
        components (list): List of component objects.

    Returns:
        int or empty list: Component index if source is found, otherwise an empty list.
    """

    jpcm = np.array(jpcm)
    comp_list = [k for k, value in enumerate(jpcm[junc_no - 1, :-2])
                 if value != 0
                 and np.sign(value) != np.sign(var.port_id[2])
                 and k + 1 != var.port_id[1]]
    for c in comp_list:
        if jpcm[junc_no - 1, -1] == 1:
            return c + 1
        elif components[c].component_type not in ['Compressor', 'Pump', 'Expansion Valve']:
            for j, value in enumerate(jpcm[:, c]):
                if value != 0 and jpcm[junc_no - 1, c] * value < 0 and jpcm[junc_no - 1, -2] == jpcm[j, -2]:
                    if components[c].source_component:
                        return c + 1
                    elif j != junc_no - 1:
                        return source_search(j, c)
    return []


def circuit_res_objfun(clone, x):
    return clone.res_objfun(x)


def circuit_objfun(clone, x):
    return clone.objfun(x)


def circuit_econ(clone, x):
    return clone.econ(x)


def circuit_iecon(clone, x):
    return clone.iecon(x)


def circuit_sim_fun(clone, x):
    return clone.sim_fun(x)


def system_solver(circuit: Circuit):
    """
    solves thermal hydraulic cycles using broydens-method and newton-homotopy pseudo-arc-length continuation
    :param x0:                  iteration start variables
    :param circuit:             Circuit object
    :return:                    solution of the system

    """

    def fun(x):

        """
        Calculate the residuals of the system equations
        :param x: iteration variables
        :return:  residuals
        """

        # sets current iteration value to tearing variables
        i = 0
        for i, var in circuit.Vt:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for var in circuit.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
                if item.status == 0:
                    [circuit.components[key].reset() for key in circuit.components]
                    return [], 0

        res = [equa.residual() for equa in circuit.res_equa]
        try:
            res += [circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor for key in circuit.design_equa]
        except:
            [circuit.components[key].reset() for key in circuit.components]
            return [], 0

        [circuit.components[key].reset() for key in circuit.components]

        return np.array(res), 1

    def sim_fun(x):

        """
        Calculate the residuals of the system equations
        :param x: iteration variables
        :return:  residuals
        """

        # sets current iteration value to tearing variables
        for i, var in enumerate(circuit.Vt):
            var.set_value(x[i] / var.scale_factor)

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
                if item.status == 0:
                    [circuit.components[key].reset() for key in circuit.components]
                    return [], 0

        res = [equa.residual() for equa in circuit.res_equa]
        try:
            for key in circuit.design_equa:
                if not circuit.design_equa[key].relaxed:
                    res += [(circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor)]
                else:
                    pass
        except Exception as e:
            print(f'Error in design equation calculation: {e}')
            [circuit.components[key].reset() for key in circuit.components]
            return [], 0

        [circuit.components[key].reset() for key in circuit.components]

        return np.array(res), 1

    def jacobian_forward(fun, x0):

        """
        Calculate the Jacobian matrix of the residual equations using the forward difference scheme.

        Args:
            fun: A function that calculates the residuals and returns a tuple (f, convergence_flag),
                 where f is the array of residuals and convergence_flag indicates whether all model execution were successful.
            x0: The values of the iteration variables from the last iteration.

        Returns:
            A list [J, convergence_flag] containing the Jacobian matrix and the convergence flag.
            J: The Jacobian matrix.
            convergence_flag: A flag indicating whether the model execution was successful.

        """

        f_0, convergence_flag = fun(x0)

        if convergence_flag == 0:
            return [], convergence_flag

        J = np.zeros((len(f_0), len(x0)))
        epsilon = 1e-6 * np.maximum(np.abs(x0), 1e-3)
        for j in range(len(x0)):
            x = x0.copy()
            x[j] += epsilon[j]
            f_fw, convergence_flag = fun(x)

            if convergence_flag == 0:
                return J, convergence_flag

            J[:, j] = (f_fw - f_0) / epsilon[j]

        return J, convergence_flag

    def objfun(x):

        # y = 0.5 * sum((x[len(circuit.Vt) + len(circuit.U):]) ** 2)

        if not np.array_equal(x, circuit.xlast):

            i = 0
            # sets current iteration value to tearing variables
            for var in circuit.Vt:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # sets current iteration values to input variables
            for var in circuit.U:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # sets current iteration values to slack variables
            for var in circuit.S:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # solves and executes equations and components to solve the circuit
            for item in circuit.exec_list:
                if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                    item.solve()
                elif isinstance(item, Component):
                    item.lamda = 1.0
                    item.solve()

            # evaluates the resiudals at the current iteration
            circuit.reslast = [equa.residual() for equa in circuit.res_equa]
            circuit.reslast += [equa.solve() for equa in circuit.loop_breaker_equa]
            try:
                # res += [((self.design_equa[key].solve() + self.design_equa[key].S.value) * self.design_equa[key].scale_factor)
                #     if self.design_equa[key].relaxed
                #     else (self.design_equa[key].solve() * self.design_equa[key].scale_factor)
                #     for key in self.design_equa]
                for key in circuit.design_equa:
                    if not circuit.design_equa[key].relaxed:
                        circuit.reslast += [(circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor)]
                    else:
                        pass
                circuit.flast = sum([(circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor) ** 2
                        if circuit.design_equa[key].relaxed else 0.0 for key in circuit.design_equa])
            except Exception as e:
                print(e)
                [circuit.components[key].reset() for key in circuit.components]
            # resets all components
            [circuit.components[key].reset() for key in circuit.components]
            circuit.xlast = x.copy()
        return circuit.flast

    def econ(x):

        if not np.array_equal(x, circuit.xlast):

            circuit.xlast = x.copy()

            i = 0
            # sets current iteration value to tearing variables
            for var in circuit.Vt:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # sets current iteration values to input variables
            for var in circuit.U:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # sets current iteration values to slack variables
            for var in circuit.S:
                var.set_value(x[i] / var.scale_factor)
                i += 1

            # solves and executes equations and components to solve the circuit
            for item in circuit.exec_list:
                if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                    item.solve()
                elif isinstance(item, Component):
                    item.lamda = 1.0
                    item.solve()

            # evaluates the resiudals at the current iteration
            circuit.reslast = [equa.residual() for equa in circuit.res_equa]
            circuit.reslast += [equa.solve() for equa in circuit.loop_breaker_equa]
            try:
                # res += [((self.design_equa[key].solve() + self.design_equa[key].S.value) * self.design_equa[key].scale_factor)
                #     if self.design_equa[key].relaxed
                #     else (self.design_equa[key].solve() * self.design_equa[key].scale_factor)
                #     for key in self.design_equa]
                for key in circuit.design_equa:
                    if not circuit.design_equa[key].relaxed:
                        circuit.reslast += [(circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor)]
                    else:
                        pass
                circuit.flast = sum([(circuit.design_equa[key].solve() * circuit.design_equa[key].scale_factor) ** 2
                        if circuit.design_equa[key].relaxed else 0.0 for key in circuit.design_equa])
            except Exception as e:
                print(e)
                [circuit.components[key].reset() for key in circuit.components]
            # resets all components
            [circuit.components[key].reset() for key in circuit.components]

            f_0 = circuit.flast
            x_new = x + np.diag([1e-6 * max(np.abs(x[i]), 1) for i, var in enumerate(circuit.Vt + circuit.U + circuit.S)])
            items = [(circuit_clones[j], x_new[j]) for j in range(len(x_new))]
            f_fw = pool.starmap(circuit_objfun, items)
            circuit.grad_objfun = np.transpose(np.array(f_fw) - np.array(f_0)) / np.diag(x_new - x)
        print(f'Max Norm Residuals: {max(np.abs(circuit.reslast))}')
        return np.array(circuit.reslast)

    def grad_objfun(pool, circuit_clones, x):
        #     grad = np.zeros(len(x))
        #     grad[len(circuit.Vt) + len(circuit.U):] = x[len(circuit.Vt) + len(circuit.U):]
        #     return grad
        f_0 = circuit_clones[0].objfun(x)
        x_new = x + np.diag([1e-6 * max(np.abs(x[i]), 1) for i, var in enumerate(circuit.Vt + circuit.U + circuit.S)])
        items = [(circuit_clones[j], x_new[j]) for j in range(len(x_new))]
        f_fw = pool.starmap(circuit_objfun, items)
        return np.transpose(np.array(f_fw) - np.array(f_0)) / np.diag(x_new - x)

    def grad_econ(pool, circuit_clones, x):
        f_0 = circuit_clones[0].econ(x)
        x_new = x + np.diag([1e-6 * max(np.abs(x[i]), 1) for i, var in enumerate(circuit.Vt + circuit.U + circuit.S)])
        items = [(circuit_clones[j], x_new[j]) for j in range(len(x_new))]
        f_fw = pool.starmap(circuit_econ, items)
        J = np.transpose(f_fw - f_0) / np.diag(x_new - x)
        return J

    def grad_iecon(pool, circuit_clones, x):
        f_0 = circuit_clones[0].iecon(x)
        x_new = x + np.diag([1e-6 * max(np.abs(x[i]), 1) for i, var in enumerate(circuit.Vt + circuit.U + circuit.S)])
        items = [(circuit_clones[j], x_new[j]) for j in range(len(x_new))]
        f_fw = pool.starmap(circuit_iecon, items)
        J = np.transpose(f_fw - f_0) / np.diag(x_new - x)
        return J

    def grad_sim_fun(pool, circuit_clones, x):
        try:
            f_0 = circuit_clones[0].sim_fun(x)
            x_new = x + np.diag([1e-6 * max(np.abs(x[i]), 1) for i in range(len(x))])
            items = [(circuit_clones[j], x_new[j]) for j in range(len(x_new))]
            f_fw = pool.starmap(circuit_sim_fun, items)
            J = np.transpose(f_fw - f_0) / np.diag(x_new - x)
            return J, 1
        except Exception as e:
            return [], 0

    def adjoint_derivative(pool, circuit_clones, u, x0):
        global L_global
        L, x, convergence_flag = simulation(u, x0)
        if convergence_flag == 1:
            X = np.append(x, u)
            f_0 = circuit.res_objfun(X)
            X_new = X + np.diag([1e-6 * max(np.abs(X[i]), 1.0) for i, var in enumerate(circuit.Vt + circuit.U + circuit.S)])
            items = [(circuit_clones[j], X_new[j]) for j in range(len(X_new))]
            f_fw = pool.starmap(circuit_res_objfun, items)
            J = np.transpose(np.array(f_fw) - np.array(f_0)) / np.diag(X_new - X)
            dr_dx = J[0:len(circuit.Vt), 0:len(circuit.Vt)]
            dL_dx = J[-1, 0:len(circuit.Vt)]
            dr_du = J[0:len(circuit.Vt), len(circuit.Vt):]
            dL_du = J[-1, len(circuit.Vt):]
            psi = np.linalg.solve(dr_dx.T, dL_dx)
            phi = np.linalg.solve(dr_dx, -dr_du)
            DL_Du = dL_du - np.matmul(psi, dr_du)
            return L, DL_Du, phi, x, convergence_flag
        else:
            return [], [], [], [], convergence_flag

    def linear_fun(x):

        """
        Calculate the residuals of the linearized system
        :param x: iteration variables
        :return: [residuals, convergence_flag]
        """

        # sets current iteration value to tearing variables
        i = 0
        for i, var in enumerate(circuit.Vt):
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for var in circuit.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 0.0
                item.solve()
                if item.status == 0:
                    [circuit.components[key].reset() for key in circuit.components]
                    return [[], 0]

        res = [equa.residual() for equa in circuit.res_equa]
        res += [equa.solve() for equa in circuit.loop_breaker_equa]
        try:
            res += [equa.solve() for equa in circuit.design_equa]
        except:
            [circuit.components[key].reset() for key in circuit.components]
            return [[], 0]

        [circuit.components[key].reset() for key in circuit.components]

        return [np.array(res), 1]

    def linear_homotopy_fun(x):
        """
        Calculate the residuals of the homotopy system equations.

        Args:
            x: Iteration variables.

        Returns:
            A list containing residuals and convergence flag.

        """

        # sets current iteration value to tearing variables
        i = 0
        for i, var in enumerate(circuit.Vt):
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for var in circuit.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 0.0
                item.solve()
                if item.status == 0:
                    [circuit.components[key].reset() for key in circuit.components]
                    return [[], 0]

        res = [equa.residual() for equa in circuit.res_equa]
        res += [equa.solve() for equa in circuit.loop_breaker_equa]
        try:
            res += [equa.solve() for equa in circuit.design_equa]
        except:
            [circuit.components[key].reset() for key in circuit.components]
            return [[], 0]

        [circuit.components[key].reset() for key in circuit.components]

        return [np.array(res), 1]

    def newton_homotopy_fun(res_0, x):

        # sets current iteration value to tearing variables
        i = 0
        for i, var in enumerate(circuit.Vt):
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for var in circuit.U:
            var.set_value(x[i] / var.scale_factor)
            i += 1

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
                if item.status == 0:
                    [circuit.components[key].reset() for key in circuit.components]
                    return [[], 0]

        res = [equa.residual() for equa in circuit.res_equa]
        res += [equa.solve() for equa in circuit.loop_breaker_equa]
        try:
            res += [equa.solve() for equa in circuit.design_equa]
        except:
            [circuit.components[key].reset() for key in circuit.components]
            return [[], 0]

        [circuit.components[key].reset() for key in circuit.components]

        return [np.array(np.array(res) - (1 - x[-1]) * np.array(res_0)), 1]

    def augmented_system(fun, x0, tau, ds, x):
        """
        Calculate the augmented system for the arc-length continuation method.

        Args:
            fun: Function of residual equations with embedded homotopy parameter.
            x0: Values of iteration variables from the last iteration.
            tau: Tangential vector.
            ds: Arc-length step-size.
            x: Iteration variables.

        Returns:
            A list containing function values and convergence flag.

        """
        F, convergence_flag = fun(x)
        if convergence_flag == 0:
            return [[], 0]
        else:
            F = np.append(F, np.dot(tau[0:-1], x[0:-1] - x0[0:-1]) + tau[-1] * (x[-1] - x0[-1]) - ds)
            return [F, 1]

    def arc_length_equa(x0, tau, ds, x):
        """
        Calculate the arc-length equation.

        Args:
            x0: Values of iteration variables from the last iteration.
            tau: Tangential vector.
            ds: Arc-length step-size.
            x: Iteration variables.

        Returns:
            A list containing function values and convergence flag.

        """
        return [np.array([np.dot(tau[0:-1], x[0:-1] - x0[0:-1]) + tau[-1] * (x[-1] - x0[-1]) - ds]), 1]

    def simulation(u, x0):
        for i, var in enumerate(circuit.U):
            var.set_value(u[i] / var.scale_factor)
        for clone in circuit_clones:
            for i, var in enumerate(clone.U):
                var.set_value(u[i] / var.scale_factor)
        try:
            sol = mixed_newton_broyden_method(sim_fun, max_iter=50, epsilon=1e-5, print_convergence=True, x0=x0)
        except Exception as e:
            print(f'Error in objective function evaluation: {e}')
            return float('inf'), x0, 0
        if sol['converged']:
            x = np.append(sol['x'], u)
            return circuit.objfun(x), x, 1
        else:
            return float('inf'), x0, 0

    def bfgs_update(H_inv, s, y):
        rho = 1.0 / np.dot(y, s)
        I = np.eye(len(s))
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H_inv = np.dot(A, np.dot(H_inv, B)) + rho * np.outer(s, s)
        return H_inv

    def line_search(pool, circuit_clones, L, DL_Du, u, x, phi, p, lower_bound, upper_bound, alpha=1.0, n_fails=0, max_fails=5, mu1=1e-4, mu2=1.0):

        alpha_lower = np.maximum(np.linalg.solve(np.diag(p + 1e-9), np.array(lower_bound) - u), 0)
        alpha_upper = np.maximum(np.linalg.solve(np.diag(p + 1e-9), np.array(upper_bound) - u), 0)

        # Check if both alpha_lower and alpha_upper are empty
        if any(alpha_lower > 0):
            alpha_clip = min(alpha_lower[alpha_lower > 0])
        else:
            alpha_clip = 1
        if any(alpha_upper > 0):
            alpha_clip = min(alpha_clip, min(alpha_upper[alpha_upper > 0]))
        else:
            pass
            # At least one of the arrays is non-empty, proceed with calculating alpha_clip
        if alpha_clip < alpha:
            alpha = alpha_clip * 0.1

        while True:
            u_new = u + alpha * p
            x0 = x + np.matmul(phi, u_new - u)
            L_new, DL_Du_new, phi_new, x_new, convergence_flag = adjoint_derivative(pool, circuit_clones, u_new, x0)
            if convergence_flag == 1 and L_new <= L[-1] + mu1 * alpha * np.dot(DL_Du[-1], p):
                if np.linalg.norm(np.dot(DL_Du_new, p)) <= mu2 * np.linalg.norm(np.dot(DL_Du[-1], p)) or n_fails > max_fails:
                    n_fails = 0
                    print(f'Objective Value: {L_new}, Gradient Norm: {np.linalg.norm(DL_Du_new) }')
                    alpha *= 1.5
                    return u_new, L_new, DL_Du_new, phi_new, x_new, alpha, n_fails
                else:
                    n_fails += 1
            else:
                n_fails += 1
                alpha *= 0.5
                if n_fails > max_fails:
                    return u, L[-1], DL_Du[-1], phi, x, alpha, n_fails

    def BFGS_method(pool, circuit_clones, u0, x0):
        H_inv = np.eye(len(u0))
        u = [np.array(u0)]
        x = [np.array(x0)]
        L0, DL_Du0, phi0, x0, convergence_flag= adjoint_derivative(pool, circuit_clones, u0, x0)
        L = [L0]
        phi = phi0
        I = np.identity(len(u0))
        lower_bound = [var.bounds[0] * var.scale_factor for var in circuit_clones[0].U]
        upper_bound = [var.bounds[1] * var.scale_factor for var in circuit_clones[0].U]
        DL_Du = [DL_Du0]
        alpha = 1.0
        tol = 1e-6
        max_iter = 100
        max_fails = 3
        s = 1e12
        λ = 1.0
        for iteration in range(max_iter):
            print(f'Iteration: {iteration+1}')
            # p = -np.matmul(H_inv + λ * I, DL_Du[-1])
            p = -np.matmul(H_inv, DL_Du[-1])
            u_new, L_new, DL_Du_new, phi_new, x_new, alpha, n_fails = line_search(pool, circuit_clones, L, DL_Du, u[-1], x[-1], phi, p, lower_bound, upper_bound, alpha)
            if n_fails < max_fails:
                u.append(u_new)
                DL_Du.append(DL_Du_new)
                L.append(L_new)
                x.append(x_new)
                if np.abs(np.dot(DL_Du[-1] / np.linalg.norm(DL_Du[-1]), p / np.linalg.norm(p))) > 1e-3:
                    s = u[-1] - u[-2]
                    y = DL_Du[-1] - DL_Du[-2]
                    try:
                        H_inv = bfgs_update(H_inv, s, y)
                        cond_Hinv = np.linalg.cond(H_inv)
                        print(f'Condition Number Hessian: {cond_Hinv}')
                        if cond_Hinv < 100:
                            λ = max(1e-5, λ * 1e-1)
                        else:
                            λ = min(1e5, λ * 10)
                    except Exception as e:
                        print(e)
                        H_inv = np.eye(len(u0))
                else:
                    H_inv = np.eye(len(u0))
            else:
                print('Reset of Hessian...')
                H_inv = np.eye(len(u0))

            if np.linalg.norm(DL_Du_new) <= tol or np.linalg.norm(s) < 1e-9:
                break

        return {'x': np.append(x[-1], u[-1]), 'f': L[-1], 'convergence': True}

    def Levenberg_Marquardt(pool, circuit_clones, u0, x0):
        H_inv = np.eye(len(u0))
        u = [np.array(u0)]
        x = [np.array(x0)]
        L0, DL_Du0, phi0, x0, convergence_flag= adjoint_derivative(pool, circuit_clones, u0, x0)
        L = [L0]
        phi = phi0
        lower_bound = [var.bounds[0] * var.scale_factor for var in circuit_clones[0].U]
        upper_bound = [var.bounds[1] * var.scale_factor for var in circuit_clones[0].U]
        DL_Du = [DL_Du0]
        alpha = 1.0
        tol = 1e-6
        max_iter = 100
        I = np.identity(len(u0))
        s = 1e12
        λ = 1.0
        mu1 = 0.01
        mu2 = 0.9
        n_fails = 0
        max_fails = 20
        for it in range(max_iter):

            print(f'Iteration: {it+1}')
            alpha = 1
            while True:
                p = -np.matmul(H_inv + λ * I, DL_Du[-1])

                alpha_lower = np.maximum(np.linalg.solve(np.diag(p + 1e-9), np.array(lower_bound) - u[-1]), 0)
                alpha_upper = np.maximum(np.linalg.solve(np.diag(p + 1e-9), np.array(upper_bound) - u[-1]), 0)

                # Check if both alpha_lower and alpha_upper are empty
                if not alpha_lower.any() and not alpha_upper.any():
                    # Both arrays are empty, handle this case (you can set alpha_clip to a default value)
                    pass
                else:
                    if not alpha_lower[alpha_lower > 0].any() and not alpha_upper[alpha_upper > 0].any():
                        # At least one of the arrays is non-empty, proceed with calculating alpha_clip
                        alpha_clip = min(min(alpha_lower[alpha_lower > 0]), min(alpha_upper[alpha_upper > 0]))
                        if alpha_clip < alpha:
                            alpha = alpha_clip
                            print('One value got clipped to boundary value!')

                u_new = u[it] + alpha * p
                x0 = x[-1] + np.matmul(phi, u_new - u[it])
                L_new, DL_Du_new, phi_new, x_new, convergence_flag = adjoint_derivative(pool, circuit_clones, u_new, x0)
                if n_fails <= max_fails:
                    if convergence_flag == 1:
                        if L_new <= L[-1] + mu1 * alpha * np.dot(DL_Du[-1], p):
                            # if np.linalg.norm(np.dot(DL_Du_new, p)) <= np.linalg.norm(np.dot(DL_Du[-1], p)):
                            print(f'Objective Value: {L_new}, Gradient Norm: {np.linalg.norm(DL_Du_new) }')
                            u += [u_new]
                            L += [L_new]
                            DL_Du += [DL_Du_new]
                            phi = phi_new
                            x += [x_new]
                            print(f'lamda got decreased {λ} => {max(λ * 1e-1, 1e-12)}')
                            λ = max(λ * 1e-1, 1e-12)
                            n_fails = 0
                            break
                        else:
                            n_fails += 1
                            alpha *= 0.5
                            print(f'lamda got increased {λ} => {min(λ * 10, 1e9)}')
                            λ = min(λ * 10, 1e12)
                    else:
                        alpha *= 0.5
                        n_fails += 1
                else:
                    u += [u[-1]]
                    L += [L[-1]]
                    DL_Du += [DL_Du[-1]]
                    x += [x[-1]]
                    n_fails = 0
                    break

            s = u[it+1] - u[it]
            if np.linalg.norm(DL_Du[it+1]) <= tol or np.linalg.norm(s) < 1e-9:
                break
            y = DL_Du[it+1] - DL_Du[it]
            H_inv = bfgs_update(H_inv, s, y)
            cond_Hinv = np.linalg.cond(H_inv)
            print(f'Condition Number Hessian: {cond_Hinv}')

        return {'x': np.append(x[-1], u[-1]), 'f': L[-1], 'convergence': True}

    def broyden_method(fun, max_iter, epsilon, print_convergence, x0):

        """
        Implements the Broyden method for solving a system of nonlinear equations.

        Args:
            fun: A function that calculates the residuals of the equations and returns a tuple (F, convergence_flag),
                 where F is the array of residuals and convergence_flag indicates whether the model execution was successful.
            J: The initial Jacobian matrix.
            max_iter: The maximum number of iterations allowed.
            epsilon: The convergence criteria threshold.
            x0: The initial values of the iteration variables.

        Returns:
            A dictionary with the following keys:
                - 'x': The final values of the iteration variables.
                - 'f': The final residuals.
                - 'n_it': The number of iterations performed.
                - 'converged': A boolean indicating whether the solver converged.
                - 'message': A message describing the outcome of the solver.

        """
        x = [x0]
        it = 0
        λmin = 1e-9
        F, convergence_flag = fun(x[0])

        if convergence_flag == 0:
            print('Broyden-Solver stopped due to failure in model execution!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'model execution error': True}

        J, convergence_flag = jacobian_forward(fun, x0)
        if convergence_flag == 0:
            print('Broyden-Solver stopped due to failure in model execution!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'failed to compute initial jacobian!': True}

        F_array = []

        while np.linalg.norm(F, 2) > epsilon:
            λ = 1.0
            try:
                dx = np.linalg.solve(J, -F)
                x_new = x[it] + λ * dx
            except:
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'singular jacobian!'}
            while any([(x_new[i] < var.bounds[0] * var.scale_factor or x_new[i] > var.bounds[1] * var.scale_factor) for
                       i, var in enumerate(circuit.Vt)]) and λ > λmin:
                λ *= 1 / 2
                x_new = x[it] + λ * dx
            x.append(x[it] + λ * dx)
            newF, convergence_flag = fun(x[-1])

            if convergence_flag == 0:
                print('Broyden-Solver stopped due to failure in model execution!')
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'model execution error'}

            dF = newF - F
            J = J + np.outer(dF - np.dot(J, dx), dx) / np.dot(dx, dx)
            F = newF
            F_array.append(np.max(np.abs(F[:-1])))
            it += 1
            if print_convergence:
                print(f'Iteration {it} / Max-Norm of Residuals: {np.max(np.abs(F))}')

            if it > max_iter:
                break

        if np.max(np.abs(F)) > epsilon:
            print('Convergence criteria not reached within allowed number of iteration steps!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False,
                    'message': 'convergence criteria not reached within allowed number of iteration steps'}
        else:
            print('Broyden Solver converged!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': True, 'message': 'solver converged'}

    def mixed_newton_broyden_method(fun, max_iter, epsilon, print_convergence, x0):
        """
        Implements a Mixed Newton-Broyden method for solving a system of nonlinear equations.

        Args:
            fun: A function that calculates the residuals of the equations and returns a tuple (F, convergence_flag),
                 where F is the array of residuals and convergence_flag indicates whether the model execution was successful.
            J: The initial Jacobian matrix.
            max_iter: The maximum number of iterations allowed.
            epsilon: The convergence criteria threshold.
            x0: The initial values of the iteration variables.

        Returns:
            A dictionary with the following keys:
                - 'x': The final values of the iteration variables.
                - 'f': The final residuals.
                - 'n_it': The number of iterations performed.
                - 'converged': A boolean indicating whether the solver converged.
                - 'message': A message describing the outcome of the solver.

        """

        x = [x0]
        it = 0
        n_fails = 0
        max_fails = 10
        λmin = 1e-9

        F, convergence_flag = fun(x[0])
        if convergence_flag == 0:
            print('Broyden-Solver stopped due to failure in model execution!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'model execution error': True}

        J, convergence_flag = grad_sim_fun(pool, circuit_clones, x0)
        # J, convergence_flag = jacobian_forward(fun, x0)
        if convergence_flag == 0:
            print('Broyden-Solver stopped due to failure in model execution!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'failed to compute initial jacobian!': True}

        F_array = [np.max(F)]
        improved = True
        while np.max(np.abs(F)) > epsilon or it > max_iter or n_fails > max_fails:
            λ = 1.0
            try:
                dx = np.linalg.solve(J, -F)
                x_new = x[it] + λ * dx
            except:
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'singular jacobian!'}
            while any([(x_new[i] < var.bounds[0] * var.scale_factor or x_new[i] > var.bounds[1] * var.scale_factor) for
                       i, var in enumerate(circuit.Vt)]) and λ > λmin:
                λ *= 1 / 2
                x_new = x[it] + λ * dx
            x.append(x[it] + λ * dx)
            newF, convergence_flag = fun(x[-1])

            if convergence_flag == 0:
                print('Broyden-Solver stopped due to failure in model execution!')
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'model execution error'}
            if improved:
                dF = newF - F
                J = J + np.outer(dF - np.dot(J, dx), dx) / np.dot(dx, dx)
            else:
                n_fails += 1
                J, convergence_flag = grad_sim_fun(pool, circuit_clones, x0)
                # J, convergence_flag = jacobian_forward(fun, x[-1])
            if convergence_flag == 0:
                print('Broyden-Solver stopped due to failure in model execution!')
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'model execution error'}

            F = newF
            F_array.append(np.max(np.abs(F)))
            it += 1
            if abs(F_array[-1]) < abs(F_array[-2]):
                improved = True
            else:
                improved = False
            if print_convergence:
                print(f'Iteration {it} / Max-Norm of Residuals: {np.max(np.abs(F[:-1]))}')
            if it > max_iter:
                break
            elif n_fails > max_fails:
                break

        if np.linalg.norm(F, 2) > epsilon and it > max_iter:
            # print('Convergence criteria not reached within allowed number of iteration steps!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False,
                    'message': 'convergence criteria not reached within allowed number of iteration steps'}
        if np.linalg.norm(F, 2) and n_fails > max_fails:
            # print('maximum number of failed improvement reached!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False,
                    'message': 'maximum number of failed improvement reached!'}
        else:
            print('Broyden Solver converged!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': True, 'message': 'solver converged'}

    def system_linearization(x):

        print('linearizing system components...')

        # sets current iteration value to tearing variables
        for i, var in enumerate(circuit.Vt):
            var.set_value(x[i] / var.scale_factor)

        for item in circuit.exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.solve()

        for key in circuit.components:
            circuit.components[key].x0 = []
            for port in circuit.components[key].ports:
                if isinstance(circuit.components[key], PressureBasedComponent):
                    if port.port_type == 'in':
                        circuit.components[key].x0.append(port.p.value)
                        circuit.components[key].x0.append(port.h.value)
                    elif port.port_type == 'out':
                        circuit.components[key].x0.append(port.p.value)
                elif isinstance(circuit.components[key], MassFlowBasedComponent):
                    if port.port_type == 'in':
                        circuit.components[key].x0.append(port.p.value)
                        circuit.components[key].x0.append(port.h.value)
                        circuit.components[key].x0.append(port.m.value)
                else:
                    pass

        [(circuit.components[key].jacobian(),
          setattr(circuit.components[key], 'linearized', True)) for key in circuit.components
         if not (isinstance(circuit.components[key], BypassComponent)
                 and isinstance(circuit.components[key], Source)
                 and isinstance(circuit.components[key], Sink))]
        [circuit.components[key].reset() for key in circuit.components]

    def pseudo_arc_length_continuation(x0, type):
        """
        Perform pseudo arc-length continuation for solving a system of equations.

        Args:
            x0: Initial values of the iteration variables.

        Returns:
            A dictionary containing the final solution, convergence status, and additional information.

        """

        ds = [1.0]  # Initial arc-length step-size
        lamda_min = 1e-6  # Minimum lambda value
        lamda_max = 0.5  # Maximum lambda value
        lamda_final = 1.0  # Final lambda value
        tol = 1e-2  # Tolerance between setpoint and actual lambda value
        epsilon = 1e-6  # Convergence criteria for Broyden solver
        max_fails = 20  # Maximum number of allowed solver fails
        N_max_outer = 50  # Maximum number of outer iterations
        N_max = 20  # Maximum number of allowed Broyden step iterations
        N_opt = 10  # Setpoint number of Broyden step iterations
        a = 0.5  # Acceleration factor
        sigma = 1.0  # Scaling factor
        lamda_g = 0.5  # Initial value of scale factor sigma and periodic rescaling
        dlamda_ds_max = 0.5  # Value at which rescaling is invoked

        if type == 'linear homotopy':
            system_linearization(x0)
            J, convergence_flag = jacobian_forward(linear_fun, x0[0:-1])
            if convergence_flag == 0:
                return {'x': x0, 'converged': False, 'message': 'model execution error'}
            print('Executes Broyden Solver to solve linearized system...')
            sol = broyden_method(linear_fun, J, max_iter, epsilon, False, x0[0:-1])
            x = [np.append(sol['x'], 0.0)]
        elif type == 'newton homotopy':
            res_0 = fun(x0[0:-1])  # residuals of system at initial values
            if res_0[1] == 0:
                print('Broyden-Solver stopped due to failure in model execution!')
                return {'x': x0, 'converged': False, 'message': ' model execution error'}
            else:
                res_0 = res_0[0]
            x = [x0.copy()]

        it = 0
        while abs(x[it][-1] - 1) > tol:

            print('arc-length iteration-step:', it + 1, '/ lambda:', x[it][-1])

            print('computing Jacobian...')
            if type == 'linear homotopy':
                J, convergence_flag = jacobian_forward(linear_homotopy_fun, x[it])
            elif type == 'newton homotopy':
                J, convergence_flag = jacobian_forward(partial(newton_homotopy_fun, res_0), x[it])
            if convergence_flag == 0:
                return {'x': x[-1], 'converged': False, 'message': 'model execution error'}

            dx_dlamda = np.linalg.solve(J[:, 0:-1], -J[:, -1])
            dlamda_ds = (1 + np.linalg.norm(dx_dlamda) ** 2) ** (-0.5)

            if abs(dlamda_ds) > dlamda_ds_max:
                sigma = dlamda_ds / np.sqrt(lamda_g) * np.sqrt((1 - lamda_g) / (1 - dlamda_ds ** 2))
                dlamda_ds = (1 + sigma ** 2 * np.linalg.norm(dx_dlamda) ** 2) ** (-0.5)

            if it > 0:
                if np.sign(dlamda_ds) != np.sign(
                        sigma ** 2 * np.dot(dx_dlamda, x[it][0:-1] - x[it - 1][0:-1]) + x[it][-1] - x[it - 1][-1]):
                    dlamda_ds = -dlamda_ds

            dx_ds = dx_dlamda * dlamda_ds
            tau = np.append(dx_ds, dlamda_ds)

            if it == 0:
                if tau[-1] <= 0:
                    tau = tau * (-1)

            n_fails = 0
            while True:
                if type == 'linear homotopy':
                    partial_system = partial(augmented_system, linear_homotopy_fun, x[it], tau, ds[it])
                elif type == 'newton homotopy':
                    partial_system = partial(augmented_system, partial(newton_homotopy_fun, res_0), x[it], tau, ds[it])
                J_arc_length_eq, convergence_flag = jacobian_forward(partial(arc_length_equa, x[it], tau, ds[it]),
                                                                     x[it])
                if convergence_flag == 0:
                    return {'x': x[-1], 'converged': False, 'message': 'model execution error'}
                J_augmented = np.vstack((J, J_arc_length_eq))
                print('Executes Broyden Solver for corrector-step...')
                sol = broyden_method(partial_system, J_augmented, N_max, epsilon, False, x[it] + tau * ds[it])
                N_it = sol['n_it']
                converged = sol['converged']
                if (sol['x'][-1] - lamda_final) > tol:
                    if abs(tau[-1] * ds[it]) > lamda_min:
                        print('Reducing arc-length-step-size: ' + str(round(ds[it], 6)) + ' -> ' + str(
                            round(ds[it] / 2, 6)))
                        ds[it] /= 2
                    else:
                        print('Solver stopped: minimum arc-length-step-size reached')
                        return {'x': x[-1], 'converged': False, 'message': 'minimum arc length step size reached'}
                    n_fails += 1
                elif n_fails > max_fails:
                    print('Solver stopped: maximum number of allowed fails reached')
                    sol = broyden_method(fun, J, max_iter, epsilon, True, x[-1][0:-1])
                    if sol['converged']:
                        return sol
                    else:
                        return {'x': x[-1], 'converged': False, 'message': 'maximum number of allowed fails reached'}
                elif abs(tau[-1] * ds[it]) < lamda_min:
                    print('Solver stopped: minimum arc-length-step-size reached')
                    print('Try to solve system with Broyden Solver with last iteration '
                          'values of Pseudo-Arc-Length Solver...')
                    sol = broyden_method(fun, J, max_iter, 1e-12, True, x[-1][0:-1])
                    if sol['converged']:
                        return sol
                    else:
                        return {'x': x[-1], 'converged': False, 'message': 'minimum arc length step size reached'}
                elif it > N_max_outer:
                    print('Solver stopped: maximum number of iterations reached')
                    print('Try to solve system with Broyden Solver with last iteration values of'
                          ' Pseudo-Arc-Length Solver...')
                    sol = broyden_method(fun, J, max_iter, epsilon, True, x[-1][0:-1])
                    if sol['converged']:
                        return sol
                    else:
                        return {'x': x[-1], 'converged': False, 'message': 'maximum number of iterations reached'}
                elif converged:
                    print('Corrector-step successful!')
                    if n_fails < 1 and N_it < N_opt:
                        if tau[-1] * ds[it] < lamda_max:
                            ds.append(ds[it] * (1 + a * (N_opt / N_it) ** 2))
                            print('Increasing arc-length-step-size: ' + str(round(ds[it], 6)) + ' -> ' + str(
                                round(ds[it] * (1 + a * (N_opt / N_it) ** 2), 6)))
                        else:
                            ds.append(lamda_max / abs(tau[-1]))
                    else:
                        print('Reducing arc-length step-size: ' + str(round(ds[it], 6)) + ' -> ' + str(
                            round(ds[it] / 2, 6)))
                        ds.append(ds[it] / 2)
                    x.append(sol['x'])
                    it += 1
                    break
                else:
                    if abs(tau[-1] * ds[it]) > lamda_min:
                        print('Reducing arc-length step-size: ' + str(round(ds[it], 6)) + ' -> '
                              + str(round(ds[it] / 2, 6)))
                        ds[it] /= 2
                    n_fails += 1

        print('Successful tracking of solution curve!')

        if type in ['linear homotopy', 'linear newton-homotopy']:
            [setattr(circuit.components[key], 'linearized', False) for key in circuit.components]

        print('computing Jacobian... ')
        J, convergence_flag = jacobian_forward(fun, x[-1][0:-1])
        if convergence_flag == 0:
            return {'x': x[-1], 'converged': False, 'message': 'model execution error'}

        print('Executes Broyden Solver for solving actual system...')
        sol = broyden_method(fun, J, max_iter, 1e-6, True, x[-1][0:-1])
        return sol

    # maximum number of iterations for broyden solver
    max_iter = 100

    # convergence criteria
    epsilon = 1e-6

    # adds inputs if there are parameters set to inputs
    circuit.add_inputs()


    Vt_bnds = [(0.1e5, 5e5),
               (3e5, 6e5),
               (5.1e5, 30e5),
               (5.1e5, 30e5),
               (1e5, 3e5),
               (0.1e5, 5e5)]

    # with open('init.pkl', 'rb') as load_data:
    #     init = pickle.load(load_data)
    #     i = 0
    #     for var in circuit.Vt:
    #         var.initial_value = init[i]
    #         var.bounds = Vt_bnds[i]
    #         i += 1
    #     for var in circuit.U:
    #         var.initial_value = init[i]
    #         i += 1
        # for var in circuit.S:
        #     var.initial_value = init[i]
        #     i += 1

    x0_scaled = [0] * (len(circuit.Vt) + len(circuit.U) + len(circuit.S))
    bnds_scaled = [()] * (len(circuit.Vt) + len(circuit.U) + len(circuit.S))
    i = 0
    for var in circuit.Vt:
        x0_scaled[i] = var.initial_value * var.scale_factor
        bnds_scaled[i] = (var.bounds[0] * var.scale_factor, var.bounds[1] * var.scale_factor)
        i += 1
    for var in circuit.U:
        x0_scaled[i] = var.initial_value * var.scale_factor
        bnds_scaled[i] = (var.bounds[0] * var.scale_factor, var.bounds[1] * var.scale_factor)
        i += 1
    for var in circuit.S:
        x0_scaled[i] = var.initial_value * var.scale_factor
        bnds_scaled[i] = (var.bounds[0] * var.scale_factor, var.bounds[1] * var.scale_factor)
        i += 1

    if len(circuit.res_equa) + len(circuit.design_equa) + len(circuit.loop_breaker_equa) == len(circuit.Vt) + len(
            circuit.U) \
            and not any([True if circuit.design_equa[key].relaxed else False for key in circuit.design_equa]):

        # Runs Broyden solver, returning the solution array if the solver converges; otherwise, starts the Arc-Length Continuation solver.
        print('Start Broyden Solver \n'
              'computing Jacobian...')
        sol = mixed_newton_broyden_method(fun, max_iter, epsilon, True, x0_scaled)
        if sol['converged']:
            with open('init.pkl', 'wb') as save_data:
                pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
            return sol
        else:
            # x0_scaled = np.append(x0_scaled, 0.0)
            # print('Solver not converged!')
            # print('Start Pseudo-Arc-Length Continuation Solver')
            # sol = pseudo_arc_length_continuation(x0_scaled, 'newton homotopy')
            # if sol['converged']:
            #     print('Solver converged!')
            #     with open('init.pkl', 'wb') as save_data:
            #         pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
            # else:
            #     print('Solver not converged!')
            # for i, var in enumerate(circuit.Vt):
            #     var.initial_value = sol[0] / var.scale_factor
            # with open('init.pkl', 'wb') as save_data:
            #     pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
            # return sol
            print('Start SLSQP Algorithm to solve system')
            pool = multiprocessing.Pool(len(x0_scaled))
            circuit_clones = [deepcopy(circuit)] * len(x0_scaled)
            sol = scipy.optimize.fmin_slsqp(objfun,
                                            x0_scaled,
                                            bounds=bnds_scaled,
                                            f_eqcons=econ,
                                            fprime=partial(grad_objfun, pool, circuit_clones),
                                            fprime_eqcons=partial(grad_econ, pool, circuit_clones),
                                            full_output=True,
                                            disp=3,
                                            acc=1.0e-6,
                                            epsilon=1.0e-05,
                                            iter=1000)
            if sol[3] == 0:
                print('Solver converged!')
                i = 0
                for var in circuit.Vt:
                    var.initial_value = sol[0][i] / var.scale_factor
                    i += 1
                for var in circuit.U:
                    var.initial_value = sol[0][i] / var.scale_factor
                    i += 1
                for var in circuit.S:
                    var.initial_value = sol[0][i] / var.scale_factor
                    i += 1
                with open('init.pkl', 'wb') as save_data:
                    pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
                return {'x': sol[0], 'f': circuit.econ(sol[0]), 'n_it': sol[2], 'converged': True,
                        'message': 'solver converged'}
            else:
                return {'x': sol[0], 'converged': False, 'message': 'model execution error'}

    else:
        pool = multiprocessing.Pool(len(circuit.Vt))
        circuit_clones = [deepcopy(circuit)] * len(x0_scaled)
        global x0_global, L_global
        x0_global = x0_scaled[0:len(circuit.Vt)]
        L_global = 1e12
        print('Start L-BFGS-B Algorithm to solve system')
        # sol = Levenberg_Marquardt(pool, circuit_clones, x0_scaled[len(circuit.Vt):], x0_scaled[0:len(circuit.Vt)])
        # sol = BFGS_method(pool, circuit_clones, x0_scaled[len(circuit.Vt):], x0_scaled[0:len(circuit.Vt)])
        # x0_scaled = sol['x']
        # if sol['converged']:
        #     x = np.append(sol['x'])
        #     print('Solver converged!')
        #     i = 0
        #     for var in circuit.Vt:
        #         var.initial_value = x[i] / var.scale_factor
        #         i += 1
        #     for var in circuit.U:
        #         var.initial_value = x[i] / var.scale_factor
        #         i += 1
        #     for var in circuit.S:
        #         var.initial_value = x[i] / var.scale_factor
        #         i += 1
        #     with open('init.pkl', 'wb') as save_data:
        #         pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
        #     return {'x': x, 'f': circuit.econ(x), 'n_it': sol['nit'], 'converged': True,
        #             'message': 'solver converged'}
        # else:
        #     return {'x': [], 'converged': False, 'message': 'model execution error'}
        print('Start SLSQP Algorithm to solve system')
        sol = scipy.optimize.fmin_slsqp(objfun,
                                        x0_scaled,
                                        bounds=bnds_scaled,
                                        f_eqcons=econ,
                                        fprime=partial(grad_objfun, pool, circuit_clones),
                                        fprime_eqcons=partial(grad_econ, pool, circuit_clones),
                                        full_output=True,
                                        disp=3,
                                        acc=1.0e-6,
                                        iter=1000)

        # sol = scipy.optimize.fmin_slsqp(objfun,
        #                                 x0_scaled,
        #                                 bounds=bnds_scaled,
        #                                 f_eqcons=circuit.econ,
        #                                 fprime=partial(grad_objfun, pool, circuit_clones),
        #                                 fprime_eqcons=circuit.jacobian,
        #                                 full_output=True,
        #                                 disp=3,
        #                                 acc=1.0e-6,
        #                                 iter=1000)

        if sol[3] == 0:
            print('Solver converged!')
            i = 0
            for var in circuit.Vt:
                var.initial_value = sol[0][i] / var.scale_factor
                i += 1
            for var in circuit.U:
                var.initial_value = sol[0][i] / var.scale_factor
                i += 1
            for var in circuit.S:
                var.initial_value = sol[0][i] / var.scale_factor
                i += 1
            with open('init.pkl', 'wb') as save_data:
                pickle.dump([var.initial_value for var in circuit.Vt + circuit.U + circuit.S], save_data)
            return {'x': sol[0], 'f': circuit.econ(sol[0]), 'n_it': sol[2], 'converged': True,
                    'message': 'solver converged'}
        else:
            return {'x': sol[0], 'converged': False, 'message': 'model execution error'}


def logph(h: List[list], p: List[list], no: List[list], fluids: List[str]):
    """
    Plot the pressure-enthalpy diagram for a given fluid.

    Args:
        h (array-like): List or array lists of specific enthalpy values.
        p (array-like): List or array of lists of pressure values.
        no: (array-like): List or array of list of state numbers
        fluids (str): Name of the fluid.

    Returns:
        None
    """

    for i, fluid in enumerate(fluids):

        Fig = [plt.subplots(figsize=(12, 8))] * len(fluids)
        Tmin = PropsSI('TMIN', fluid)
        Pmin = PropsSI('PMIN', fluid)
        Tcrit = PropsSI("Tcrit", fluid)
        Pcrit = PropsSI("Pcrit", fluid)
        Ts = [Tmin]
        while Ts[-1] < Tcrit:
            T = Ts[-1] + 1
            if T >= Tcrit:
                Ts.append(Tcrit)
            else:
                Ts.append(T)

        Ps = PropsSI("P", "T", Ts, "Q", 0, fluid) / 100000
        Hs = PropsSI("H", "T", Ts, "Q", 0, fluid) / 1000
        Ht = PropsSI("H", "T", Ts, "Q", 1, fluid) / 1000
        T = np.linspace(Tmin + 1, Tcrit, 20)

        Fig[i][1].plot(Hs, Ps, 'k')
        Fig[i][1].plot(Ht, Ps, 'k')

        for j in range(len(T)):
            P1 = np.linspace(1, PropsSI('P', 'T', T[j], 'Q', 1, fluid) - 1, 1000)
            H1 = PropsSI('H', 'T', T[j], 'P', P1, fluid)
            P2 = PropsSI('P', 'T', T[j], 'Q', 1, fluid)
            P1 = np.append(P1, P2)
            H2 = PropsSI('H', 'T', T[j], 'Q', 1, fluid)
            H1 = np.append(H1, H2)
            P3 = PropsSI('P', 'T', T[j], 'Q', 0, fluid)
            P1 = np.append(P1, P3)
            H3 = PropsSI('H', 'T', T[j], 'Q', 0, fluid)
            H1 = np.append(H1, H3)
            P4 = np.linspace(PropsSI('P', 'T', T[j], 'Q', 0, fluid) + 1, Pcrit + 10000000, 1000)
            P1 = np.append(P1, P4)
            H4 = PropsSI('H', 'T', T[j], 'P', P4, fluid)
            H1 = np.append(H1, H4)
            Fig[i][1].plot(H1 / 1000, P1 / 100000, 'b', linewidth=0.7,
                           label='T=' + str(int(T[j] - 273.15)) + '°C')

        P = np.linspace(Pmin, Pcrit + 1e8, 1000)
        T = [Tcrit + j * 10 for j in range(1, 20)]
        for j in range(len(T)):
            H = PropsSI('H', 'P', P, 'T', T[j], fluid)
            Fig[i][1].plot(np.array(H) / 1e3, P / 1e5, 'b', linewidth=0.7,
                           label='T=' + str(int(T[j] - 273.15)) + '°C')

        labelLines(Fig[i][1].get_lines(), align=True, fontsize=7, backgroundcolor='none')

        for line in h[i]:
            if isinstance(line, list):
                for j in range(len(h[i])):
                    Fig[i][1].plot(h[i][j], p[i][j], "r-o")
                    k = 0
                    for x, y in zip(h[i][j], p[i][j]):
                        plt.annotate(str(no[i][j][k]),
                                     (x, y),
                                     textcoords="offset points",
                                     xytext=(0, 8),
                                     ha='right',
                                     color="red",
                                     fontweight="bold")
                        k += 1
                break
            else:
                Fig[i][1].plot(h[i], p[i], "r-o")
                k = 0
                for x, y in zip(h[i], p[i]):
                    plt.annotate(str(no[i][k]),
                                 (x, y),
                                 textcoords="offset points",
                                 xytext=(0, 8),
                                 ha='right',
                                 color="red",
                                 fontweight="bold")
                    k += 1
                break
        if isinstance(h[i][0], list):
            Fig[i][1].set_xlim([min(min(h[i][:])) - 100, max(max(h[i][:])) + 100])
            Fig[i][1].set_ylim([min(min(p[i][:])) - 10 ** np.floor(np.log10(min(min(p[i][:])))),
                                Pcrit * 1e-5 + 10 ** np.floor(np.log10(Pcrit * 1e-4))])
        else:
            Fig[i][1].set_xlim([min(h[i]) - 100, max(h[i]) + 100])
            Fig[i][1].set_ylim([min(p[i]) - 10 ** np.floor(np.log10(min(p[i]))),
                                Pcrit * 1e-5 + 10 ** np.floor(np.log10(Pcrit * 1e-4))])

        Fig[i][1].set_xlabel('specific Enthalpy / kJ/kg', fontweight='bold')
        Fig[i][1].set_ylabel('Pressure / bar', fontweight='bold')
        Fig[i][1].set_yscale('log')
        Fig[i][1].grid(True)
        plt.title(fluid)
        plt.draw()
        plt.show()


def coolpropsHTP(T, p, fluid):
    def calculate_derivative(T, p, fluid, prop, var, const):
        if PhaseSI('T', T, 'P', p, fluid) in ['twophase']:
            if var == 'T':
                delta = 1e-6 * max(abs(T), 1.0)
                prop_plus = PropsSI(prop, var, T + delta, "P", p, fluid)
                prop_minus = PropsSI(prop, var, T - delta, "P", p, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            else:
                delta = 1e-6 * max(abs(p), 1.0)
                prop_plus = PropsSI(prop, const, T, var, p + delta, fluid)
                prop_minus = PropsSI(prop, const, T, var, p - delta, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            return derivative
        else:
            return PropsSI(f"d({prop})/d({var})|{const}", 'T', T, "P", p, fluid)
    if isinstance(T, DualNumber) and isinstance(p, DualNumber):
        h_no = PropsSI("H", "T", T.no, "P", p.no, fluid)
        dH_dT = calculate_derivative(T.no, p.no, fluid, "H", "T", "P")
        dH_dP = calculate_derivative(T.no, p.no, fluid, "H", "P", "T")
        h_der = dH_dT * T.der + dH_dP * p.der
        return DualNumber(h_no, h_der)
    elif not isinstance(T, DualNumber) and isinstance(p, DualNumber):
        h_no = PropsSI("H", "T", T, "P", p.no, fluid)
        dH_dP = calculate_derivative(T, p.no, fluid, "H", "P", "T")
        h_der = dH_dP * p.der
        return DualNumber(h_no, h_der)
    elif isinstance(T, DualNumber) and not isinstance(p, DualNumber):
        h_no = PropsSI("H", "T", T.no, "P", p, fluid)
        dH_dT = calculate_derivative(T.no, p, fluid, "H", "T", "P")
        h_der = dH_dT * T.der
        return DualNumber(h_no, h_der)
    else:
        return PropsSI("H", "T", T, "P", p, fluid)


def coolpropsHTQ(T, Q, fluid):
    if isinstance(T, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "T", T.no, "Q", Q.no, fluid), PropsSI("d(H)/d(T)|sigma", "T", T.no, "Q", Q.no, fluid) * T.der + (PropsSI('H', 'T', T.no, 'Q', 1.0, 'R134a') - PropsSI('H', 'T', T.no, 'Q', 0.0, fluid)) * Q.der)
    elif not isinstance(T, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "T", T, "Q", Q.no, fluid), (PropsSI('H', 'T', T.no, 'Q', 1.0, 'R134a') - PropsSI('H', 'T', T.no, 'Q', 0.0, fluid)) * Q.der)
    elif isinstance(T, DualNumber) and not isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "T", T.no, "Q", Q, fluid), PropsSI("d(H)/d(T)|sigma", "T", T.no, "Q", Q, fluid) * T.der)
    else:
        return PropsSI("H", "T", T, "Q", Q, fluid)


def coolpropsHPQ(p, Q, fluid):
    if isinstance(P, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "P", p.no, "Q", Q.no, fluid), PropsSI("d(H)/d(P)|sigma", "P", p.no, "Q", Q.no, fluid) * p.der + (PropsSI('H', 'p', p.no, 'Q', 1.0, 'R134a') - PropsSI('H', 'P', p.no, 'Q', 0.0, fluid)) * Q.der)
    elif not isinstance(p, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "P", p, "Q", Q.no, fluid), (PropsSI('H', 'P', p.no, 'Q', 1.0, 'R134a') - PropsSI('H', 'P', p.no, 'Q', 0.0, fluid)) * Q.der)
    elif isinstance(p, DualNumber) and not isinstance(Q, DualNumber):
        return DualNumber(PropsSI("H", "P", p.no, "Q", Q, fluid), PropsSI("d(H)/d(T)|sigma", "P", p.no, "Q", Q, fluid) * p.der)
    else:
        return PropsSI("H", "P", p, "Q", Q, fluid)


def coolpropsTPH(p, h, fluid):
    def calculate_derivative(p, h, fluid, prop, var, const):
        if PhaseSI('P', p, 'H', h, fluid) in ['twophase']:
            if var == 'P':
                delta = 1e-6 * max(abs(p), 1.0)
                prop_plus = PropsSI(prop, var, p + delta, const, h, fluid)
                prop_minus = PropsSI(prop, var, p - delta, const, h, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            else:
                delta = 1e-6 * max(abs(h), 1.0)
                prop_plus = PropsSI(prop, const, p, var, h + delta, fluid)
                prop_minus = PropsSI(prop, const, p, var, h - delta, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            return derivative
        else:
            return PropsSI(f"d({prop})/d({var})|{const}", 'P', p, "H", h, fluid)

    if isinstance(p, DualNumber) and isinstance(h, DualNumber):
        T_no = PropsSI("T", "P", p.no, "H", h.no, fluid)
        dT_dP = calculate_derivative(p.no, h.no, fluid, "T", "P", "H")
        dT_dH = calculate_derivative(p.no, h.no, fluid, "T", "H", "P")
        T_der = dT_dP * p.der + dT_dH * h.der
        return DualNumber(T_no, T_der)

    elif not isinstance(p, DualNumber) and isinstance(h, DualNumber):
        T_no = PropsSI("T", "P", p, "H", h.no, fluid)
        dT_dH = calculate_derivative(p, h.no, fluid, "T", "H", "P")
        T_der = dT_dH * h.der
        return DualNumber(T_no, T_der)

    elif isinstance(p, DualNumber) and not isinstance(h, DualNumber):
        T_no = PropsSI("T", "P", p.no, "H", h, fluid)
        dT_dP = calculate_derivative(p.no, h, fluid, "T", "P", "H")
        T_der = dT_dP * p.der
        return DualNumber(T_no, T_der)

    else:
        return PropsSI("T", "P", p, "H", h, fluid)


def coolpropsTPQ(p, Q, fluid):
    if isinstance(p, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("T", "P", p.no, "Q", Q.no, fluid), PropsSI("d(D)/d(T)|sigma", "P", p.no, "Q", Q.no, fluid) * p.der + PropsSI("d(T)/d(H)|sigma", "P", p.no, "Q", Q.no, fluid) * Q.der)
    elif not isinstance(p, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("T", "P", p, "Q", Q.no, fluid), 0.0)
    elif isinstance(p, DualNumber) and not isinstance(Q, DualNumber):
        return DualNumber(PropsSI("T", "P", p.no, "Q", Q, fluid), PropsSI("d(T)/d(P)|sigma", "P", p.no, "Q", Q, fluid) * p.der)
    else:
        return PropsSI("T", "P", p, "Q", Q, fluid)


def coolpropsDPT(p, T, fluid):
    if isinstance(p, DualNumber) and isinstance(T, DualNumber):
        return DualNumber(PropsSI("D", "P", p.no, "T", T.no, fluid), PropsSI("d(D)/d(P)|T", "P", p.no, "T", T.no, fluid) * p.der + PropsSI("d(D)/d(T)|P", "P", p.no, "T", T.no, fluid) * T.der)
    elif not isinstance(p, DualNumber) and isinstance(T, DualNumber):
        return DualNumber(PropsSI("D", "P", p, "T", T.no, fluid), PropsSI("d(D)/d(T)|P", "P", p, "T", T.no, fluid) * T.der)
    elif isinstance(p, DualNumber) and not isinstance(T, DualNumber):
        return DualNumber(PropsSI("D", "P", p.no, "T", T, fluid), PropsSI("d(D)/d(P)|H", "P", p.no, "T", T, fluid) * p.der)
    else:
        return PropsSI("D", "P", p, "T", T, fluid)


def coolpropsDPH(p, h, fluid):
    def calculate_derivative(p, h, fluid, prop, var, const):
        if PhaseSI('P', p, 'H', h, fluid) in ['twophase']:
            if var == 'P':
                delta = 1e-6 * max(abs(p), 1.0)
                prop_plus = PropsSI(prop, var, p + delta, const, h, fluid)
                prop_minus = PropsSI(prop, var, p - delta, const, h, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            else:
                delta = 1e-6 * max(abs(h), 1.0)
                prop_plus = PropsSI(prop, const, p, var, h + delta, fluid)
                prop_minus = PropsSI(prop, const, p, var, h - delta, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            return derivative
        else:
            return PropsSI(f"d({prop})/d({var})|{const}", 'P', p, "H", h, fluid)

    if isinstance(p, DualNumber) and isinstance(h, DualNumber):
        D_no = PropsSI("D", "P", p.no, "H", h.no, fluid)
        dD_dP = calculate_derivative(p.no, h.no, fluid, "D", "P", "H")
        dD_dH = calculate_derivative(p.no, h.no, fluid, "D", "H", "P")
        D_der = dD_dP * p.der + dD_dH * h.der
        return DualNumber(D_no, D_der)

    elif not isinstance(p, DualNumber) and isinstance(h, DualNumber):
        D_no = PropsSI("D", "P", p, "P", h.no, fluid)
        dD_dH = calculate_derivative(p, h.no, fluid, "D", "H", "P")
        D_der = dD_dH * h.der
        return DualNumber(D_no, D_der)

    elif isinstance(p, DualNumber) and not isinstance(h, DualNumber):
        D_no = PropsSI("D", "P", p.no, "H", h, fluid)
        dD_dP = calculate_derivative(p.no, h, fluid, "D", "P", "H")
        D_der = dD_dP * p.der
        return DualNumber(D_no, D_der)

    else:
        return PropsSI("D", "P", p, "H", h, fluid)


def coolpropsDPQ(p, Q, fluid):
    if isinstance(p, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("D", "P", p.no, "Q", Q.no, fluid), PropsSI("d(D)/d(P)|sigma", "P", p.no, "Q", Q.no, fluid) * p.der + (1/PropsSI('D', 'P', p.no, 'Q', 1.0, 'R134a') - 1/PropsSI('D', 'P', p.no, 'Q', 0.0, 'R134a')) / (1/PropsSI('D', 'P', p.no, 'Q', Q.no, fluid)) * Q.der)
    elif not isinstance(p, DualNumber) and isinstance(Q, DualNumber):
        return DualNumber(PropsSI("D", "P", p, "Q", Q.no, fluid), (1/PropsSI('D', 'P', p.no, 'Q', 1.0, 'R134a') - 1/PropsSI('D', 'P', p.no, 'Q', 0.0, 'R134a')) / (1/PropsSI('D', 'P', p.no, 'Q', Q.no, fluid)) * Q.der)
    elif isinstance(p, DualNumber) and not isinstance(Q, DualNumber):
        return DualNumber(PropsSI("D", "P", p.no, "Q", Q, fluid), PropsSI("d(D)/d(P)|sigma", "P", p.no, "Q", Q, fluid) * p.der)
    else:
        return PropsSI("D", "P", p, "H", Q, fluid)


def coolpropsCPH(p, h, fluid):
    def calculate_derivative(p, h, fluid, prop, var, const):
        if PhaseSI('P', p, 'H', h, fluid) in ['twophase']:
            if var == 'P':
                delta = 1e-6 * max(abs(p), 1.0)
                prop_plus = PropsSI(prop, var, p + delta, const, h, fluid)
                prop_minus = PropsSI(prop, var, p - delta, const, h, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            else:
                delta = 1e-6 * max(abs(h), 1.0)
                prop_plus = PropsSI(prop, const, p, var, h + delta, fluid)
                prop_minus = PropsSI(prop, const, p, var, h - delta, fluid)
                derivative = (prop_plus - prop_minus) / (2 * delta)
            return derivative
        else:
            return PropsSI(f"d({prop})/d({var})|{const}", 'P', p, "H", h, fluid)

    if isinstance(p, DualNumber) and isinstance(h, DualNumber):
        C_no = PropsSI("C", "P", p.no, "H", h.no, fluid)
        dC_dP = calculate_derivative(p.no, h.no, fluid, "C", "P", "H")
        dC_dH = calculate_derivative(p.no, h.no, fluid, "C", "H", "P")
        C_der = dC_dP * p.der + dC_dH * h.der
        return DualNumber(C_no, C_der)

    elif not isinstance(p, DualNumber) and isinstance(h, DualNumber):
        C_no = PropsSI("C", "P", p, "H", h.no, fluid)
        dC_dH = calculate_derivative(p, h.no, fluid, "C", "H", "P")
        C_der = dC_dH * h.der
        return DualNumber(C_no, C_der)

    elif isinstance(p, DualNumber) and not isinstance(h, DualNumber):
        C_no = PropsSI("C", "P", p.no, "H", h, fluid)
        dC_dP = calculate_derivative(p.no, h, fluid, "C", "P", "H")
        C_der = dC_dP * p.der
        return DualNumber(C_no, C_der)

    else:
        return PropsSI("C", "P", p, "H", h, fluid)