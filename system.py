"""
Created on Thu Jan 19 07:14:57 2023

@author: Mirco Ganz

Accompany the Master Thesis:

M. Ganz (2023) Numerical Modeling and Analysis of an Adaptive Refrigeration Cycle Simulator
"""

import csv
from CoolProp.CoolProp import PropsSI
import numpy as np
from functools import partial
from labellines import labelLines
import matplotlib.pyplot as plt
from typing import List


# dictionary to convert junction port connectivity matrix string values to integer values
psd = {'0': 0,
       'p': 1, '-p': -1,
       'h': 2, '-h': -2,
       'c': 3, '-c': -3,
       'g': 4, '-g': -4,
       'l': -5, '-l': -5,
       'm1': 6, '-m1': -6,
       'm2': 7, '-m2': -7
       }


class Variable:

    """
    Represents a variable in the system.

    Attributes:
        name (str): Name of the variable.
        port_typ (str): Port type.
        port_id (list): Port ID.
        var_typ (str): Variable type.
        value (float): Value of the variable.
        known (bool): Indicates if the variable value is known.
    """

    def __init__(self, name: str, port_typ: str, port_id: list, var_typ: str, initial_value: float):
        """
        Initialize a Variable object.

        Args:
            name: Name of the variable.
            port_typ: Port type.
            port_id: Port ID.
            var_typ: Variable type.
            initial_value: Initial value of the variable.
        """
        self.name = name
        self.port_typ = port_typ
        self.port_id = port_id
        self.var_typ = var_typ
        self.value = initial_value
        self.known = False

    def set_value(self, value: float):
        """
        Set the value of the variable.

        Args:
            value: Value to set.
        """
        self.value = value
        self.known = True

    def reset(self):
        """
        Reset the variable to its initial state.
        """
        self.known = False
        self.value = None


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

    def __init__(self, port_id: list, port_typ: str):
        """
        Initialize a Port object.

        Args:
            port_id: Port ID.
            port_typ: Port type.
        """
        self.port_id = port_id
        self.port_typ = port_typ
        self.fluid = None
        self.p = Variable(f"p({self.port_typ})", self.port_typ, self.port_id, "p", 1)
        self.h = Variable(f"h({self.port_typ})", self.port_typ, self.port_id, "h", 1)
        self.m = Variable(f"m({self.port_typ})", self.port_typ, self.port_id, "m", 1)


class Component:

    """
    A base class representing a component in a system.

    Attributes:
        boundary_typ (str): The boundary type of the component. Default value is "undefined".
        number (int): The component number.
        component_typ (str): The component type.
        name (str): The component name.
        source_component (bool): Indicates if the component is a source component.
        fluid_loop_list (list): The list of fluid loops associated with the component.
        executed (bool): Indicates if the component has been executed.
        ports (list): The list of ports associated with the component.
        specifications (list): The list of specifications for the component.
        parameter (list): The list of parameters for the component.
        inputs (list): The inputs of the component.
        outputs (list): The outputs of the component.
        solver_path (str): The path to the solver for the component.
        status (int): The status of the component.
        lamda (float): The lambda value of the component.
        linearized(bool): Indicates if the linearized component is used in solver
        J (list): The J values of the component.
        F0 (list): The F0 values of the component.
        x0 (list): The x0 values of the component.
        no_in_ports (int): The number of input ports.
        no_out_ports (int): The number of output ports.
    """

    boundary_typ = "undefined"

    def __init__(self, number: int, component_typ: str, name: str, jpcm: list):
        """
        Initialize a Component object.

        Args:
            number: Component number.
            component_typ: Component type.
            name: Component name.
            jpcm: List of jpcm values.
        """
        self.number = number
        self.component_typ = component_typ
        self.name = name
        self.source_component = False
        self.fluid_loop_list = set()
        self.executed = False
        self.solved = False
        self.ports = []
        self.specifications = []
        self.parameter = []
        self.inputs = ["", None]
        self.outputs = ["", None]
        self.solver_path = ""
        self.status = 1
        self.lamda = 1.0
        self.linearized = False
        self.J = []
        self.F0 = []
        self.x0 = []
        self.no_in_ports = 0
        self.no_out_ports = 0

        for j, line in enumerate(jpcm):
            if line[self.number - 1] > 0:
                self.fluid_loop_list.add(line[-2])
                self.ports.append(Port([j + 1, self.number, line[self.number - 1], line[-2], line[-1]], "in"))
                if line[-1] == 1:
                    self.source_component = True
                else:
                    self.no_in_ports += 1
            elif line[self.number - 1] < 0:
                self.ports.append(Port([j + 1, self.number, line[self.number - 1], line[-2], line[-1]], "out"))
                self.no_out_ports += 1

        self.fluid_loop_list = list(self.fluid_loop_list)

    def reset(self):
        """
        Reset the Component object.
        """
        self.executed = False
        self.status = 1
        for port in self.ports:
            if port.port_id[-1] != 1:
                port.p.reset()
                port.h.reset()
                port.m.reset()


class PressureBasedComponent(Component):
    boundary_typ = "Pressure Based"

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

        This method calculates the Jacobian matrix of the mass flow based component using finite differences.
        The Jacobian matrix represents the partial derivatives of the component's output variables
        with respect to its input variables.

        This method updates the `J` attribute of the component.

        Returns:
            None
        """
        i = 0
        self.J = np.zeros([2 * self.no_out_ports + self.no_in_ports, 2 * self.no_in_ports + self.no_out_ports])
        self.F0 = np.zeros([2 * self.no_out_ports + self.no_in_ports])
        for port in self.ports:
            if port.port_typ == 'in' and port.port_id[-1] == 0:
                port.p.set_value(self.x0[i])
                port.h.set_value(self.x0[i+1])
                i += 2
            elif port.port_typ == 'out' and port.port_id[-1] == 0:
                port.h.set_value(self.x0[i])
                i += 1
        self.solve()
        i = 0
        for port in self.ports:
            if port.port_typ == 'in' and not port.port_id[-1] == 1:
                self.F0[i] = port.m.value
                i += 1
            elif port.port_typ == 'out' and not port.port_id[-1] == 1:
                self.F0[i] = port.h.value
                self.F0[i+1] = port.m.value
                i += 2

        i = 0
        for port in self.ports:
            if port.port_typ == 'in' and not port.port_id[-1] == 1:
                j = 0
                port.p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'in' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.m.value - self.F0[i]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 1
                    elif port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.m.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 2
                port.p.set_value(self.x0[i])
                i += 1

                j = 0
                port.h.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'in' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.m.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 1
                    elif port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.m.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 2
                port.h.set_value(self.x0[i])
                i += 1

            if port.port_typ == 'out' and not port.port_id[-1] == 1:
                j = 0
                port.p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'in' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.m.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 1
                    elif port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.h.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.m.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 2
                port.p.set_value(self.x0[i])
                i += 1

        self.reset()

    def count_input_unknowns(self) -> int:

        """
        Count the number of unknown variables in the input ports of the component.

        This method counts the number of unknown variables (not known or specified) in the input ports
        of the component. It is used to determine the number of equations required to solve the component.

        Returns:
            int: The number of unknown variables in the input ports.
        """

        n_unknown = 0
        for port in self.ports:
            if port.port_typ == "in":
                if not port.p.known:
                    n_unknown += 1
                if not port.h.known:
                    n_unknown += 1
            elif port.port_typ == "out":
                if not port.p.known:
                    n_unknown += 1
        return n_unknown

    def is_executable(self) -> bool:
        """
        Check if the component is ready for execution.

        This method checks if all the input ports of the component have known values, i.e., they are known or specified.
        It also checks if all the output ports have known values for the pressure variable.

        Returns:
            bool: True if the component is ready for execution, False otherwise.
        """
        return all((port.p.known and port.h.known)
                   for port in self.ports if port.port_typ == "in") and \
               all(port.p.known for port in self.ports if port.port_typ == "out")


class MassFlowBasedComponent(Component):
    boundary_typ = "Mass Flow Based"

    def execute(self):
        if not self.is_executable():
            raise RuntimeError(f"Tried to solve Component:  {self.component_typ}, but it is not executable yet!")
        self.executed = True
        for port in self.ports:
            if port.port_typ == "out":
                port.p.known = True
                port.h.known = True
                port.m.known = True

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

        This method updates the `J` attribute of the component.

        Returns:
            None
        """

        i = 0
        self.J = np.zeros([3 * self.no_out_ports, 3 * self.no_in_ports])
        self.F0 = np.zeros([3 * self.no_out_ports])
        for port in self.ports:
            if port.port_typ == 'in' and port.port_id[-1] == 0:
                port.p.set_value(self.x0[i])
                port.h.set_value(self.x0[i+1])
                port.m.set_value(self.x0[i+2])
                i += 3
        self.solve()
        i = 0
        for port in self.ports:
            if port.port_typ == 'out' and not port.port_id[-1] == 1:
                self.F0[i] = port.p.value
                self.F0[i+1] = port.h.value
                self.F0[i+2] = port.m.value
                i += 3

        i = 0
        for port in self.ports:
            if port.port_typ == 'in' and port.port_id[-1] == 0:
                j = 0
                port.p.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.h.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+2, i] = (port_inside.m.value - self.F0[j+2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.p.set_value(self.x0[i])
                i += 1

                j = 0
                port.h.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.h.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+2, i] = (port_inside.m.value - self.F0[j+2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.h.set_value(self.x0[i])
                i += 1

                j = 0
                port.m.set_value(self.x0[i] + 1e-6 * max(abs(self.x0[i]), 0.01))
                self.solve()
                for port_inside in self.ports:
                    if port_inside.port_typ == 'out' and not port_inside.port_id[-1] == 1:
                        self.J[j, i] = (port_inside.p.value - self.F0[j]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+1, i] = (port_inside.h.value - self.F0[j+1]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        self.J[j+2, i] = (port_inside.m.value - self.F0[j+2]) / (1e-6 * max(abs(self.x0[i]), 0.01))
                        j += 3
                port.m.set_value(self.x0[i])
                i += 1

        self.reset()

    def count_input_unknowns(self):
        """
        Count the number of unknown variables in the input ports of the component.

        This method counts the number of unknown variables (not known or specified) in the input ports
        of the component. It is used to determine the number of equations required to solve the component.

        Returns:
            int: The number of unknown variables in the input ports.
        """
        count = 0
        for port in self.ports:
            if port.port_typ == 'in':
                if not port.p.known:
                    count += 1
                if not port.h.known:
                    count += 1
                if not port.m.known:
                    count += 1
        return count

    def is_executable(self) -> bool:
        """
        Check if the component is ready for execution.

        This method checks if all the input ports of the component have known values, i.e., they are known or specified.
        It verifies that all the variables (p, h, and m) in the input ports are known.

        Returns:
            bool: True if the component is ready for execution, False otherwise.
        """
        return all(port.p.known and port.h.known and port.m.known for port in self.ports if port.port_typ == 'in')


class BypassComponent(Component):

    """
    Represents a Bypass Component.
    """

    boundary_typ = "Bypass"

    def solve(self):
        """
        Solve the bypass component.

        This method solves the bypass component by executing the solver function specified in the solver path.
        """
        MODULE_PATH = self.solver_path + "/__init__.py"
        with open(MODULE_PATH) as f:
            code = compile(f.read(), MODULE_PATH, 'exec')
        namespace = {}
        exec(code, namespace)
        namespace['solver'](self)

    def count_input_unknowns(self):
        """
        Count the number of unknowns in the input ports.

        Returns:
            int: The count of unknown variables in the input ports.
        """
        count = 0
        for port in self.ports:
            if port.port_typ == 'in':
                if not port.p.known:
                    count += 1
                if not port.h.known:
                    count += 1
                if not port.m.known:
                    count += 1
            else:
                if not port.p.known:
                    count += 1
        return count

    def is_executable(self) -> bool:
        """
        Check if the component is ready for execution.

        This method checks if all the input ports of the component have known values, i.e., they are known or specified.
        It verifies that all the variables (p, h, and m) in the input ports are known.

        Returns:
            bool: True if the component is ready for execution, False otherwise.
        """
        return all((port.p.known and port.h.known and port.m.known)
                   for port in self.ports if port.port_typ == "in") and \
               all(port.p.known for port in self.ports if port.port_typ == "out")


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

    def solve(self):
        """
        Solve the mass flow balance equation.

        This method calculates the unknown variable by summing the known variables and applying mass conservation.
        """
        m_total = 0
        unknown_variable = None
        for variable in self.variables:
            if variable.known:
                if variable.port_typ == 'out':
                    m_total += variable.value
                elif variable.port_typ == 'in':
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
        res = sum(variable.value for variable in self.variables if variable.port_typ == "out") \
              - sum(variable.value for variable in self.variables if variable.port_typ == "in")
        return res * 10


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

    def solve(self):
        """
        Solve the pressure equality equation.

        This method sets the unknown variable to the known pressure value.
        """
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
        return (self.variables[0].value - self.variables[1].value) * 1e-5


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

    def solve(self):
        """
        Solve the enthalpy equality equation.

        This method sets the unknown variable to the known enthalpy value.
        """
        if not self.is_solvable():
            raise RuntimeError("Tried to solve equation: " + self.name + ", but it is not solvable yet.")

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
        return (self.variables[0].value - self.variables[1].value) * 1e-5


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

        H_total = sum(variable[0].value * variable[1].value * self.port_typ_multiplier[variable[0].port_typ]
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
        res = sum(variable[0].value * variable[1].value * self.port_typ_multiplier[variable[0].port_typ]
                  for variable in self.variables)
        return res * 1e-5 * 10


class DesignEquation:
    """
    Represents a design equation for a component.

    Args:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.

    Attributes:
        component (Component): The component associated with the equation.
        DC_value (float): The design condition value.
        res (float): The calculated result of the equation.
    """

    def __init__(self, component: Component, DC_value: float, port_type: str, var_type: str, port_id: int):
        self.component = component
        self.DC_value = DC_value
        self.DC_port_type = port_type
        self.DC_var_type = var_type
        self.port_id = port_id
        self.res = float()
        self.J = []
        self.x0 = []
        self.F0 = []
        self.linearized = False


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

    def solve(self):
        """
        Solve the superheat equation.

        Returns:
            float: The residual value of the equation.
        """
        for port in self.component.ports:
            x = np.array([port.p.value, port.h.value])
            if port.port_id[2] == psd['-c']:
                T_sat = PropsSI("T", "P", port.p.value, "Q", 1.0, port.fluid)
                if self.DC_value < 1e-4:
                    h_SH = PropsSI("H", "P", port.p.value, "Q", 1.0, port.fluid)
                else:
                    h_SH = PropsSI("H", "P", port.p.value, "T", T_sat + self.DC_value, port.fluid)
                self.res = (port.h.value - h_SH) * 1e-5
                break
        if self.linearized:
            return self.component.lamda * self.res + (1 - self.component.lamda) * (np.dot(self.J, x - self.x0) + self.F0)
        else:
            return self.res
    
    def jacobian(self):
        self.J = np.zeros([2])
        for port in self.component.ports:
            if port.port_id[2] == psd['-c']:
                port.p.set_value(self.x0[0])
                port.h.set_value(self.x0[1])
                self.F0 = self.solve()
                port.p.set_value(self.x0[0] + (1e-6 * max(abs(self.x0[0]), 0.01)))
                self.J[0] = (self.solve() - self.F0) / (1e-6 * max(abs(self.x0[0]), 0.01))
                port.p.set_value(self.x0[0])
                port.h.set_value(self.x0[1] + (1e-6 * max(abs(self.x0[1]), 0.01)))
                self.J[1] = (self.solve() - self.F0) / (1e-6 * max(abs(self.x0[1]), 0.01))
                port.p.reset()
                port.h.reset()
                break


class SubcoolingEquation(DesignEquation):

    """
    Represents a subcooling equation for a component.
    """

    def solve(self):
        """
        Solve the subcooling equation.

        Returns:
            float: The residual value of the equation.
        """

        for port in self.component.ports:
            if port.port_id[2] == psd['-h']:
                T_sat = PropsSI("T", "P", port.p.value, "Q", 0.0, port.fluid)
                if self.DC_value < 1e-4:
                    h_SC = PropsSI("H", "P", port.p.value, 'Q', 0.0, port.fluid)
                else:
                    h_SC = PropsSI("H", "P", port.p.value, "T", T_sat - self.DC_value, port.fluid)
                self.res = (port.h.value - h_SC) * 1e-5
                break
        return self.res


class DesignParameterEquation(DesignEquation):

    """
    Represents a design parameter equation.

    Attributes:
        DC_var_typ (str): The variable type string.
        DC_port_typ (str): The port type string.
        res (float): The equation residual.
    """

    def solve(self):
        """
        Solve the design parameter equation.

        Returns:
            float: The residual value of the equation.
        """
        for port in self.component.ports:
            if port.p.var_typ == self.DC_var_type and port.p.port_typ == self.DC_port_type:
                self.res = (port.p.value - self.DC_value) * 1e-9
                break
        return self.res


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
            for port in ic.ports:
                if port.port_typ == 'out' and port.port_id[0] == self.number:
                    pressure_variables.append(port.p)
                    mass_flow_variables.append(port.m)
                    if len(self.in_comp) == 1:
                        enthalpy_variables.append(port.h)
                    enthalpy_flow_variables.append([port.m, port.h])

        for oc in self.out_comp:
            for port in oc.ports:
                if port.port_typ == 'in' and port.port_id[0] == self.number:
                    pressure_variables.append(port.p)
                    mass_flow_variables.append(port.m)
                    enthalpy_variables.append(port.h)
                    enthalpy_flow_variables.append([port.m, port.h])

        pressure_equations = []
        for pressure_variable in pressure_variables[1:]:
            pressure_equations.append(PressureEquality([pressure_variables[0], pressure_variable], self.fluid_loop))

        enthalpy_equations = []
        for enthalpy_variable in enthalpy_variables[1:]:
            enthalpy_equations.append(EnthalpyEquality([enthalpy_variables[0], enthalpy_variable], self.fluid_loop))

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

        for j, line in enumerate(jpcm):
            input_comp = [components[c] for c, i in enumerate(line[:-2]) if i < 0]
            output_comp = [components[c] for c, i in enumerate(line[:-2]) if i > 0]

            if line[-1] == 0:
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
        for i, equation in enumerate(self.U):
            if equation.fluid_loop not in index_set and isinstance(equation, MassFlowBalance):
                self.U.pop(i)
                index_set.add(equation.fluid_loop)

        self.C = components
        self.E = [(v, u) for v in self.V for u in self.U if v in u.variables]
        self.Ed = [(v, c) for c in self.C for v in self.V
                   if v.port_id[1] == c.number and ((isinstance(c, PressureBasedComponent)
                                                     and ((v.port_typ == 'in' and v.var_typ in ['p', 'h'])
                                                          or (v.port_typ == 'out' and v.var_typ == 'p')))
                                                    or (isinstance(c, MassFlowBasedComponent) and v.port_typ == 'in')
                                                    or (isinstance(c, BypassComponent) and (
                            (v.port_typ == 'in' and v.var_typ in ['p', 'h', 'm'])
                            or (v.port_typ == 'out' and v.var_typ == 'p'))))]
        self.Ed.extend([(c, v) for c in self.C for v in self.V
                        if v.port_id[1] == c.number and (isinstance(c, PressureBasedComponent)
                                                         and (v.var_typ == 'm' or (
                            v.var_typ == 'h' and v.port_typ == 'out'))
                                                         or isinstance(c,
                                                                       MassFlowBasedComponent) and v.port_typ == 'out'
                                                         or isinstance(c,
                                                                       BypassComponent) and v.port_typ == 'out' and v.var_typ in [
                                                             'h', 'm'])])


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

    while len(comp_exec) != len(tpg.C):
        if any(c.component_typ == 'Compressor' for c in comp_not_exec):
            indices = [i for i, c in enumerate(comp_not_exec)
                       if c.component_typ == 'Compressor']
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
        Vt.extend([e[0] for e in tpg.Ed if e[1] == comp_not_exec[id] and e[0] not in V])
        V.extend([e[0] for e in tpg.Ed if e[1] == comp_not_exec[id] and e[0] not in V])
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
                                                                                 for item in sublist]).intersection(set(V))).difference(set(V))))
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
        elif components[c].component_typ not in ['Compressor', 'Pump', 'Expansion Valve']:
            for j, value in enumerate(jpcm[:, c]):
                if value != 0 and jpcm[junc_no - 1, c] * value < 0 and jpcm[junc_no - 1, -2] == jpcm[j, -2]:
                    if components[c].source_component:
                        return c + 1
                    elif j != junc_no - 1:
                        return source_search(j, c)
    return []


def system_solver(x0: list, Vt: list, component_list: list, exec_list: list, residual_equations: list,
                  design_equations: list, scale_factors: list):
    """
    solves thermal hydraulic cycles using broydens-method and newton-homotopy pseudo-arc-length continuation
    :param x0:                  iteration start variables
    :param Vt:                  tearing variables
    :param component_list:      list of system components
    :param exec_list:           list of system components to be executed in the induced order
    :param residual_equations:  list of system residual equations
    :param design_equations:    list of design equations
    :param scale_factors:       list of tearing variables scale factors
    :return:                    solution of the system
    
    """

    def fun(x):

        """
        Calculate the residuals of the system equations
        :param x: iteration variables
        :return:  residuals
        """

        for i, variable in enumerate(Vt):
            if variable.var_typ == 'p':
                variable.set_value(x[i] / scale_factors[0])
            elif variable.var_typ == 'h':
                variable.set_value(x[i] / scale_factors[1])
            elif variable.var_typ == 'm':
                variable.set_value(x[i] / scale_factors[2])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 1.0
                item.solve()
                if item.status == 0:
                    for component in component_list:
                        component.reset()
                    return [[], 0]

        res = [residual_equation.residual() for residual_equation in residual_equations]
        try:
            res.extend([design_equation.solve() for design_equation in design_equations])
        except:
            [component.reset() for component in component_list]
            return [[], 0]

        [component.reset() for component in component_list]

        return [np.array(res), 1]

    def linear_fun(x):

        """
        Calculate the residuals of the linearized system equations
        :param x: iteration variables
        :return: [residuals, convergence_flag]
        """

        for i, variable in enumerate(Vt):
            if variable.var_typ == 'p':
                variable.set_value(x[i] / scale_factors[0])
            elif variable.var_typ == 'h':
                variable.set_value(x[i] / scale_factors[1])
            elif variable.var_typ == 'm':
                variable.set_value(x[i] / scale_factors[2])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = 0.0
                item.solve()
                if item.status == 0:
                    for component in component_list:
                        component.reset()
                    return [[], 0]

        res = [residual_equation.residual() for residual_equation in residual_equations]
        try:
            res.extend([design_equation.solve() for design_equation in design_equations])
        except:
            [component.reset() for component in component_list]
            return [[], 0]

        # Resets all System Components, Variables and Equations
        [component.reset() for component in component_list]

        return [np.array(res), 1]

    def probability_one_homotopy_fun(a, x):

        """
        Calculate the residuals of the Newton homotopy system equations
        :param res_0: residuals at initial values
        :param x: iteration variables
        :return: [residuals, convergence_flag]
        """

        for i, variable in enumerate(Vt):
            if variable.var_typ == 'm':
                variable.set_value(x[i] / scale_factors[2])
            elif variable.var_typ == 'p':
                variable.set_value(x[i] / scale_factors[0])
            elif variable.var_typ == 'h':
                variable.set_value(x[i] / scale_factors[1])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = x[-1]
                item.solve()
                if item.status == 0:
                    for component in component_list:
                        component.reset()
                    return [[], 0]

        res = [residual_equation.residual() for residual_equation in residual_equations]
        try:
            res.extend([design_equation.solve() for design_equation in design_equations])
        except:
            [component.reset() for component in component_list]
            return [[], 0]

        [component.reset() for component in component_list]

        return [np.array(np.array(res) - (1 - x[-1]) * (x[0:-1] - a)), 1]

    def linear_newton_homotopy_fun(res_0, x):

        """
        Calculate the residuals of the Newton homotopy system equations
        :param res_0: residuals at initial values
        :param x: iteration variables
        :return: [residuals, convergence_flag]
        """

        for i, variable in enumerate(Vt):
            if variable.var_typ == 'm':
                variable.set_value(x[i] / scale_factors[2])
            elif variable.var_typ == 'p':
                variable.set_value(x[i] / scale_factors[0])
            elif variable.var_typ == 'h':
                variable.set_value(x[i] / scale_factors[1])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = x[-1]
                item.solve()
                if item.status == 0:
                    for component in component_list:
                        component.reset()
                    return [[], 0]

        res = [residual_equation.residual() for residual_equation in residual_equations]
        try:
            res.extend([design_equation.solve() for design_equation in design_equations])
        except:
            [component.reset() for component in component_list]
            return [[], 0]

        [component.reset() for component in component_list]

        return [np.array(np.array(res) - (1 - x[-1]) * np.array(res_0)), 1]

    def linear_homotopy_fun(x):
        """
        Calculate the residuals of the homotopy system equations.

        Args:
            x: Iteration variables.

        Returns:
            A list containing residuals and convergence flag.

        """
        for i, variable in enumerate(Vt):
            if variable.var_typ == 'p':
                variable.set_value(x[i] / scale_factors[0])
            elif variable.var_typ == 'h':
                variable.set_value(x[i] / scale_factors[1])
            elif variable.var_typ == 'm':
                variable.set_value(x[i] / scale_factors[2])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.lamda = x[-1]
                item.solve()
                if item.status == 0:
                    for component in component_list:
                        component.reset()
                    return [[], 0]

        res = [residual_equation.residual() for residual_equation in residual_equations]
        try:
            res.extend([design_equation.solve() for design_equation in design_equations])
        except:
            [component.reset() for component in component_list]
            return [[], 0]

        [component.reset() for component in component_list]

        return [np.array(res), 1]

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
            return [[], 0]

        J = np.zeros((len(f_0), len(x0)))
        epsilon = 1e-6 * np.maximum(np.abs(x0), 0.01)

        for j in range(len(x0)):
            x = x0.copy()
            x[j] += epsilon[j]
            f_fw, convergence_flag = fun(x)

            if convergence_flag == 0:
                return [[], 0]

            J[:, j] = (f_fw - f_0) / epsilon[j]

        return [J, 1]

    def broyden_method(fun, J, max_iter, epsilon, print_convergence, x0):

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
        F, convergence_flag = fun(x[0])

        if convergence_flag == 0:
            print('Broyden-Solver stopped due to failure in model execution!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'model execution error': True}

        F_array = []

        while np.linalg.norm(F, 2) > epsilon:
            λ = 1.0
            try:
                dx = np.linalg.solve(J, -F)
            except:
                return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False, 'message': 'singular jacobian!'}
            while any(x[it] + λ * dx < 0):
                λ *= 1/2
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
                print(f'Iteration {it} / Max-Norm of Residuals: {np.max(np.abs(F[:-1]))}')

            if it > max_iter:
                break

        if np.linalg.norm(F, 2) > epsilon:
            print('Convergence criteria not reached within allowed number of iteration steps!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': False,
                    'message': 'convergence criteria not reached within allowed number of iteration steps'}
        else:
            print('Broyden Solver converged!')
            return {'x': x[-1], 'f': F, 'n_it': it + 1, 'converged': True, 'message': 'solver converged'}

    def system_linearization():

        print('linearizing system components...')

        for j, variable in enumerate(Vt):
            if variable.var_typ == 'p':
                variable.set_value(x0[j])
            elif variable.var_typ == 'h':
                variable.set_value(x0[j])
            elif variable.var_typ == 'm':
                variable.set_value(x0[j])

        for item in exec_list:
            if isinstance(item, BalanceEquation) or isinstance(item, EnthalpyFlowBalance):
                item.solve()
            elif isinstance(item, Component):
                item.solve()

        for c in component_list:
            c.x0 = []
            for port in c.ports:
                if isinstance(c, PressureBasedComponent):
                    if port.port_typ == 'in' and port.port_id[-1] == 0:
                        c.x0.append(port.p.value)
                        c.x0.append(port.h.value)
                    elif port.port_typ == 'out' and port.port_id[-1] == 0:
                        c.x0.append(port.p.value)
                elif isinstance(c, MassFlowBasedComponent):
                    if port.port_typ == 'in' and port.port_id[-1] == 0:
                        c.x0.append(port.p.value)
                        c.x0.append(port.h.value)
                        c.x0.append(port.m.value)
        for equa in design_equations:
            if isinstance(equa, SuperheatEquation):
                for port in equa.component.ports:
                    equa.x0.append(port.p.value)
                    equa.x0.append(port.h.value)
                    break

        [(c.jacobian(), setattr(c, 'linearized', True)) for c in component_list if not isinstance(c, BypassComponent)]
        # [(equa.jacobian(), setattr(equa, 'linearized', True)) for equa in design_equations if isinstance(equa, SuperheatEquation)]
        [c.reset() for c in component_list]

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
        epsilon = 1e-9  # Convergence criteria for Broyden solver
        max_fails = 20  # Maximum number of allowed solver fails
        N_max_outer = 50  # Maximum number of outer iterations
        N_max = 20  # Maximum number of allowed Broyden step iterations
        N_opt = 10  # Setpoint number of Broyden step iterations
        a = 0.5  # Acceleration factor
        sigma = 1.0  # Scaling factor
        lamda_g = 0.5  # Initial value of scale factor sigma and periodic rescaling
        dlamda_ds_max = 0.5  # Value at which rescaling is invoked

        if type == 'linear homotopy':
            system_linearization()
            J, convergence_flag = jacobian_forward(linear_fun, x0[0:-1])
            if convergence_flag == 0:
                return {'x': x0, 'converged': False, 'message': 'model execution error'}
            print('Executes Broyden Solver to solve linearized system...')
            sol = broyden_method(linear_fun, J, max_iter, epsilon, False, x0[0:-1])
            if not sol['converged']:
                return {'x': x0, 'converged': False, 'message': 'failed to solve linearized system'}
            x = [np.append(sol['x'], 0.0)]

        elif type == 'linear newton-homotopy':
            system_linearization()
            res_0, convergence_flag = fun(x0[0:-1])
            if convergence_flag == 0:
                return {'x': x0, 'converged': False, 'message': 'failed to solve newton-homotopy system at initial values'}
            x = [x0]

        elif type == 'probability-one homotopy':
            system_linearization()
            a_random = x0[0:-1].copy()
            x = [x0]

        it = 0
        while abs(x[it][-1] - 1) > tol:

            print('arc-length iteration-step:', it + 1, '/ lambda:', x[it][-1])

            print('computing Jacobian...')
            if type == 'linear homotopy':
                J, convergence_flag = jacobian_forward(linear_homotopy_fun, x[it])
            elif type == 'linear newton-homotopy':
                J, convergence_flag = jacobian_forward(partial(linear_newton_homotopy_fun, res_0), x[it])
            elif type == 'probability-one homotopy':
                J, convergence_flag = jacobian_forward(partial(probability_one_homotopy_fun, a_random), x[it])
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
                elif type == 'linear newton-homotopy':
                    partial_system = partial(augmented_system, partial(linear_newton_homotopy_fun, res_0), x[it], tau, ds[it])
                elif type == 'probability-one homotopy':
                    partial_system = partial(augmented_system, partial(probability_one_homotopy_fun, a_random), x[it], tau, ds[it])
                J_arc_length_eq, convergence_flag = jacobian_forward(partial(arc_length_equa, x[it], tau, ds[it]), x[it])
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
            [setattr(c, 'linearized', False) for c in component_list]

        print('computing Jacobian... ')
        J, convergence_flag = jacobian_forward(fun, x[-1][0:-1])
        if convergence_flag == 0:
            return {'x': x[-1], 'converged': False, 'message': 'model execution error'}

        print('Executes Broyden Solver for solving actual system...')
        sol = broyden_method(fun, J, max_iter, 1e-12, True, x[-1][0:-1])
        return sol

    # maximum number of iterations for broyden solver
    max_iter = 50

    # convergence criteria
    epsilon = 1e-12

    # scales initial values
    x0_scaled = np.zeros(len(x0))
    for i, var in enumerate(x0):
        if Vt[i].var_typ == 'p':
            x0_scaled[i] = x0[i] * scale_factors[0]
        elif Vt[i].var_typ == 'h':
            x0_scaled[i] = x0[i] * scale_factors[1]
        else:
            x0_scaled[i] = x0[i] * scale_factors[2]

    # Runs Broyden solver, returning the solution array if the solver converges; otherwise, starts the Arc-Length Continuation solver.
    print('Start Broyden Solver')
    print('computing Jacobian...')
    J, convergence_flag = jacobian_forward(fun, x0_scaled)
    if convergence_flag == 0:
        return {'x': x0, 'converged': False, 'message': 'failed to compute initial jacobian!'}
    sol = broyden_method(fun, J, max_iter, epsilon, True, x0_scaled)
    if sol['converged']:
        return sol
    else:
        x0_scaled = np.append(x0_scaled, 0.0)
        print('Solver not converged!')
        print('Start Pseudo-Arc-Length Continuation Solver')
        sol = pseudo_arc_length_continuation(x0_scaled, 'linear homotopy')
        if sol['converged']:
            print('Solver converged!')
        else:
            print('Solver not converged!')
        return sol


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
            Fig[i][1].set_ylim([min(min(p[i][:])) - 10 ** np.floor(np.log10(min(min(p[i][:])))), Pcrit * 1e-5 + 10 ** np.floor(np.log10(Pcrit * 1e-4))])
        else:
            Fig[i][1].set_xlim([min(h[i]) - 100, max(h[i]) + 100])
            Fig[i][1].set_ylim([min(p[i]) - 10 ** np.floor(np.log10(min(p[i]))), Pcrit * 1e-5 + 10 ** np.floor(np.log10(Pcrit * 1e-4))])

        Fig[i][1].set_xlabel('specific Enthalpy / kJ/kg', fontweight='bold')
        Fig[i][1].set_ylabel('Pressure / bar', fontweight='bold')
        Fig[i][1].set_yscale('log')
        Fig[i][1].grid(True)
        plt.title(fluid)
        plt.draw()
        plt.show()


def initialization(component: [Component], root: str, file: str):

    """
    Initialize a component by reading specifications and parameters from a CSV file.

    Args:
        component (Component): The component to be initialized.
        root (str): The root directory path.
        file (str): The name of the CSV file (without extension) containing the component information.

    Raises:
        RuntimeError: If the component's boundary type is not defined or unsupported.

    """

    specification_list = []

    with open(root + "/" + file + '.csv', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=";")
        specification_list.extend(list(csv_reader))

    # Creates component specification table from component .csv file
    spec_start_index = next((i for i, line in enumerate(specification_list) if line[0] == 'Specification:'), None)
    if spec_start_index is not None:
        for spec in specification_list[spec_start_index + 1:]:
            if spec[0] != '':
                component.specifications.append(spec)
            else:
                break

    # Creates component inputs from component .csv file
    input_start_index = next((i for i, line in enumerate(specification_list) if line[0] == "Inputs:"), None)
    if input_start_index is not None:
        for inp in specification_list[input_start_index + 1:]:
            if inp[0] != '':
                component.inputs.append([inp[0], float])
            else:
                break

    # Creates component outputs from component .csv file
    output_start_index = next((i for i, line in enumerate(specification_list) if line[0] == "Outputs:"), None)
    if output_start_index is not None:
        for outp in specification_list[output_start_index + 1:]:
            if outp[0] != '':
                component.outputs.append([outp[0], float])
            else:
                break

    # Defines Components Model Typ (Pressure Based or Mass Flow Based) from .csv file
    boundary_type = next((line[1] for line in specification_list if line[0] == 'Boundary Typ:'), None)
    if boundary_type == "Pressure Based":
        component.__class__ = PressureBasedComponent
    elif boundary_type == "Mass Flow Based":
        component.__class__ = MassFlowBasedComponent
    else:
        raise RuntimeError(f"Tried to initialize {component.name} Component but no allowed Boundary Type was defined")

    # Reads Component Parameter
    parameter_start_index = next((i for i, line in enumerate(specification_list) if line[0] == "Parameter:"), None)
    if parameter_start_index is not None:
        component.parameter.extend(specification_list[parameter_start_index + 1:])

        
# function for just creating widget to define boundary conditions and later on saving them in variables
def set_bc_values_onestage(pi_v_so=1.0, Ti_v_so=-10, mi_v_so=1.0, pi_c_so=1.0, Ti_c_so=20, mi_c_so=1.0):
    print(f'Evaporator: pi_v_so= {pi_v_so:5.1f} bar, Ti_v_so = {Ti_v_so:5.1f} °C, mi_v_so = {mi_v_so:5.1f} kg/s \n')
    print(f'Condenser:  pi_c_so= {pi_c_so:5.1f} bar, Ti_c_so = {Ti_c_so:5.1f} °C, mi_c_so = {mi_c_so:5.1f} kg/s \n')
    return pi_v_so, Ti_v_so, mi_v_so, pi_c_so, Ti_c_so, mi_c_so


# function for just creating widget to define boundary conditions and later on saving them in variables
def set_bc_values_cascade(pi_v_so=1.0, Ti_v_so=-10, mi_v_so=1.0,
                          pi_c_so=1.0, Ti_c_so=20, mi_c_so=1.0,
                          pi_gc_so=1.0, Ti_gc_so=20, mi_gc_so=1.0):
    print(f'Evaporator: pi_v_so= {pi_v_so:5.1f} bar, Ti_v_so = {Ti_v_so:5.1f} °C, mi_v_so = {mi_v_so:5.1f} kg/s \n')
    print(f'Condenser:  pi_c_so= {pi_c_so:5.1f} bar, Ti_c_so = {Ti_c_so:5.1f} °C, mi_c_so = {mi_c_so:5.1f} kg/s \n')
    print(f'Gas cooler:  pi_gc_so= {pi_gc_so:5.1f} bar, Ti_gc_so = {Ti_gc_so:5.1f} °C, mi_gc_so = {mi_gc_so:5.1f} kg/s \n')
    return pi_v_so, Ti_v_so, mi_v_so, pi_c_so, Ti_c_so, mi_c_so, pi_gc_so, Ti_gc_so, mi_gc_so


# function for just creating widget to define dc values and later on saving them in variables
def set_dc_values_onestage(SH_v=5.0):
    print(f'Evaporator Superheat:  SH_v= {SH_v:5.1f} K')
    print('\n--> Change the values with the sliders,',
          '\n    then execute all (code) cells below the widget again!',
          '\n    Only then the new values are used for the computations.')
    return SH_v


# function for just creating widget to define dc values and later on saving them in variables
def set_dc_values_cascade(SH_v=10.0, SH_chx1=5.0, SH_chx2=5.0):
    print(f'Evaporator Superheat:  SH_v= {SH_v:5.1f} K')
    print(f'Cascade HX 1 Superheat:  SH_v_chx1= {SH_chx1:5.1f} K')
    print(f'Cascade HX 2 Superheat:  SH_v_chx2= {SH_chx2:5.1f} K')
    print('\n--> Change the values with the sliders,',
          '\n    then execute all (code) cells below the widget again!',
          '\n    Only then the new values are used for the computations.')
    return SH_v, SH_chx1, SH_chx2
