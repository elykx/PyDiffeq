from methods.euler_method import EulerMethod
from methods.rk2_method import RK2Method
from methods.rk4_method import RK4Method


class ODE_Library:
    def __init__(self, system,method):
        self.system = system
        self.method = method

    def choose_method(self):
        if self.method == 'Euler':
            return EulerMethod(self.system)
        if self.method == 'RK2':
            return RK2Method(self.system)
        if self.method == 'Euler':
            return RK4Method(self.system)