import numpy as np

from pydiffeq.methods import EulerMethod, RK2Method, RK4Method, ImplicitEulerMethod, TrapezoidMethod, MiddlePointMethod


class ODE_Library:
    def __init__(self, system, method, decimal_place=6):
        self.system = system
        self.method = method
        self.decimal_place = decimal_place

    def solve(self, t, y0):
        solution = None

        if self.method == 'EULER':
            return EulerMethod(self.system).solve(t, y0)
        if self.method == 'RK2':
            return RK2Method(self.system).solve(t, y0)
        if self.method == 'RK4':
            return RK4Method(self.system).solve(t, y0)
        if self.method == 'IMPLICIT_EULER':
            return ImplicitEulerMethod(self.system).solve(t, y0)
        if self.method == 'TRAPEZOID':
            return TrapezoidMethod(self.system).solve(t, y0)
        if self.method == 'MIDDLE':
            return MiddlePointMethod(self.system).solve(t, y0)

        if solution is not None:
            return np.round(solution, self.decimal_place)
        return solution
