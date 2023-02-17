import numpy as np

from pydiffeq.methods import ExplicitEulerMethod, ExplicitRK2Method, ExplicitRK4Method, ImplicitEulerMethod, \
    TrapezoidMethod, \
    MiddlePointMethod, KuttaMersonMethod, RKFMethod, ExplicitAdams2Method, SemiImplicitEuler, ImplicitRK4Method, \
    ImplicitRK2Method, SemiImplicitRK2Method, SemiImplicitRK4Method


class ODE_Library:
    def __init__(self, system, method, decimal_place=6):
        """
        Инициализация библиотеки для выбора метода решения системы оду.

        Parameters:
        - system: система оду
        - method: метод решения
        - decimal_place: параметр округления
        """
        self.system = system
        self.method = method
        self.decimal_place = decimal_place

    def solve(self, t, y0):
        solution, t_eval = None, None

        if self.method == 'EXPLICIT_EULER':
            solution, t_eval = ExplicitEulerMethod(self.system).solve(t, y0)
        if self.method == 'IMPLICIT_EULER':
            solution, t_eval = ImplicitEulerMethod(self.system).solve(t, y0)
        if self.method == 'SEMI_IMPLICIT_EULER':
            solution, t_eval = SemiImplicitEuler(self.system).solve(t, y0)
        if self.method == 'EXPLICIT_RK2':
            solution, t_eval = ExplicitRK2Method(self.system).solve(t, y0)
        if self.method == 'IMPLICIT_RK2':
            solution, t_eval = ImplicitRK2Method(self.system).solve(t, y0)
        if self.method == 'SEMI_IMPLICIT_RK2':
            solution, t_eval = SemiImplicitRK2Method(self.system).solve(t, y0)
        if self.method == 'EXPLICIT_RK4':
            solution, t_eval = ExplicitRK4Method(self.system).solve(t, y0)
        if self.method == 'IMPLICIT_RK4':
            solution, t_eval = ImplicitRK4Method(self.system).solve(t, y0)
        if self.method == 'SEMI_IMPLICIT_RK4':
            solution, t_eval = SemiImplicitRK4Method(self.system).solve(t, y0)
        if self.method == 'TRAPEZOID':
            solution, t_eval = TrapezoidMethod(self.system).solve(t, y0)
        if self.method == 'MIDDLE':
            solution, t_eval = MiddlePointMethod(self.system).solve(t, y0)
        if self.method == 'KM':
            solution, t_eval = KuttaMersonMethod(self.system).solve(t, y0)
        if self.method == 'RKF':
            solution, t_eval = RKFMethod(self.system).solve(t, y0)
        if self.method == 'EXPLICIT_ADAMS':
            solution, t_eval = ExplicitAdams2Method(self.system).solve(t, y0)


        if solution is not None:
            return np.round(solution, self.decimal_place), np.round(t_eval, self.decimal_place)
        return solution, t_eval
