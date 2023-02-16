import numpy as np

from pydiffeq.methods import EulerMethod, RK2Method, RK4Method, ImplicitEulerMethod, TrapezoidMethod, MiddlePointMethod,\
    KuttaMersonMethod, RKFMethod


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

        if self.method == 'EULER':
            solution, t_eval = EulerMethod(self.system).solve(t, y0)
        if self.method == 'RK2':
            solution, t_eval = RK2Method(self.system).solve(t, y0)
        if self.method == 'RK4':
            solution, t_eval = RK4Method(self.system).solve(t, y0)
        if self.method == 'IMPLICIT_EULER':
            solution, t_eval = ImplicitEulerMethod(self.system).solve(t, y0)
        if self.method == 'TRAPEZOID':
            solution, t_eval = TrapezoidMethod(self.system).solve(t, y0)
        if self.method == 'MIDDLE':
            solution, t_eval = MiddlePointMethod(self.system).solve(t, y0)
        if self.method == 'KM':
            solution, t_eval = KuttaMersonMethod(self.system).solve(t, y0)
        if self.method == 'RKF':
            solution, t_eval = RKFMethod(self.system).solve(t, y0)

        if solution is not None:
            return np.round(solution, self.decimal_place), np.round(t_eval, self.decimal_place)
        return solution, t_eval
