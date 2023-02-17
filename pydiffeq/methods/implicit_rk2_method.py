from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver
from pydiffeq.utils.dichotomy_method import dichotomy_method


class ImplicitRK2Method(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            # Явная схема для k1 и k2
            k1 = self.system.func(y, t)
            k2 = self.system.func(y + dt/2 * k1, t + dt/2)
            k = dt * k2
            # Находим y_n+1 неявно с помощью метода дихотомии
            y = dichotomy_method(y, y + k, k)
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval