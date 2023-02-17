from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver
from pydiffeq.utils.dichotomy_method import dichotomy_method


class SemiImplicitEuler(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        y_next = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            # Явная схема
            k = dt * self.system.func(y, t)
            #Находим y_n+1 неявно с помощью метода дихотомии
            y_next = dichotomy_method(y, y+k, k)
            #Полунеявня схема
            y = y + dt * self.system.func(y_next, t)
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval
