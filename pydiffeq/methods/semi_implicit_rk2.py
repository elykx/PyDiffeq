from copy import copy

import numpy as np

from pydiffeq.methods import ExplicitEulerMethod
from pydiffeq.ode_solver import ODE_Solver


class SemiImplicitRK2Method(ODE_Solver):
    def solve(self, t_eval, y0):
        y_euler, t = ExplicitEulerMethod(self.system).solve(t_eval, y0)
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for i, t in enumerate(t_eval[1:]):
            # Явная схема для k1
            k1 = self.system.func(y, t)
            # Неявная схема для k2
            k2 = self.system.func(y_euler[i + 1] + dt / 2 * k1, t + dt / 2)
            # Полунеяная схема для y_n+1
            k = dt * k2
            y += k
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval
