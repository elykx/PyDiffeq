from copy import copy

import numpy as np

from pydiffeq.methods import ExplicitEulerMethod
from pydiffeq.ode_solver import ODE_Solver


class SemiImplicitEuler(ODE_Solver):
    def solve(self, t_eval, y0):
        y_euler, t = ExplicitEulerMethod(self.system).solve(t_eval, y0)
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        y_next = copy(y0)
        solution = [y0]
        for i, t in enumerate(t_eval[1:]):
            # Явная схема
            k = dt * self.system.func(y, t)
            #Полунеявня схема
            y = y + dt/2 * (self.system.func(y, t) + self.system.func(y_euler[i+1], t))
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval
