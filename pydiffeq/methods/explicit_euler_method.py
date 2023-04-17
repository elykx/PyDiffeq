from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class ExplicitEulerMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            #Явная схема
            k = dt * self.system.func(y, t)
            y += k
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval
