from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class EulerMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            y += dt * self.system.func(y, t)
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution)
