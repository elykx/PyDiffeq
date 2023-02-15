from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver


# Kutta-Merson
class KuttaMersonMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            k1 = dt * self.system.func(y, t)
            k2 = dt * self.system.func(y + k1 / 3, t + dt / 3)
            k3 = dt * self.system.func(y + k1 / 6 + k2 / 6, t + dt / 2)
            k4 = dt * self.system.func(y - k2 / 3 + k3, t + 2 * dt / 3)
            k5 = dt * self.system.func(y + k1 / 8 + 3 * k2 / 8 + 3 * k3 / 8 + k4 / 8, t + dt)
            y += (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6 + k5 / 6)
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution)