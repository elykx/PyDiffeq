import numpy as np

from ode_solver import ODE_Solver


class MiddlePointMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = y0.copy()
        solution = [y]
        for t in t_eval[1:]:
            k1 = self.system.func(y, t)
            k2 = self.system.func(y + dt / 2 * k1, t + dt / 2)
            y += dt * k2
            solution.append(y)
        return np.array(solution)
