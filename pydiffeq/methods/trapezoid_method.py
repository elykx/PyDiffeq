import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class TrapezoidMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = y0.copy()
        solution = [y]
        for t in t_eval[1:]:
            k1 = self.system.func(y, t)
            k2 = self.system.func(y + dt * k1, t + dt)
            y += dt / 2 * (k1 + k2)
            solution.append(y)
        return np.array(solution)
