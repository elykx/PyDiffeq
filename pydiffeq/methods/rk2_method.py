import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class RK2Method(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = y0.copy()
        solution = [y]
        for t in t_eval[1:]:
            k1 = dt * self.system.func(y, t)
            k2 = dt * self.system.func(y + k1/2, t + dt/2)
            y += k2
            solution.append(y)
        return np.array(solution)