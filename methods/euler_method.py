import numpy as np

from ode_solver import ODE_Solver


class EulerMethod(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = y0.copy()
        solution = [y]
        for t in t_eval[1:]:
            y += dt * self.system.func(y, t)
            solution.append(y)
        return np.array(solution)