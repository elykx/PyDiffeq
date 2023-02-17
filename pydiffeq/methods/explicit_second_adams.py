from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class ExplicitAdams2Method(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for i, t in enumerate(t_eval[1:]):
            if i == 0:
                # Use Euler method to get second point
                y1 = y + dt * self.system.func(y, t)
                y0 = copy(y1)
                solution.append(y0)
                y = y1
            else:
                # Use Adams-Bashforth 2-step method
                y += dt/2 * (3*self.system.func(y, t) - self.system.func(y0, t-dt))
                y0 = copy(y)
                solution.append(y0)
        return np.array(solution), t_eval