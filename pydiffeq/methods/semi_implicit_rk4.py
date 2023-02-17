from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver
from pydiffeq.utils.dichotomy_method import dichotomy_method


class SemiImplicitRK4Method(ODE_Solver):
    def solve(self, t_eval, y0):
        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        for t in t_eval[1:]:
            # Явная схема для k1,k2,k3
            k1 = self.system.func(y, t)
            k2 = self.system.func(y + dt / 2 * k1, t + dt / 2)
            k3 = self.system.func(y + dt / 2 * k2, t + dt / 2)

            # Неявная схема для k4

            a, b = y + dt*k3, y + dt/2*(k2+k3)
            k4 = dichotomy_method(a, b, dt/6 * (k1 + 2*k2 + 2*k3 + self.system.func(b, t+dt)))

            # Полунеяная схема для y_n+1
            y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            y0 = copy(y)
            solution.append(y0)
        return np.array(solution), t_eval
