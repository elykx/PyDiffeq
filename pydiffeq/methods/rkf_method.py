from copy import copy

import numpy as np

from pydiffeq.ode_solver import ODE_Solver


class RKFMethod(ODE_Solver):
    def solve(self, t_eval, y0, rtol=1e-6, atol=1e-6):
        a = np.array([[0, 0, 0, 0, 0, 0],
                      [1/4, 0, 0, 0, 0, 0],
                      [3/32, 9/32, 0, 0, 0, 0],
                      [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                      [439/216, -8, 3680/513, -845/4104, 0, 0],
                      [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]])
        b = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        b_star = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])

        dt = t_eval[1] - t_eval[0]
        y = copy(y0)
        solution = [y0]
        t = t_eval[0]
        t_list = [t_eval[0]]
        while t < t_eval[-1]:
            if t + dt > t_eval[-1]:
                dt = t_eval[-1] - t
            k = np.zeros((6, len(y)))
            for i in range(6):
                y_i = y + np.dot(a[i, :i], k[:i, :]) * dt
                t_i = t + c[i] * dt
                k[i] = self.system.func(y_i, t_i)
            err = np.abs(np.dot(b-b_star, k)*dt) / (atol + rtol * np.abs(y))
            err_max = np.max(err)
            if err_max > 1:
                dt *= 0.8 * err_max**(-0.25)
            else:
                y += np.dot(b, k) * dt
                t += dt
                dt *= 0.8 * err_max**(-0.2)
                y0 = copy(y)
                solution.append(y0)
                t_list.append(t)
        return np.array(solution), t_list
