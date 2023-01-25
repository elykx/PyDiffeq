from abc import ABC, abstractmethod


class ODE_Solver(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def solve(self, t_eval, y0):
        pass
