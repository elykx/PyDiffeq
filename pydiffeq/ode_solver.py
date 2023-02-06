from abc import ABC, abstractmethod


class ODE_Solver(ABC):
    def __init__(self, system, decimal_place=6):
        self.system = system
        self.decimal_place = decimal_place

    @abstractmethod
    def solve(self, t_eval, y0):
        pass
