from abc import ABC, abstractmethod


class ODE_Solver(ABC):
    def __init__(self, system, decimal_place=6):
        """
        Абстрактный класс для численного решения ОДУ

        Parameters:
        - system: система ОДУ
        - decimal_place: параметр округления
        """
        self.system = system
        self.decimal_place = decimal_place

    @abstractmethod
    def solve(self, t_eval, y0):
        """
         Абстрактный метод для численного решения системы ОДУ.

         Parameters:
        - t_eval: массив времени
        - y0: начальные значения системы оду
        """
        pass
