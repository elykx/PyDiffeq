class ODE_System:
    def __init__(self, y0):
        """
        Инициализация системы ОДУ.

        Parameters:
        - y0: начальные значения системы оду
        """
        self.y0 = y0

    def func(self, y, t):
        """
        Абстрактный метод для вычисления системы оду

        Parameters:
        - t_eval: массив времени
        - y0: начальные значения системы оду
        """
        pass
