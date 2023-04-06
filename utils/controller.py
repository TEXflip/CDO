import sys 
sys.path.append('.')
import numpy as np

class Controller:
    """
    value controller, from 0 to 1
    """
    def __init__(self):
        pass

    def compute(self, t):
        return t


class LinearStepController(Controller):
    """
    linear step controller, from 0 to 1, in steps
    1|      __
     |    __
     |  __
    0|________ t
    """
    def __init__(self, step):
        super().__init__()
        self.step = step

    def compute(self, t):
        return np.floor(t * self.step) / (self.step - 1)

class FunctionStepController(Controller):
    """
    function step controller, from 0 to 1, 
    in steps following a continuous function
    """
    def __init__(self, step, function=lambda x: x*x):
        super().__init__()
        self.step = step
        self.f = function

    def compute(self, t):
        return self.f(np.floor(t * self.step)) / self.f(self.step - 1)

class AdaptiveController(Controller):
    """
    adaptive controller, from 0 to 1
    """
    def __init__(self):
        super().__init__()

    def compute(self, t):
        return 0

CONTROLLERS = {c.__qualname__: c for c in Controller.__subclasses__()}