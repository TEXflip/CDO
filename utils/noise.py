import numpy as np
from utils.controller import Controller

class Controllable(object):
    """
        Decorator for controllable objects
    """
    def __call__(self, cls):
        class ControllableClass(cls):
            controller = Controller()
            def setController(self, controller: Controller):
                self.controller = controller
                self.controlled_value = -1

            def __call__(self, *args):
                t = self.controller.compute(args[0])
                self.controlled_value = t
                return super().__call__(t, *args[1:])
        return ControllableClass

@Controllable()
class Noise:
    def __init__(self, size, ch=3):
        self.size = size[:2]
        self.ch = ch
    
    def __call__(self, *args) -> np.ndarray:
        return np.random.randint(0, 255, size=(*self.size, self.ch)).astype(np.float32) / 255.0

@Controllable()
class NoiseByFrequency:
    def __init__(self, size, ch=3, order=2):
        self.sigma = 10000
        self.size = size[:2]
        self.ch = ch
        h , w = size[:2]
        self.center = (h//2  , w//2)
        self.order = order

        exponent = np.log2(min(h, w))
        self.radius_fun = lambda freq: np.power(2, exponent * freq) # mapping from [0, 1] to [1, size] in log scale

        self.mesh = np.meshgrid(np.arange(h) - self.center[0], np.arange(w) - self.center[1])

    def __call__(self, *args) -> np.ndarray:
        """
        Generate noise from frequency domain
        :param freq: frequency, from 0 to 1
        """
        radius = self.radius_fun(1 - args[0]) # mapping from [0, 1] to [1, 200] in log scale
        mask = 1/(1+(np.sqrt(self.mesh[0]**2 + self.mesh[1]**2)/radius)**(2*self.order))

        # generate noise in frequency domain, real = module, imag = phase
        real = np.random.normal(0, self.sigma, size=(*self.size, self.ch))
        # imag = np.random.normal(0, self.sigma, size=(*self.size, self.ch))

        fshift = np.vectorize(complex)(real, real)

        # the center of the image is the DC frequency, so there is no phase and high module
        fshift[self.center[0], self.center[1]] = np.vectorize(complex)(np.zeros(3) + (self.size[0] * self.size[1] * 127.5), np.zeros(3))
        fshift = fshift * mask.reshape((*self.size, 1))

        f_ishift = np.fft.ifftshift(fshift, axes=(0, 1))
        img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
        img_back = np.abs(img_back)

        # normalize
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())

        return img_back

@Controllable()
class NoiseByAmplitude:
    def __init__(self, size, ch=3) -> None:
        self.size = size[:2]
        self.ch = ch

    def __call__(self, *args) -> np.ndarray:
        """
        Generate noise by amplitude
        :param amplitude: [0, 1]
        """
        noise_amplitude = args[0] * 127
        return np.random.randint(0 + noise_amplitude, 255 - noise_amplitude, size=(*self.size, self.ch)).astype(np.float32) / 255.0

@Controllable()
class NoiseBlending:
    def __init__(self, exp=4, max_value=0.6) -> None:
        self.exp = exp
        self.max_value = max_value

    def alpha_fun(self, t):
        return np.power(t, self.exp) * self.max_value
    
    def __call__(self, *args) -> np.ndarray:
        alpha = self.alpha_fun(args[0])
        return (1 - alpha) * args[1] + alpha * args[2]

NOISE_GENERATORS = {
    'freq': NoiseByFrequency,
    'amp': NoiseByAmplitude,
    'normal': Noise,
    '': Noise
}

NOISE_POSTPROCESSORS = {
    'alpha': NoiseBlending
}
