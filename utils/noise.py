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
        self.value = -1
    
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
        self.value = -1

        exponent = np.log2(min(h, w))
        self.radius_fun = lambda freq: np.power(2, exponent * freq) # mapping from [0, 1] to [1, size] in log scale

        self.mesh = np.meshgrid(np.arange(h) - self.center[0], np.arange(w) - self.center[1])

    def __call__(self, *args) -> np.ndarray:
        """
        Generate noise from frequency domain
        :param freq: frequency, from 0 to 1
        """
        radius = self.radius_fun(1 - args[0]) # mapping from [0, 1] to [1, 200] in log scale
        self.value = radius # for visualization
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
        self.value = -1

    def __call__(self, *args) -> np.ndarray:
        """
        Generate noise by amplitude
        :param amplitude: [0, 1]
        """
        noise_amplitude = args[0] * 127
        self.value = noise_amplitude
        return np.random.randint(0 + noise_amplitude, 255 - noise_amplitude, size=(*self.size, self.ch)).astype(np.float32) / 255.0

@Controllable()
class NoiseBlending:
    def __init__(self, exp=4, max_value=0.6) -> None:
        self.exp = exp
        self.max_value = max_value
        self.value = -1

    def alpha_fun(self, t):
        return np.power(t, self.exp) * self.max_value
    
    def __call__(self, *args) -> np.ndarray:
        alpha = self.alpha_fun(args[0])
        self.value = alpha
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

if "__main__" == __name__:
    from PIL import Image
    import yaml
    import os

    os.mkdir('noise_generation')

    for noise_gen in ['freq', 'amp', 'normal']:
        noise = NOISE_GENERATORS[noise_gen]((256, 256))
        
        img = noise(0)
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_gen}_0.png')
        img = noise(0.5)
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_gen}_0.5.png')
        img = noise(1)
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_gen}_1.png')
    
    # load image from dataset
    conf = yaml.safe_load(open('../config.yaml', 'r'))
    path = os.path.join(conf['DATA_ROOT'], conf['MVTEC2D_DIR'])
    img = np.array(Image.open(os.path.join(path, 'cable', 'train', 'good', '00000.png')))
    patch_mask = np.zeros(img.shape[:2], dtype=np.float32)
    patch_mask[50:100, 80:150] = 1.0
    noise_patch = Noise((256, 256))(0)[patch_mask > 0]

    for noise_post in ['alpha']:
        noise = NOISE_POSTPROCESSORS[noise_post]()

        img = noise(0, np.zeros((256, 256, 3)), np.ones((256, 256, 3)))
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_post}_0.png')
        img = noise(0.5, np.zeros((256, 256, 3)), np.ones((256, 256, 3)))
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_post}_0.5.png')
        img = noise(1, np.zeros((256, 256, 3)), np.ones((256, 256, 3)))
        Image.fromarray((img * 255).astype(np.uint8)).save(f'noise_generation/{noise_post}_1.png')
        