import numpy as np
import utils.controller as controllers

class Controllable(object):
    """
        Decorator for controllable objects
    """
    def __call__(self, cls):
        class ControllableClass(cls):
            controller = controllers.Controller()
            def setController(self, controller: controllers.Controller):
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


class NoiseSelector:
    """
        static class for selecting noise generator and postprocessor
        during training per epoch
    """
    noise_generators = [v for k,v in NOISE_GENERATORS.items() if k]
    noise_postprocessors = [v for k,v in NOISE_POSTPROCESSORS.items() if k]
    max_epoch = 50
    epoch = 0
    loss_trajectory = []
    selection_trajectory = []
    genotype_trajectory = []
    init = 1

    @staticmethod
    def reset(size, ch=3):
        NoiseSelector.epoch = 0
        NoiseSelector.loss_trajectory = []
        NoiseSelector.selection_trajectory = []

        if NoiseSelector.init:
            for i in range(len(NoiseSelector.noise_generators)):
                NoiseSelector.noise_generators[i] = NoiseSelector.noise_generators[i](size, ch)
                NoiseSelector.noise_generators[i].setController(controllers.RandomController())

            for i in range(len(NoiseSelector.noise_postprocessors)):
                NoiseSelector.noise_postprocessors[i] = NoiseSelector.noise_postprocessors[i]()
                NoiseSelector.noise_postprocessors[i].setController(controllers.RandomController())
            NoiseSelector.init = 0
        
        # first gene is the noise generator
        # second gene is the postprocessor (0 = off), otherwise i-1 of the postprocessor
        gene = [0, 0]

        NoiseSelector.genotype_trajectory = [gene]

    @staticmethod
    def __call__(*args):
        pass

    @staticmethod
    def step(loss):
        NoiseSelector.epoch += 1
        NoiseSelector.loss_trajectory.append(loss)

        # selection phase
        last = NoiseSelector.loss_trajectory[-5:]
        if len(last) < 5:
            gene = [np.random.randint(len(NoiseSelector.noise_generators)), 
                    np.random.randint(len(NoiseSelector.noise_postprocessors) + 1)]
            NoiseSelector.genotype_trajectory.append(gene)
        else:
            best = np.argmin(last)
            if np.random.rand() < 0.5:
                gene = NoiseSelector.genotype_trajectory[-5:][best]
            else:
                gene = [np.random.randint(len(NoiseSelector.noise_generators)), 
                        np.random.randint(len(NoiseSelector.noise_postprocessors) + 1)]
                NoiseSelector.genotype_trajectory.append(gene)

        if NoiseSelector.epoch < 5:
            return
        
        # log polynomial fitting of the loss trajectory
        traj = NoiseSelector.loss_trajectory
        x_size = len(traj)
        x = np.arange(x_size)
        # find the point c of -log(a, x) + c
        # where the gradient 1 for the normalized log trajectory
        axis_ratio = np.abs(np.max(traj) - np.min(traj))/x_size
        c_i = np.abs(np.gradient(-traj, 1) - axis_ratio).argmin()
        c = traj[c_i]

        # find the base of the log function (a = 2^(log2(x)/(log(a,x)+c)); where -traj ~ log(a, x) + c)
        log_base_traj = np.exp2(np.log2(x)/(-traj + c))
        # since -traj is not a perfect log function, we take the median of the second half of the trajectory as base
        log_base = np.median(log_base_traj[x_size//2:])
        
        # now we can fit with the polynomial of the power of the log
        y = np.power(log_base, -traj) * 1e4 # linearize the trajectory inverting the log function
        a, b, c, d = np.polyfit(x, y, 3)
        x = x[-1] # we need only the value of this epoch
        fit_vec = a*x**3 + b*x**2 + c*x + d
        # fit_grad = 2*a*x + b
        diff = (y[-1] - fit_vec)/(abs(y[-1]-fit_vec) + fit_vec) # normalized difference between the trajectory and the fit
        NoiseSelector.selection_trajectory.append(diff)

    @staticmethod
    def augment(img, patch_mask):
        gene = NoiseSelector.genotype_trajectory[-1]
        noise = NoiseSelector.noise_generators[gene[0]](0)
        augmented_image = img
        if gene[1]:
            blended_noise = NoiseSelector.noise_postprocessors[gene[1]-1](1, noise, img)
            augmented_image[patch_mask > 0] = blended_noise
        else:
            augmented_image[patch_mask > 0] = noise[patch_mask > 0]
        return augmented_image

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
        