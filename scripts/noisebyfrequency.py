import numpy as np
from PIL import Image
import cv2

class NoiseBlending:
    def __init__(self, exp=1, max_value=1) -> None:
        self.exp = exp
        self.max_value = max_value
        self.value = -1

    def alpha_fun(self, t):
        return np.power(t, self.exp) * self.max_value
    
    def __call__(self, *args) -> np.ndarray:
        alpha = self.alpha_fun(args[0])
        self.value = alpha
        return (1 - alpha) * args[1] + alpha * args[2]

def noise_fft(radius, size=(200, 200), order=2):
	sigma = 10500
	r, c = size
	cr, cc = r//2 , c//2
	mesh = np.meshgrid(np.arange(r) - cr, np.arange(c) - cc)
	mask = 1/(1+(np.sqrt(mesh[0]**2 + mesh[1]**2)/radius)**(2*order))

	real = np.random.normal(0, sigma, size=(size[0], size[1], 3))
	# imag = np.random.normal(0, sigma, size=(200, 200, 3))
	fshift = np.vectorize(complex)(real, real)
	fshift[cr, cc] = np.vectorize(complex)(np.ones(3) * (size[0] * size[1] * 127.5), np.zeros(3))
	fshift = fshift * mask.reshape((size[0], size[1], 1))

	f_ishift = np.fft.ifftshift(fshift, axes=(0, 1))
	img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
	img_back = np.abs(img_back)

	# normalize
	img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
	img_back = (img_back  * 255).astype(np.uint8)
	return img_back.copy(), mask.copy()


def noise_gauss(radius):
	noise_image = np.random.uniform(0, 255, size=(200, 200, 3)).astype(np.uint8)
	noise_image = cv2.GaussianBlur(noise_image, (radius, radius), 0)
	noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
	noise_image = (noise_image  * 255).astype(np.uint8)
	return noise_image.copy()


def create_video(size=(200, 200)):
	# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	# video_writer = cv2.VideoWriter("noise.avi", fourcc, 4, (200, 200))
	# video_writer_mask = cv2.VideoWriter("mask.avi", fourcc, 4, (400, 400))

	for i in np.linspace(0, 1, 100):
		radius = np.power(2, 7.643856189774724 * i) # mapping from [0, 1] to [1, 200] in log scale

		img, mask = noise_fft(radius, size)
		# draw text of radius value
		img = cv2.putText(img, str(radius), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		# save frame to dir
		cv2.imwrite("frames/%f.jpg" % i, img.copy())
		# video_writer.write(img)

		# bring mask to 3 channels
		mask = (np.stack((mask, mask, mask), axis=2) * 255).astype(np.uint8)
		# video_writer_mask.write(mask)

	# video_writer.release()
	# cv2.destroyAllWindows()

# img, mask = noise_fft(1024, (1024, 1024))
img_noise = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
img_bg = np.zeros((100, 100, 3), dtype=np.uint8)
img0 = NoiseBlending()(0, img_noise, img_bg).astype(np.uint8)
img1 = NoiseBlending()(0.5, img_noise, img_bg).astype(np.uint8)
img2 = NoiseBlending()(1, img_noise, img_bg).astype(np.uint8)
Image.fromarray(img0).save("im0.png")
Image.fromarray(img1).save("im1.png")
Image.fromarray(img2).save("im2.png")
# img, mask = noise_fft(10, (200, 200))
# Image.fromarray(img).save("noise2.png")
# create_video()
# import matplotlib.pyplot as plt
# noise_image = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)

# f = np.fft.fft2(noise_image, axes=(0, 1))
# print(f[0, 0])
# fshift = np.fft.fftshift(f, axes=(0, 1))


# plt.hist(gauss, 100, range=(-1<<exp,1<<exp))
# plt.show()
# plt.hist(imag.flatten(), 100, range=(-1<<exp,1<<exp))
# plt.show()

# gaussian distribution fitting
# from scipy.optimize import curve_fit
# def gaussian(x, a, x0, sigma):
# 	return a*np.exp(-(x-x0)**2/(2*sigma**2))

# hist, bins = np.histogram(real.flatten(), 100, range=(-1<<exp,1<<exp))
# bins = bins[:-1]
# popt, pcov = curve_fit(gaussian, bins, hist, p0=[100, 0, 10000])
# print(popt)
# plt.plot(bins, hist, 'b+:', label='data')
# plt.plot(bins, gaussian(bins, *popt), 'r-', label='fit')
# plt.show()


# real: 202.46050906   0 10440
# Image.fromarray(mask*255).show()
# Image.fromarray(s).show()
# Image.fromarray(fshift.real).show()
# Image.fromarray(img_back).show()