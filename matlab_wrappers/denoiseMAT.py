import denoising
import numpy as np 

def denoiseit(img, x, y, filePath, DnCNNnoise = 0.1):
	"""Details:
	img: a VECTOR from matlab. (MATLAB still can't pass in 2D data to python...)
	x: the first dimension of the shape of img (so if img were 2d, img.shape[0]
	y: the second dimension of shape of img
	filePath: the path of the file
	DnCNNnoise: the noiselevel we normalize to


	"""
	img = np.array(img)
	img = img.reshape((int(x),int(y)))
	print(type(img))	
	denoiser = denoising.denoiser("DnCNN", DnCNNfile = filePath, DnCNNnoise = DnCNNnoise)
	newimg = denoiser.denoise(img)
	return newimg

def denoise2(img):
	return img + 1

def denoise3(img):
	print('hi')

def denoise4(array):
	print(len(array))
	return 
