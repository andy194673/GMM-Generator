#!/usr/bin/python
import numpy as np
from lib import generateGMMs
from matplotlib.pylab import *

if __name__ == '__main__':
	'''
	n_gauss: number of gaussian in a GMM
	n_dim: number of dimension of a gaussian
	n_samples: number of data sampled from a GMM

	<Output format>
	Each GMM class gets its own mean, co-var, weigths and samples
	generateGMMs returns a list of GMM
	
	'''
	n_gauss, n_dim, n_samples_train, n_samples_test = 32, 2, 10000, 1000
	print('Generating data, takes around 25 secs')

	# Training
	np.random.seed(112)
	GMMs_train = generateGMMs(n_gauss, n_dim, n_samples_train)
	print('Done generating training set')

	# get mean, co-variance, weight, samples from first GMM in training set
	GMM = GMMs_train[0]
	mean, var, weight, samples, pdf = GMM.mean, GMM.var, GMM.weight, GMM.samples, GMM.pdf

	# Testing
	np.random.seed(113)
	GMMs_test = generateGMMs(n_gauss, n_dim, n_samples_test)
	print('Done generating testing set')

	GMM = GMMs_test[0]
	mean, var, weight, samples, pdf = GMM.mean, GMM.var, GMM.weight, GMM.samples, GMM.pdf


	# TO DO
	# Visualization
#	X = data_test[0]
#	plot(X[:,0], X[:,1], '.')
#	for i in range(len(mc)):
#		x1, x2 = gmm.gauss_ellipse_2d(centroids[i], ccov[i])
#		plot(x1, x2, 'k', linewidth=2)
#		xlabel('$x_1$'); ylabel('$x_2$')
