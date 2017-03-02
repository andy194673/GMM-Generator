import numpy as np
import pypr.clustering.gmm as gmm
from math import exp

class GMM():
	def __init__(self, mean, var, weight, n_samples, n_gauss):
		self.mean = mean # mean vector, data shape: (n_gauss, dim)
		self.var = var # co-variance matrix, data shape: (n_gauss, dim, dim)
		self.weight = weight # weights between gaussians, data shape (n_gauss, )
		self.n_samples = n_samples # number of samples, a scalar
		self.n_gauss = n_gauss # number of guassians, a scalar
		self.generate_samples()
		self.generate_pdf()

	def generate_samples(self):
		self.samples = gmm.sample_gaussian_mixture(self.mean, self.var, self.weight, samples=self.n_samples)
		'''
		return sample points, 2d np array (n_samples, dim)
		'''

	def generate_pdf(self):
		self.pdf = gmm.gmm_pdf(self.samples, self.mean, self.var, self.weight)
		'''
		return pdf of each sample point, 1d np array
		'''
	

# A matrix contains a mean point in each row 
def gen_mean_dist(n_gauss, n_dim):
	return np.random.uniform(low=-10.0, high=10.0, size=(n_gauss, n_dim))

# A list of diagonal matrix for gaussians
def gen_var_dist(n_gauss, n_dim):
	diag_values = np.random.uniform(low=0, high=1.0, size=(n_gauss, n_dim)) # diagonal value
	return np.array([np.diag(v) for v in diag_values])

def softmax(v):
	sum_exp = sum([exp(ele) for ele in v])
	res = [exp(ele)/sum_exp for ele in v]
	return np.array(res)

# A vector for gaussain weights
def gen_weight_dist(n_gauss):
	while(True):
		res = softmax( np.random.uniform(low=0, high=1.0, size=(n_gauss,)) )
		if sum(res) == 1.0: # with a very low prob, the sum of softmax is very close to 1 due to precision, re-gen
			return res

	
def generateGMMs(n_gauss, n_dim, n_samples):
	n_gmm = 1000

	'''
	Experiment 1, 1 mean, 1 var, 1 weight, 1 GMM here
	'''
	GMMs = []
	fix_mean, fix_var, fix_weight = gen_mean_dist(n_gauss, n_dim), gen_var_dist(n_gauss, n_dim), gen_weight_dist(n_gauss)
	
	GMMs.append( GMM(fix_mean, fix_var, fix_weight, n_samples, n_gauss) )

	
	'''
	Experiment 2, 1000 mean, 1 var, 1 weight, 1000 GMM here
	'''
	for i in range(n_gmm):
		mean = gen_mean_dist(n_gauss, n_dim)
		GMMs.append( GMM(mean, fix_var, fix_weight, n_samples, n_gauss) )

	'''
	Experiment 3, 1 mean, 1000 var, 1 weight, 1000 GMM here
	'''
	for i in range(n_gmm):
		var = gen_var_dist(n_gauss, n_dim)
		GMMs.append( GMM(fix_mean, var, fix_weight, n_samples, n_gauss) )

	'''
	Experiment 3, 1 mean, 1 var, 1000 weight, 1000 GMM here
	'''
	for i in range(n_gmm):
		weight = gen_weight_dist(n_gauss)
		GMMs.append( GMM(fix_mean, fix_var, weight, n_samples, n_gauss) )

	'''
	Experiment 3, 10 mean, 10 var, 10 weight, 1000 GMM here
	'''
	for i in range(10):
		mean = gen_mean_dist(n_gauss, n_dim)
		for j in range(10):
			var = gen_var_dist(n_gauss, n_dim)
			for k in range(10):
				weight = gen_weight_dist(n_gauss)
				GMMs.append( GMM(mean, var, weight, n_samples, n_gauss) )

	return GMMs
