# -*- coding: utf-8 -*-
import math
import numpy as np

class k_ridge_regression:
	def __init__(self):
		self.lam = 0.
		self.sigma = 1.

	def set_data(self, X, t):
		self.X = X
		self.t = np.array(t)

	def set_lam(self, lam):
		self.lam = lam

	def set_sigma(self, sigma):
		self.sigma = sigma

	def gaussian(self, x1, x2, sigma):
		return math.exp( -np.linalg.norm( np.array(x1) - np.array(x2) )  / (sigma**2) )

	def get_w(self):
		return self.a

	def learn(self, ktype = 1):
		dim = len( self.X )
		self.dim  = dim
		K = np.zeros( (dim, dim) )
	
		# K = [[k(x1,x1),k(x1,x2),・・・・,k(x1,xn)],・・・・,[k(xn,x1),k(xn,x2),・・・・・k(xn,xn)]]
		for i in range( dim ):
			for j in range( dim ):
				K[i,j] = self.gaussian(self.X[i], self.X[j], self.sigma)
		
		
		self.a = np.zeros( (dim, 1) )
		I = np.matrix(np.identity(dim))
		self.a = np.dot( np.linalg.inv( K + self.lam * I ), self.t )  #(K + lambda*I)^-1 * t

	def pred(self, x):
		k = np.zeros( (1, self.dim) )
		for i in range( self.dim ):
			k[0,i] = self.gaussian(self.X[i], x, self.sigma)
		return np.dot(self.a, k.T)


