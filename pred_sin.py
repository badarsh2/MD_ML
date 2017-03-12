# -*- coding: utf-8 -*-

import numpy as np
from k_ridge_regression import k_ridge_regression 
import matplotlib.pyplot as plt

def gause_noise(data):
	data += np.random.randn(len(data))
	return data



if __name__ == "__main__":
	X = np.arange(0,2,0.1)
	t = np.zeros( len(X))
	
	for i in range(len(X)):
		t[i] = np.sin(np.pi * X[i]) * 10
	t = gause_noise(t)

	
	#plot
	labels = ["lambda=0.0001", "lambda=0.001","lambda=0.01","lambda=0.1","lambda=1"]
	plts   = ["b--.", "g--.", "c--.", "m--.", "y--."]

	l = k_ridge_regression()
	l.set_data(X,t)

	lam = 0.0001
	for i in range(5):
		l.set_lam(lam)
		lam *= 10
		l.learn()
		
		result = np.zeros(len(X))
		for j in range(len(X)):
			result[j] = l.pred(X[j])
		plt.plot(X,result,plts[i], label=labels[i])
	
	plt.plot(X, t, "r+", label="learn data")
	plt.legend()
	plt.savefig("plt1.png")
	plt.show()
	print result

