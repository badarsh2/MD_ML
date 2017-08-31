import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

path = 'Al_structure_files/DatasetA/bulk_1250K.xyz'

xyz = open(path , 'r')

f1, f2, f3 = [], [], []

while True:
	n = xyz.readline()
	if(n != ''):
		x1, x2, x3  = [], [], []
		n = int(n)
		info = xyz.readline()

		for i in range(int(n)):
			line = xyz.readline()
			row = line.split()
			x1.append(float(row[1]))
			x2.append(float(row[2]))
			x3.append(float(row[3]))
			f1.append(float(row[5]))
			f2.append(float(row[6]))
			f3.append(float(row[7]))

		x_m1 = np.array([x1] * n)
		r1 = np.absolute(x_m1 - x_m1.T)

		x_m2 = np.array([x2] * n)
		r2 = np.absolute(x_m2 - x_m2.T)

		x_m3 = np.array([x3] * n)
		r3 = np.absolute(x_m3 - x_m3.T)

		r_euc = np.sqrt(np.multiply(r1, r1) + np.multiply(r2, r2) + np.multiply(r3, r3))

		cos1 = np.divide(r1, r_euc)
		cos1[np.where(np.isnan(cos1))] = 0.

		cos2 = np.divide(r2, r_euc)
		cos2[np.where(np.isnan(cos2))] = 0.

		cos3 = np.divide(r3, r_euc)
		cos3[np.where(np.isnan(cos3))] = 0.

		r_critical = np.amax(r_euc)
		damping = 0.5 * (np.cos(r_euc * math.pi / r_critical) + 1)

		v1 = np.sum(np.multiply(cos1, np.multiply(np.exp(np.multiply(r_euc/n, r_euc/n) * -1), damping)), axis = 0)
		v2 = np.sum(np.multiply(cos2, np.multiply(np.exp(np.multiply(r_euc/n, r_euc/n) * -1), damping)), axis = 0)
		v3 = np.sum(np.multiply(cos3, np.multiply(np.exp(np.multiply(r_euc/n, r_euc/n) * -1), damping)), axis = 0)

		try:
			X_train = np.hstack((X_train, np.vstack((np.vstack((v1, v2)), v3))))
		except NameError:
			X_train = np.vstack((np.vstack((v1, v2)), v3))

		print X_train.shape

	else:
		break

xyz.close()

lambdas = np.logspace(-5, 5, 50)
sigmas = np.logspace(-5, 5, 50)

parameter_grid = {
	'alpha': lambdas,
	'gamma': sigmas
}

gscv = GridSearchCV(KernelRidge(kernel='rbf'), param_grid=parameter_grid, cv=5, n_jobs=-1)
# gscv = KernelRidge(kernel='rbf', alpha=1., gamma=0.01)
gscv.fit(X_train[:, :6000].T, np.array(f1)[:6000])
print gscv.predict(X_train[:, 6000:].T), np.array(f1)[6000:]

print gscv.best_estimator_
print gscv.cv_results_
