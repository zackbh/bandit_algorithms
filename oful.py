import numpy as np;
import numpy.random as ra;
import scipy.linalg as sla;
import numpy.linalg as la;
import pdb;
#import emd

N = 25  # number of observations
d = 5   # number of covariates
n_a = 10  # number of actions
S = 1.0
R = 1;

    #- generate theta_star
v = ra.normal(0,1,d);
v /= la.norm(v)
theta_star = v * S;

    #- generate X
X = ra.normal(0,1,(N, n_a, d));
#norms = la.norm(X);
#X /= norms.reshape(-1,1)
#X = X;

#- save expected rewards
expt_reward = np.dot(X, theta_star);
best_arm = np.argmax(expt_reward);

def calc_sqrt_beta_det2(d,t,R,ridge,delta,S_hat,logdetV):
  return R * np.sqrt( logdetV - d * np.log(ridge) + np.log(1/(delta**2)) ) + np.sqrt(ridge) * S_hat

ridge = .1;
delta = 0.1;
S_hat = 1; # not ideal.. but a makeshift
t = 1;

XTy = np.zeros(d)
invVt = np.eye(d) / ridge
#X_invVt_norm_sq = np.sum(X * X, axis=1) / ridge;
logdetV = d * np.log(ridge);
sqrt_beta = calc_sqrt_beta_det2(d,t,R,ridge,delta,S_hat,logdetV);      
theta_hat = np.zeros(d);
Vt = ridge * np.eye(d); 
my_c = .1 # Agressiveness (inv)


for t in range(N):

#For t = 0 choose randomly

	x = X[t]
#	X_invVt_norm_sq = np.sum(x * x, axis=1) / ridge;
	X_invVt_norm_sq = np.sum(np.dot(x, invVt) * x, 1)
	obj_func = np.dot(x, theta_hat)  + my_c * sqrt_beta * np.sqrt(X_invVt_norm_sq) 
	pulled_idx = np.argmax(obj_func);  # Pull the arm with the highest estimated value
	#pulled_idx = np.random.randint(0, n_a)  # Pull a random arm

	xt = x[pulled_idx, :]
	reward = np.dot(xt, theta_star) + R * ra.normal(0,5)

	XTy += reward * xt
	Vt += np.outer(xt, xt);

	tempval1 = np.dot(invVt, xt)    # d by 1, O(d^2)
	tempval2 = np.dot(tempval1, xt)      # scalar, O(d)
	logdetV += np.log(1 + tempval2);

	if (t % 20 == 0):
		invVt = la.inv(Vt);
#		X_invVt_norm_sq -= (np.dot(x, tempval1) ** 2) / (1 + tempval2) # efficient update, O(Nd)
	else:
		invVt -= np.outer(tempval1, tempval1) / (1 + tempval2) 
#       X_invVt_norm_sq -= (np.dot(x, tempval1) ** 2) / (1 + tempval2) # efficient update, O(Nd)

	theta_hat = np.dot(invVt, XTy)
	#print(theta_hat)
	my_t = t + 1
	sqrt_beta = calc_sqrt_beta_det2(d, my_t, R, ridge, delta, S_hat, logdetV);  



print(theta_star)
print(theta_hat)
