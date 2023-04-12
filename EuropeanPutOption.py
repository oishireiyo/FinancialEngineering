import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class EuropeanPutOption():
    def __init__(self, K, r, sigma, T):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
    def _Get_dminus(self, t, S):
        K, r, sigma, T = self.K, self.r, self.sigma, self.T
        if t <= T:
            dminus = (np.log(S/K) + (r-0.5*np.power(sigma, 2))*(T-t)) / (sigma * np.sqrt(T-t))
        else:
            print('Error: t should be smaller than T')
            exit(1)
        return dminus
    
    def _Get_dplus(self, t, S):
        K, r, sigma, T = self.K, self.r, self.sigma, self.T
        if t <= T:
            dplus = (np.log(S/K) + (r+0.5*np.power(sigma, 2))*(T-t)) / (sigma * np.sqrt(T-t))
        else:
            print('Error; t should be smaller than T')
            exit(1)
        return dplus
    
    def Get_ds(self, t, S):
        dminus = self._Get_dminus(t, S)
        dplus = self._Get_dplus(t, S)
        return dminus, dplus

    def Get_Premium(self, t, S):
        K, r, sigma, T = self.K, self.r, self.sigma, self.T
        dminus, dplus = self.Get_ds(t, S)
        premium = S * (norm.cdf(dplus)-1.0) - K*np.exp(-r*(T-t))*(norm.cdf(dminus)-1.0)
        return premium
    
# parameters
S0 = 100 # initial stock price
r = 0.05 # interest rate / year
sigma = 0.4 # volatility
Ks = [k for k in range(50, 150, 1)] # strike price

# Case 1. T = 0.0
T = 0.0001
premiums1 = []
for K in Ks:
    Model1 = EuropeanPutOption(K, r, sigma, T)
    premium1 = Model1.Get_Premium(0.0, S0)
    premiums1.append(premium1)

# Case 2. T = 0.5
T = 0.5
premiums2 = []
for K in Ks:
    Model2 = EuropeanPutOption(K, r, sigma, T)
    premium2 = Model2.Get_Premium(0.0, S0)
    premiums2.append(premium2)

# Case 3. T = 1.0
T = 1.0
premiums3 = []
for K in Ks:
    Model3 = EuropeanPutOption(K, r, sigma, T)
    premium3 = Model3.Get_Premium(0.0, S0)
    premiums3.append(premium3)

X = np.array(Ks)
Y1 = np.array(premiums1)
Y2 = np.array(premiums2)
Y3 = np.array(premiums3)

p1, = plt.plot(X, Y1)
p2, = plt.plot(X, Y2)
p3, = plt.plot(X, Y3)

plt.title('Call Option ($\sigma$ = %.2f, r = %.2f, $S_{0}$ = %d)' % (sigma, r, S0))
plt.xlabel('Strike price')
plt.ylabel('Premium')
plt.grid(True)
plt.legend([p1, p2, p3], ['T = 0.0', 'T = 0.5', 'T = 1.0'])
