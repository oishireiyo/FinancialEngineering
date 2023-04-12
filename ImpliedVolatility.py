import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

class ImpliedVolatility():
    def __init__(self, S0, r, T):
        self.S0 = S0
        self.r = r
        self.T = T
        self.iteration = 50 # The number of iteration for Newton method

    def _Get_dminus(self, K, volatility):
        dminus = (np.log(S0/K) + (self.r - 0.5*np.power(volatility, 2)*self.T)) / (volatility * np.sqrt(self.T))
        return dminus
    
    def _Get_dplus(self, K, volatility):
        dplus = (np.log(S0/K) + (self.r + 0.5*np.power(volatility, 2)*self.T)) / (volatility * np.sqrt(self.T))
        return dplus
        
    def Get_ds(self, K, volatility):
        dminus = self._Get_dminus(K, volatility)
        dplus = self._Get_dplus(K, volatility)
        return dminus, dplus

    def Get_CallPremium(self, K, volatility):
        dminus, dplus = self.Get_ds(K, volatility)
        premium = self.S0*norm.cdf(dplus) - K*np.exp(-self.r*self.T)*norm.cdf(dminus)
        return premium
    
    def Get_PutPremium(self, K, volatility):
        dminus, dplus = self.Get_ds(K, volatility)
        premium = -self.S0*norm.cdf(-dplus) + K*np.exp(-self.r*self.T)*norm.cdf(-dminus)
        return premium

    def Get_Vega(self, K, volatility):
        # Vega is defined as the partial differentiation of PREMINUM by VOLATILITY
        _, dplus = self.Get_ds(K, volatility)
        vega = self.S0*np.sqrt(T)*np.exp(-0.5*np.power(dplus, 2)) / np.sqrt(2.0*np.pi)
        return vega
    
    def Get_ImpliedVolatilityWithNewtonMethod(self, Observed, K, Type='call'):
        implied_vol = 1.0
        for i in range(self.iteration):
            if Type == 'call':
                implied_vol = implied_vol - ((self.Get_CallPremium(K, implied_vol) - Observed) / self.Get_Vega(K, implied_vol))
            elif Type == 'put':
                implied_vol = implied_vol - ((self.Get_PutPremium(K, implied_vol) - Observed) / self.Get_Vega(K, implied_vol))
            else:
                print('You need to pass \'call\' or \'put\' but %s was given.' % (Type))
                exit(1)
        return implied_vol

if __name__ == '__main__':
    # Data set
    df = pd.read_csv('/kaggle/input/forimpliedvolatility/ForImpliedVolatility.csv')
    df_call = df[df['call_value_market'] != 0]
    df_put = df[df['put_value_market'] != 0]
    
    # Parameters
    S0 = 29053.97 # initial stock price
    r = 0.00 # interest rate / year
    T = 44.0 / 365.0 # in year
    
    Object = ImpliedVolatility(S0, r, T)

    implied_volatility_with_call = np.zeros(len(df_call))
    for i_index, index in enumerate(df_call.index):
        K = df_call.at[index, 'K'] # Strike price
        Observed = df_call.at[index, 'call_value_market'] # Observed price
        implied_volatility_with_call[i_index] = Object.Get_ImpliedVolatilityWithNewtonMethod(Observed, K, 'call')

    for i_index, index in enumerate(df_put.index):
        K = df_put.at[index, 'K'] # Strike price
        Observed = df_put.at[index, 'put_value_market'] # Observed price
        implied_volatility_with_put[i_index] = Object.Get_ImpliedVolatilityWithNewtonMethod(Observed, K, 'put')

    p_call, = plt.plot(df_call['K'], implied_volatility_with_call)
    p_put, = plt.plot(df_put['K'], implied_volatility_with_put)
    
    plt.xlabel('Strike price (K)')
    plt.ylabel('Implied volatility ($\sigma$)')
    plt.grid('True')
    plt.legend([p_call, p_put], ['Call option', 'Put option'])
