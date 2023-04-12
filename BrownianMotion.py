import random
import math
import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion():
    def __init__(self):
        self.mu = 0.0
        self.sigma = 0.1
        self.S0 = 100

        self.T = 1
        self.n = 100
        self.h = float(self.T) / float(self.n)
        
    def GetRandomValue(self):
        random_value = random.gauss(0, 1)
        return random_value
    
    def GenerateGBM(self):
        Ss = []
        Ss.append(self.S0)
        S = self.S0
        for i in range(self.n):
            random_value = self.GetRandomValue()
            S = S * math.exp((self.mu - 0.5 * math.pow(self.sigma, 2)) * self.h
                             + self.sigma * math.sqrt(self.h) * random_value)
            Ss.append(S)

        print(Ss)
        return Ss

    def Plotter(self):
        X = np.array([float(i) / float(self.n) for i in range(0, self.n+1)])

        Y1 = np.array(self.GenerateGBM())
        Y2 = np.array(self.GenerateGBM())
        Y3 = np.array(self.GenerateGBM())

        p1, = plt.plot(X, Y1)
        p2, = plt.plot(X, Y2)
        p3, = plt.plot(X, Y3)
        plt.title('Brownian Motion ($\mu$ = %.2f, $\sigma$ = %.2f, $S_{0}$ = %.2f)' % (self.mu, self.sigma, self.S0))
        plt.xlabel('Time')
        plt.ylabel('$S_{t}$')
        plt.grid(True)
        plt.legend([p1, p2, p3], ['Sample 1', 'Sample 2', 'Sample 3'])

Obj = BrownianMotion()
Obj.Plotter()
