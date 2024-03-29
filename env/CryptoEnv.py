import gym
from gym import spaces
import pandas as pd
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt

class CustomEnv(gym.Env):
    def __init__(self, timesteps, pairName, ordersize = 0.3, debug = False):
        super(CustomEnv, self).__init__()
        self.timesteps = timesteps
        self.ordersize = ordersize
        self.pairName = pairName

        self.debug = debug

        # buy, sell, doNothing
        self.action_space = spaces.Discrete(3) 
        
        # open, high, low, close, volume, canbuy, cansell
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.timesteps, 7), dtype=np.float32)

        self.buffer = deque(maxlen=self.timesteps)
        self.orders = deque()

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f'data/{self.pairName}.csv')

        self.data = pd.read_csv(filename)
        self.N = len(self.data)

        self.budget = 0
        self.nobuys = 0
        self.nosells = 0
        self.currtime = 0
        self.selltimes = 0

        self.budgethist = []
        self.imgind = 0

        self.done = False

        self.ind = 0

    def calcTotalBudget(self, currprice):
        total = 0
        for order in self.orders:
            total += order['size'] * currprice
        return total + self.budget

    def buffer2obs(self):
        arr = np.array(list(self.buffer))
        arr = arr.T

        for col in range(arr.shape[0]):
            minval = np.min(arr[col])
            maxval = np.max(arr[col])

            if maxval != minval: arr[col] = (arr[col]-minval) / (maxval - minval)

        return arr.T

    def step(self, action):
        curriter = np.array(self.data.iloc[self.ind][0:5])
        currprice = curriter[3]
        totalBudget = self.calcTotalBudget(currprice)
        self.budgethist.append(totalBudget)

        orderprice = totalBudget * self.ordersize
        taxprice = orderprice * 0.001

        taxorderprice = orderprice - taxprice

        reward = 0
        if action == 0:
            # buy
            if self.budget > orderprice:
                # can buy
                order = {
                    'size': taxorderprice / currprice,
                    'price': taxorderprice,
                    'time': self.currtime
                }
                if self.debug: print(f"BUY {order['size']} {order['price']} {order['time']}")

                self.orders.append(order)
                self.budget -= orderprice
                self.nobuys += 1
                reward += 0.1
            else: reward -= 0.1
        elif action == 1:
            # sell
            if len(self.orders) > 0:
                # can sell
                order = self.orders[0]
                self.orders.popleft()

                size = order['size']
                price = order['price']
                buytime = order['time']

                sellprice = size * currprice * (1-0.001)

                reward += (sellprice - price) if (sellprice - price) >= 0 else (sellprice - price) * 0.1

                if self.debug: print(f"SELL {sellprice} iniPrice: {price} profit: {sellprice-price} time: {self.currtime-buytime}")
                self.selltimes += (self.currtime - buytime)

                self.budget += sellprice
                self.nosells += 1
            else: reward -= 0.1
        else: reward -= 0.01

        canbuysell = np.array([1 if self.budget > orderprice else 0, 1 if len(self.orders) > 0 else 0])

        tobuffer = np.concatenate((curriter, canbuysell), 0)
        self.buffer.append(tobuffer)

        observation = self.buffer2obs()
        self.ind += 1
        self.currtime += 1
        self.done = self.ind == self.N or totalBudget <= 60
        return observation, reward, self.done, {}

    def reset(self):
        print(f"Budget: {self.calcTotalBudget(self.data.iloc[self.ind-1][3])} buys: {self.nobuys} sells: {self.nosells} avgselltimes: {self.selltimes/(max(self.nosells, 1))}")
        plt.plot(self.budgethist)
        plt.savefig(f'{self.pairName}_{self.imgind}.png')
        plt.clf()
        self.imgind += 1

        self.ind = 0
        self.buffer.clear()
        self.orders.clear()
        self.budgethist.clear()

        self.done = False

        self.nobuys = 0
        self.nosells = 0
        
        self.currtime = 0
        self.selltimes = 0

        self.budget = 100.0

        observation, reward, done, info = self.step(2)
        for _ in range(self.timesteps):
            observation, reward, done, info = self.step(2)

        return observation