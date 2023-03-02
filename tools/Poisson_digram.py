# -*- coding : utf-8 -*-
# @Time :  2022/8/21 19:02
# @Author : hxc
# @File : Poisson_digram.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Poisson分布
    x = np.random.poisson(lam=10, size=10000)  # lam为λ size为k
    pillar = 100
    a = plt.hist(x, bins=pillar, range=[0, pillar], color='g', alpha=0.5)
    plt.plot(a[1][0:pillar], a[0], 'r')
    plt.grid()
    plt.show()