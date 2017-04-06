#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

x = np.linspace(-10, 10, 10000)
y = x*x*x
plt.plot(x,y)
plt.show()
