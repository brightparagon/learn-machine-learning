import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)

# graph
plt.plot(x, y)
plt.show()
