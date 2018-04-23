import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

img_path = 'image path'
img = imread(img_path)

plt.imshow(img)
plt.show()
