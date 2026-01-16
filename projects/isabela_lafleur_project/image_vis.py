import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from math import sqrt

from utils import load_data
   

X_train, X_val, y_train, y_val, X_test, _ = load_data()

sample = X_train[0]
print(sample.min(), sample.max())

img = sample.reshape(5, 41)

plt.figure()
plt.imshow(img, cmap="gray")

plt.colorbar()
plt.title("Sample image (5 x 41)")
plt.axis("off")
plt.show()
