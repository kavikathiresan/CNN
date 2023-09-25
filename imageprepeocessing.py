import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = np.arange(1,256)# taking the image value  0 to 255
print(a)
values = a.reshape(1,-1)# converting 1 row and 255 column
print(values.shape)
values=np.repeat(values,100,axis=0)# converting 100 row and 255 column
print(values.shape)
plt.imshow(values,cmap='gray')
plt.show()