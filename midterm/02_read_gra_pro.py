import numpy as np
import matplotlib.pyplot as plt
# Load the gravity profile data from a file

profile = np.load('midterm\data_out\density_map.npy')
plt.imshow(profile)
plt.show()