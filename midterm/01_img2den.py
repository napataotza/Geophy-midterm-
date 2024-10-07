import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

img = Image.open('midterm\model_a.png')
target_color = np.array([255, 204, 0]) # select your color
density_ore = 4700
output_dir = 'midterm\data_out'  # Directory to save the .npy file

#-----------------------------------------------------------------------------------------#

img_array = np.array(img)
img_array = img_array[:, :, :3]  # Ensure RGB channels
print(img_array.shape)
if img_array.shape[0] != 154 or img_array.shape[1] != 212:
    raise ValueError("The image does not have the required dimensions of 154 rows by 212 columns")
mask = np.all(img_array == target_color, axis=-1)
density_map = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
density_map[mask] = density_ore
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.save(os.path.join(output_dir, 'density_map.npy'), density_map)
plt.imshow(density_map, cmap='gray')
plt.title('Density Map')
plt.show()

#-----------------------------------------------------------------------------------------#