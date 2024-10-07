"""import sys
sys.path.append('./Lib')
import utilities as U"""
#-----------------------------------------------------------------------------------------#
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#-----------------------------------------------------------------------------------------#

def plot_density_model(density_map, output_dir='midterm\data_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(density_map, cmap='gray', origin='upper')
    ax.set_title('Density Model')
    ax.axis('off')
    ax.set_xlabel('pixel-x')
    ax.set_ylabel('pixel-y')
    cbar = plt.colorbar(im, ax=ax, fraction=0.056, pad=0.04, orientation='horizontal')
    cbar.set_label('Density')
    plt.savefig(f'{output_dir}/density_model.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

def plot_gravitational_profile(x_obs_points, gravity_profile, output_dir='midterm\data_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    x_indices = np.linspace(0, len(x_obs_points) - 1, len(x_obs_points)).astype(int)
    ax.plot(x_obs_points, (gravity_profile*100000), 'b-', marker='o', markerfacecolor='green',
            markeredgewidth=2, markeredgecolor='black', markersize=10)
    ax.set_title('Gravitational Profile')
    ax.set_xlabel('x (pixels)')
    ax.set_xticks(x_obs_points)
    ax.set_xticklabels(x_indices)
    ax.set_ylabel(r'$\mathbf{g_{z}}$ (mGal)')
    y_formatter = ScalarFormatter(useMathText=True)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.grid(True, linestyle='--')
    plt.savefig(f'{output_dir}/gravitational_profile.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()
    
#-----------------------------------------------------------------------------------------#

density_map = np.load('midterm\data_out\density_map.npy')
G = 6.67430e-11  # m^3 kg^-1 s^-2
n_y, n_x = density_map.shape  # Get the shape of the density map (height, width)
x_obs_points = np.array([14.896, 29.189, 44.858, 53.374, 64.719, 74.686, 90.511, 109.068,
                         125.047, 136.206, 144.901, 155.477, 162.132, 177.815, 195.034])
y_obs = np.array([74.894, 66.427, 55.655, 36.682, 25.261, 10.725, 7.987, 7.623,
                    21.880, 30.317, 38.539, 52.832, 64.594, 69.843, 73.201])
y_elipsoid = 55.655
#x_obs_points = np.arange(0, n_x, 15)  # Observation points along the x-axis, skipping every 5 pixels
#y_obs = 154  # Observation is along the x-axis (cross-section)
# Create arrays for pixel coordinates assuming (0, 0) is the top-left corner
x_pixels = np.arange(n_x)  # x coordinates of the pixels
y_pixels = np.arange(n_y)  # y coordinates of the pixels
X, Y = np.meshgrid(x_pixels, y_pixels)  # Grid of pixel coordinates
# Flatten the density map to create the density vector m_rho
m_rho = density_map.flatten()
# Initialize an array to store the gravitational effects for all observation points
gravity_profile = np.zeros_like(x_obs_points, dtype=float)

#-----------------------------------------------------------------------------------------#

# Compute the gravitational effect at each observation point along the x-axis
for obs_idx, x_obs in enumerate(x_obs_points):
    
    # Initialize the gravity kernel matrix A for this observation point
    A = np.zeros((1, n_x * n_y))
    # Populate the gravity kernel matrix A for the current observation point
    # NOTE the code processes rows first and then columns, it is row-major. 
    for i in range(n_y):
        
        for j in range(n_x):
            # Position of the pixel (x_j, y_i) and its distance to the observation point
            x_pixel = X[i, j]  # x-coordinate of the pixel
            y_pixel = Y[i, j]  # y-coordinate of the pixel
            # Distance between the pixel and the observation point (on the x-axis, y_obs = 0)
            r = np.sqrt((x_obs - x_pixel)**2 + (y_obs[obs_idx] - y_pixel)**2)
            # Avoid division by zero at the observation point
            if r == 0:
                A[0, i * n_x + j] = 0  
            else:
                # Gravitational prism kernel for the pixel (x_j, y_i)
                A[0, i * n_x + j] = -G * (-y_pixel * np.log(x_pixel + r) - x_pixel * np.log(y_pixel + r)) / r**2
    print('obs_idx =', obs_idx)
    # Compute the gravitational effect g_z = A * m_rho for the current observation point
    g_z = np.dot(A, m_rho)
    # Store the result in the gravity profile array
    gravity_profile[obs_idx] = g_z[0]

#-----------------------------------------------------------------------------------------#

# U.plot_model_kernel(density_map, x_obs_points, gravity_profile, output_dir='figure_out')

#plot_density_model(density_map, output_dir='midterm\data_out')
plot_gravitational_profile(x_obs_points, gravity_profile, output_dir='midterm\data_out')

#-----------------------------------------------------------------------------------------#