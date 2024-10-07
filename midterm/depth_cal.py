import numpy as np

#---------------------------------------------------------------------------#

density_map = np.load('midterm\data_out\profile_a.npy')
n_y, n_x = density_map.shape  # Get the shape of the density map (height, width)
x_obs_points = np.array([14.896, 29.189, 44.858, 53.374, 64.719, 74.686, 90.511, 109.068,
                         125.047, 136.206, 144.901, 155.477, 162.132, 177.815, 195.034])
y_obs = np.array([74.894, 66.427, 55.655, 36.682, 25.261, 10.725, 7.987, 7.623,
                    21.880, 30.317, 38.539, 52.832, 64.594, 69.843, 73.201])
x_pixels = np.arange(n_x)  # x coordinates of the pixels
y_pixels = np.arange(n_y)  # y coordinates of the pixels
X, Y = np.meshgrid(x_pixels, y_pixels)  # Grid of pixel coordinates
# Initialize an array to store the gravitational effects for all observation points
gravity_profile = np.zeros_like(x_obs_points, dtype=float)
depth_station =[]

#---------------------------------------------------------------------------#

for obs_idx, x_obs in enumerate(x_obs_points):
    for depth_find in y_pixels:
        if density_map[int(depth_find),int(x_obs)] == 4700 :
            depth_station.append(depth_find - y_obs[obs_idx])
            break    
        elif depth_find == max(y_pixels) :
            depth_station.append('none')
    print("station",obs_idx ,"\tore depth_station =",depth_station[obs_idx], "pixels")
    