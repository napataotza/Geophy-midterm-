density_map = np.load('data_out/density_map.npy')
# NOTE: the observed points are in grid coordinates (pixel), which is not physical unit.
x_obs_points = np.array([14.896, 29.189, 44.858, 53.374, 64.719, 74.686, 90.511, 109.068,
                         125.047, 136.206, 144.901, 155.477, 162.132, 177.815, 195.034])
y_obs_points = np.array([74.894, 66.427, 55.655, 36.682, 25.261, 10.725, 7.987, 7.623,
                         21.880, 30.317, 38.539, 52.832, 64.594, 69.843, 73.201])

#-----------------------------------------------------------------------------------------#

# same code

#-----------------------------------------------------------------------------------------#

total_obs_points = len(x_obs_points)

for obs_idx in range(total_obs_points):
    x_obs = x_obs_points[obs_idx]  # x-coordinate of the observation point
    y_obs = y_obs_points[obs_idx]  # y-coordinate of the observation point
    # same code

#-----------------------------------------------------------------------------------------#

# NOTE: optional for comparion
# np.save('data_out/gravity_profile.npy', gravity_profile)

#-----------------------------------------------------------------------------------------#

def plot_gravitational_profile(x_obs_points, gravity_profile, output_dir='figure_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    x_indices = np.linspace(0, len(x_obs_points) - 1, len(x_obs_points)).astype(int)
    ax.plot(x_indices, gravity_profile, 'b-', marker='o', markerfacecolor='green',
            markeredgewidth=2, markeredgecolor='black', markersize=10)
    ax.set_title('Gravitational Profile A')
    ax.set_xlabel('Station Number')
    ax.set_ylabel(r'$\mathbf{g_{z}}$ (m/s²)')
    ax.set_xticks(x_indices)  # Set x-ticks at the indices
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle='--')
    # plt.savefig(f'{output_dir}/gravitational_profile_a.svg', format='svg', bbox_inches='tight',
    #             transparent=True, pad_inches=0.0)
    plt.show()

#-----------------------------------------------------------------------------------------#