import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from movement_primitives.promp import ProMP


def generate_3d_trajectory_distribution(n_demos, n_steps, base_amplitude=1.0, noise_level=0.1):
    """Generate 3D trajectory distribution with noise around a base trajectory"""
    # Generate time points (uniformly distributed between 0 and 1)
    T = np.linspace(0, 1, n_steps)
    
    # Initialize trajectory array
    Y = np.zeros((n_demos, n_steps, 3))
    
    # Define base trajectory - a smooth 3D trajectory
    for demo in range(n_demos):
        # X component: sine curve
        Y[demo, :, 0] = base_amplitude * np.sin(2 * np.pi * T) + \
                        np.random.normal(0, noise_level, n_steps)
        
        # Y component: cosine curve
        Y[demo, :, 1] = base_amplitude * np.cos(2 * np.pi * T) + \
                        np.random.normal(0, noise_level, n_steps)
        
        # Z component: linear rise + small sine variation
        Y[demo, :, 2] = base_amplitude * (T - 0.5) + 0.3 * base_amplitude * np.sin(4 * np.pi * T) + \
                        np.random.normal(0, noise_level, n_steps)
    
    return T, Y


# Generate 3D trajectory data
n_demos = 100
n_steps = 101
T, Y = generate_3d_trajectory_distribution(n_demos, n_steps, base_amplitude=1.0, noise_level=0.08)
y_conditional_cov = np.array([0.025, 0.025, 0.025])  # 3D conditional covariance

# Initialize and train 3D ProMP
promp = ProMP(n_dims=3, n_weights_per_dim=10)
promp.imitate([T] * n_demos, Y)

# Calculate original ProMP mean and confidence interval
Y_mean = promp.mean_trajectory(T)

# Define start and end points
start_point = [-1.0, -1.0, -0.5]  # Selected start position
end_point = [0.0, 0.5, 0.5]       # Selected end position

# Create ProMP with start position constraint only
start_only_promp = promp.condition_position(
    np.array(start_point),
    y_cov=y_conditional_cov,
    t=0.0,
    t_max=1.0
)
Y_start_only = start_only_promp.mean_trajectory(T)

# Create ProMP with both start and end position constraints
start_end_promp = promp.condition_position(
    np.array(start_point),  # Constraint on start position
    y_cov=y_conditional_cov,
    t=0.0,
    t_max=1.0
).condition_position(
    np.array(end_point),    # Constraint on end position
    y_cov=y_conditional_cov,
    t=1.0,
    t_max=1.0
)
Y_start_end = start_end_promp.mean_trajectory(T)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D ProMP Trajectory Comparison")

# Plot original ProMP trajectory
ax.plot(Y_mean[:, 0], Y_mean[:, 1], Y_mean[:, 2], c="blue", lw=3, label="Original ProMP")

# Plot trajectory with start constraint only
ax.plot(Y_start_only[:, 0], Y_start_only[:, 1], Y_start_only[:, 2], c="green", lw=3, label="Start Constrained")

# Plot trajectory with both start and end constraints
ax.plot(Y_start_end[:, 0], Y_start_end[:, 1], Y_start_end[:, 2], c="red", lw=3, label="Start & End Constrained")

# Mark start and end points
ax.scatter([start_point[0]], [start_point[1]], [start_point[2]],
           marker="o", s=200, c="black", label=f"Start Point: {start_point}")
ax.scatter([end_point[0]], [end_point[1]], [end_point[2]],
           marker="X", s=200, c="purple", label=f"End Point: {end_point}")

# Add labels and legend
ax.set_xlabel("X Position [m]")
ax.set_ylabel("Y Position [m]")
ax.set_zlabel("Z Position [m]")
ax.legend(loc="best")

plt.tight_layout()
plt.show()
