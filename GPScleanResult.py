import numpy as np
import timeit
import matplotlib.pyplot as plt
import math

# 1. 2D Polar -> Cartesian Transformation
def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)  # Convert polar radius and angle to Cartesian x-coordinate
    y = r * np.sin(theta)  # Convert polar radius and angle to Cartesian y-coordinate
    return x, y

# 2. 2D Cartesian -> Polar Transformation
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)  # Calculate radius in polar coordinates
    theta = np.arctan2(y, x)   # Calculate angle in polar coordinates
    return r, theta

# 3. 3D Spherical -> Cartesian Transformation
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)  # Convert spherical coordinates to Cartesian x
    y = r * np.sin(phi) * np.sin(theta)  # Convert spherical coordinates to Cartesian y
    z = r * np.cos(phi)                   # Convert spherical coordinates to Cartesian z
    return x, y, z

# 4. 3D Cartesian -> Spherical Transformation 
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # Calculate radius in spherical coordinates
    theta = np.arctan2(y, x)          # Calculate azimuthal angle in spherical coordinates
    phi = np.arccos(z / r)             # Calculate polar angle in spherical coordinates
    return r, theta, phi

# 5. 2D Polar distance Calculation
def distance_2d_polar(r1, theta1, r2, theta2):
    x1, y1 = polar_to_cartesian(r1, theta1)  # Convert first polar point to Cartesian
    x2, y2 = polar_to_cartesian(r2, theta2)  # Convert second polar point to Cartesian
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Calculate Euclidean distance between two points

# 6. 2D Cartesian distance Calculation
def distance_2d_cartesian(x1, y1, x2, y2):  # Calculate distance between two Cartesian points
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 7. 3D Cartesian distance Calculation
def distance_3d_cartesian(x1, y1, z1, x2, y2, z2):  # Calculate distance between two 3D Cartesian points
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# 8. 3D Spherical (Sphere) distance Calculation (throughout the volume)
def distance_3d_spherical(r1, theta1, phi1, r2, theta2, phi2):
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2 * (np.sin(phi1)*np.sin(phi2)*np.cos(theta1-theta2) + np.cos(phi1)*np.cos(phi2)))

# 9. 3D Spherical (Sphere) Great Circle distance Calculation (throughout the surface)
def great_circle_distance(r, theta1, phi1, theta2, phi2):
    delta_sigma = np.arccos(np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(theta1-theta2))  # Calculate angular separation
    return r * delta_sigma  # Return distance along the surface of the sphere

# 10. Performance comparison function
def benchmark_distance_calculations():
    num_points = 50000  # Selected array size for testing; aiming for minimal variability in comparison results.
    points_polar_2d = np.random.rand(num_points, 2) * [10, 2 * np.pi]  # Define points in polar coordinates [r, theta]
    points_cartesian_2d = np.array([polar_to_cartesian(r, theta) for r, theta in points_polar_2d])  # Convert points to Cartesian coordinates

    points_cartesian_3d = np.random.rand(num_points, 3) * 10  # Generate random points in Cartesian 3D space
    points_spherical_3d = np.array([cartesian_to_spherical(x, y, z) for x, y, z in points_cartesian_3d])  # Convert to spherical coordinates

    # 11. 2D Distance Calculations
    start = timeit.default_timer()  # Start timing
    for i in range(num_points - 1):
        distance_2d_cartesian(points_cartesian_2d[i][0], points_cartesian_2d[i][1], points_cartesian_2d[i+1][0], points_cartesian_2d[i+1][1])
    time_2d_cartesian = timeit.default_timer() - start  # Calculate time for Cartesian distance

    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_2d_polar(points_polar_2d[i][0], points_polar_2d[i][1], points_polar_2d[i+1][0], points_polar_2d[i+1][1])  # Calculate time for Polar distance
    time_2d_polar = timeit.default_timer() - start

    # 12. 3D Distance Calculations
    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_3d_cartesian(points_cartesian_3d[i][0], points_cartesian_3d[i][1], points_cartesian_3d[i][2],
                              points_cartesian_3d[i+1][0], points_cartesian_3d[i+1][1], points_cartesian_3d[i+1][2])  # Calculate time for 3D Cartesian distance
    time_3d_cartesian = timeit.default_timer() - start

    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_3d_spherical(points_spherical_3d[i][0], points_spherical_3d[i][1], points_spherical_3d[i][2],
                              points_spherical_3d[i+1][0], points_spherical_3d[i+1][1], points_spherical_3d[i+1][2])  # Calculate time for 3D Spherical distance
    time_3d_spherical = timeit.default_timer() - start

    # 13. 3D Great Circle Distance Calculations
    r_earth = 6371  # Radius of the Earth, km
    start = timeit.default_timer()
    for i in range(num_points - 1):
        great_circle_distance(r_earth, points_spherical_3d[i][1], points_spherical_3d[i][2], points_spherical_3d[i+1][1], points_spherical_3d[i+1][2])  # Calculate time for Great Circle distance
    time_great_circle = timeit.default_timer() - start

    return time_2d_cartesian, time_2d_polar, time_3d_cartesian, time_3d_spherical, time_great_circle

# 14. Print the benchmark results
def print_benchmark_results(times):
    print(f"2D Cartesian Distance Calculation Time: {times[0]:.6f} second")
    print(f"2D Polar Distance Calculation Time: {times[1]:.6f} second")
    print(f"3D Cartesian Distance Calculation Time: {times[2]:.6f} second")
    print(f"3D Spherical Distance Calculation Time: {times[3]:.6f} second")
    print(f"Great Circle Distance Calculation Time: {times[4]:.6f} second")

# 15. Plot performance results
def plot_benchmark_results(times):
    methods = ['2D Cartesian', '2D Polar', '3D Cartesian', '3D Spherical', 'Great Circle']
    plt.bar(methods, times, color=['blue', 'green', 'red', 'purple', 'orange'])  # Create bar chart
    plt.xlabel('Distance Calculation Methods')
    plt.ylabel('Time (Second)')
    plt.title('Performance Comparison of Distance Calculation Methods')
    plt.show()

# 16. Plot 2D Polar and Cartesian coordinates
def plot_coordinates(points_cartesian, points_polar): 
    plt.figure(figsize=(10, 5))

    # 2D Cartesian Coordinates Plot
    plt.subplot(1, 2, 1)
    plt.scatter(points_cartesian[:, 0], points_cartesian[:, 1], alpha=0.7)
    plt.title('Cartesian Coordinates (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', lw=0.5, ls='--')  # Draw x-axis
    plt.axvline(0, color='black', lw=0.5, ls='--')  # Draw y-axis
    plt.grid()

    # Create a polar subplot
    ax = plt.subplot(1, 2, 2, projection='polar')  # Correctly set the subplot as polar
    ax.scatter(points_polar[:, 1], points_polar[:, 0], alpha=0.7)  # Use scatter instead of plt.polar
    ax.set_title('Polar Coordinates (r, θ)')
    ax.grid()

    plt.tight_layout()
    plt.show()

# 17. Main execution
if __name__ == "__main__":
    # Execute benchmark tests and print results
    times = benchmark_distance_calculations()  # Run performance benchmarks
    print_benchmark_results(times)  # Display benchmark results
    plot_benchmark_results(times)  # Plot benchmark results

    # 18.Generate random polar coordinates for plotting
    num_points = 10000  # Number of points to plot
    points_polar = np.random.rand(num_points, 2) * [10, 2 * np.pi]  # Generate random polar points
    points_cartesian = np.array([polar_to_cartesian(r, theta) for r, theta in points_polar])  # Convert polar to Cartesian

    # Plot the generated coordinates
    plot_coordinates(points_cartesian, points_polar)  # Visualize coordinates
