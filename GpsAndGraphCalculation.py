import numpy as np
import timeit
import matplotlib.pyplot as plt
import math

#  2D Polar -> Cartesian Transformation
def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# 2D Cartesian -> Polar Transformation
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

# 3D Spherical -> Cartesian Transformation
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

# 3D Cartesian -> Spherical Transformation 
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi

# 2D Polar distance Calculation
def distance_2d_polar(r1, theta1, r2, theta2):
    x1, y1 = polar_to_cartesian(r1, theta1)
    x2, y2 = polar_to_cartesian(r2, theta2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 2D Cartesian distance Calculation
def distance_2d_cartesian(x1, y1, x2, y2):  # fonksiyonu 2D Kartezyen koordinatlar arasındaki mesafeyi hesaplıyor.
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 3D Cartesian mesafe hesaplama
def distance_3d_cartesian(x1, y1, z1, x2, y2, z2): # fonksiyonu 3D Kartezyen koordinatlar arasındaki mesafeyi hesaplıyor
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# 3D Spherical (Sphre) distance Calculation (throughout the volume)
def distance_3d_spherical(r1, theta1, phi1, r2, theta2, phi2):
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2 * (np.sin(phi1)*np.sin(phi2)*np.cos(theta1-theta2) + np.cos(phi1)*np.cos(phi2)))

# 3D Spherical (Sphre) büyük daire mesafe hesaplama (througout the surface)
def great_circle_distance(r, theta1, phi1, theta2, phi2):
    delta_sigma = np.arccos(np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(theta1-theta2))
    return r * delta_sigma

# Performance comparison function
def benchmark_distance_calculations():
    num_points = 10000 # Number of points to visualize
    points_polar_2d = np.random.rand(num_points, 2) * [10, 2 * np.pi]  # [r, theta]  # Define some points in polar and cartesian coordinates for 2D visualization and verification  1-Kutupsal koordinat sistemindeki birkaç noktanın koordinatlarını tanımla.
    points_cartesian_2d = np.array([polar_to_cartesian(r, theta) for r, theta in points_polar_2d])  #2-Bu koordinatları Kartezyen koordinat sistemine dönüştür. 

    points_cartesian_3d = np.random.rand(num_points, 3) * 10  # [x, y, z]
    points_spherical_3d = np.array([cartesian_to_spherical(x, y, z) for x, y, z in points_cartesian_3d])

    # 2D  Distance Calculations
    start = timeit.default_timer() #timeit.default_timer() ile her mesafe hesaplama işleminin süresi ölçülmüş
    for i in range(num_points - 1):
        distance_2d_cartesian(points_cartesian_2d[i][0], points_cartesian_2d[i][1], points_cartesian_2d[i+1][0], points_cartesian_2d[i+1][1])
    time_2d_cartesian = timeit.default_timer() - start

    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_2d_polar(points_polar_2d[i][0], points_polar_2d[i][1], points_polar_2d[i+1][0], points_polar_2d[i+1][1]) # fonksiyonu, 2D kutupsal koordinatlar arasındaki mesafeyi hesaplıyor.
    time_2d_polar = timeit.default_timer() - start

    # 3D  Distance Calculations
    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_3d_cartesian(points_cartesian_3d[i][0], points_cartesian_3d[i][1], points_cartesian_3d[i][2],
                              points_cartesian_3d[i+1][0], points_cartesian_3d[i+1][1], points_cartesian_3d[i+1][2])
    time_3d_cartesian = timeit.default_timer() - start

    start = timeit.default_timer()
    for i in range(num_points - 1):
        distance_3d_spherical(points_spherical_3d[i][0], points_spherical_3d[i][1], points_spherical_3d[i][2],
                              points_spherical_3d[i+1][0], points_spherical_3d[i+1][1], points_spherical_3d[i+1][2])
    time_3d_spherical = timeit.default_timer() - start

    # 3D Great Circle Distance Calculations
    r_earth = 6371  # Radius of the Earth, km
    start = timeit.default_timer()
    for i in range(num_points - 1):
        great_circle_distance(r_earth, points_spherical_3d[i][1], points_spherical_3d[i][2], points_spherical_3d[i+1][1], points_spherical_3d[i+1][2])
    time_great_circle = timeit.default_timer() - start

    return time_2d_cartesian, time_2d_polar, time_3d_cartesian, time_3d_spherical, time_great_circle

# Print the results to the screen
def print_benchmark_results(times):  # Print the benchmark results and plot them
    print(f"2D Cartesian Distance Calculation Time: {times[0]:.6f} second")
    print(f"2D Polar Distance Calculation Time: {times[1]:.6f} second")
    print(f"3D Cartesian Distance Calculation Time: {times[2]:.6f} second")
    print(f"3D Spherical Distance Calculation Time: {times[3]:.6f} second")
    print(f"Great Circle Distance Calculation Time: {times[4]:.6f} second")

# Plot performance results
def plot_benchmark_results(times):
    methods = ['2D Cartesian', '2D Polar', '3D Cartesian', '3D Spherical', 'Great Circle']
    plt.bar(methods, times, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Distance Calculation Methods')
    plt.ylabel('Time (Second)')
    plt.title('Performance Comparison of Distance Calculation Methods')
    plt.show()

# Grafik Gösterimi  -Graphic Display
def plot_coordinates(points_cartesian, points_polar): # Plot 2D polar and cartesian coordinates # Plot the coordinates
    plt.figure(figsize=(10, 5))

    # Kartezyen Koordinatlar
    plt.subplot(1, 2, 1)
    plt.scatter(points_cartesian[:, 0], points_cartesian[:, 1], alpha=0.7)
    plt.title('Kartezyen Koordinatlar (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()

    # Polar Koordinatlar
    plt.subplot(1, 2, 2, polar=True)
    r = points_polar[:, 0]
    theta = points_polar[:, 1]

    # Polar grafik için bir çember çizimi - #llustration of a circle for polar chart
    ax = plt.subplot(1, 2, 2, polar=True)
    ax.set_theta_zero_location('N')  # Açının başlangıcı üstte -The beginning of the angle is at the top
    ax.set_theta_direction(-1)        # Saat yönünün tersine dön -Turn counterclockwise
    ax.grid(True)                     # Izgara çizgileri -grid lines

    # Polar noktaları çiz
    ax.scatter(theta, r, alpha=0.7, color='blue', marker='o')

    plt.tight_layout()
    plt.show()
 # 7. Dönüşüm
def verify_transformations(points_cartesian, points_polar):  # Verify the transformations 
    print("Verifying Polar to Cartesian and back:")
    for i, (r, theta) in enumerate(points_polar):
        x, y = points_cartesian[i]
        r_trans, theta_trans = cartesian_to_polar(x, y)
        print(f"\nPoint {i + 1}:")
        print(f"Original:    (r={r:.5f}, theta={theta:.5f})")
        print(f"Transformed: (r={r_trans:.5f}, theta={theta_trans:.5f})")
        assert np.isclose(r, r_trans), "Radius mismatch!"
        assert np.isclose(theta, theta_trans), "Theta mismatch!"

        # 8. Ters dönüşüm doğrulaması   ->Verify the accuracy of the calculations by ensuring that the starting coordinates match those obtained after the inverse transformation.
def verify_spherical_transformations(points_spherical):
    print("Verifying Spherical to Cartesian and back:")
    for i, (r, theta, phi) in enumerate(points_spherical):
        x, y, z = spherical_to_cartesian(r, theta, phi)
        r_trans, theta_trans, phi_trans = cartesian_to_spherical(x, y, z)
        print(f"\nPoint {i + 1}:")
        print(f"Original:    (r={r:.5f}, theta={theta:.5f}, phi={phi:.5f})")
        print(f"Transformed: (r={r_trans:.5f}, theta={theta_trans:.5f}, phi={phi_trans:.5f})")
        assert np.isclose(r, r_trans), "Radius mismatch!"
        assert np.isclose(theta, theta_trans), "Theta mismatch!"
        assert np.isclose(phi, phi_trans), "Phi mismatch!"

   # Conversions and verification  # Dönüşümler ve doğrulama 
    print("Conversion from spherical coordinates to Cartesian and verification of inverse transformation:")  #Küresel koordinatlardan Kartezyene dönüşüm ve ters dönüşüm doğrulaması
    verify_spherical_transformations(points_spherical)
    
   # Great circle distance calculation  # Büyük daire mesafesi hesaplama  # 9. Mesafe hesaplama
    print("\nSurface distance between spherical coordinates (Great Circle):")
    r = 6371  # Dünya yarıçapı
    for i in range(len(points_spherical) - 1):
        distance = great_circle_distance(r, points_spherical[i][1], points_spherical[i][2], points_spherical[i + 1][1], points_spherical[i + 1][2])
        print(f"Point {i + 1} with Point {i + 2} Distance between: {distance:.2f} km")

   # Farthest distance calculation  # En uzak mesafe hesaplama  # 10. 3D Kartesyen Mesafe Hesabı
    print("\nFarthest distance between all points:")
    max_distance = 0
    for i in range(len(points_spherical)):
        for j in range(i + 1, len(points_spherical)):
            distance = great_circle_distance(r, points_spherical[i][1], points_spherical[i][2], points_spherical[j][1], points_spherical[j][2])
            if distance > max_distance:
                max_distance = distance
                max_pair = (i + 1, j + 1)  # 1 tabanlı indeksleme
    print(f"Point {max_pair[0]} with Point {max_pair[1]} farthest distance between: {max_distance:.2f} km")

def test_spherical_coordinate_transformations(): # Test spherical transformations and great circle distance calculation
    # Sphere koordinatlar (r, theta, phi)
    points_spherical = np.array([
        [1, np.pi / 6, np.pi / 3],
        [2, np.pi / 4, np.pi / 6],
        #[3, np.pi / 3, np.pi / 4], #Remove the comment line if you want to enable more points
        #[4, np.pi / 2, np.pi / 2], #Remove the comment line if you want to enable more points
        [5, np.pi, np.pi / 3]
    ])

def test_global_coordinate_transformations():
  
    # 6. Küresel koordinat sisteminde birkaç nokta tanımlanması
    points_global = np.array([
        [1, np.pi / 6, np.pi / 3],  # (r, theta, phi)
        [2, np.pi / 4, np.pi / 6],
        [3, np.pi / 3, np.pi / 4],
        [4, np.pi / 2, np.pi / 2]
    ]) 
    # 7. Dönüşüm
    print("Küresel koordinatları Kartezyen koordinatlara dönüştürme:")
    points_cartesian_from_global = []
    for r, theta, phi in points_global:
        x, y, z = spherical_to_cartesian(r, theta, phi)
        points_cartesian_from_global.append((x, y, z))
    
    points_cartesian_from_global = np.array(points_cartesian_from_global)
    
    # 8. Ters dönüşüm doğrulaması
    print("Kartezyen koordinatları tekrar küresel koordinatlara dönüştürme ve doğrulama:")
    for i, (x, y, z) in enumerate(points_cartesian_from_global):
        r_trans, theta_trans, phi_trans = cartesian_to_spherical(x, y, z)
        print(f"\nPoint {i + 1}:")
        print(f"Original:    (r={points_global[i][0]:.5f}, theta={points_global[i][1]:.5f}, phi={points_global[i][2]:.5f})")
        print(f"Transformed: (r={r_trans:.5f}, theta={theta_trans:.5f}, phi={phi_trans:.5f})")
        assert np.isclose(points_global[i][0], r_trans), "Radius mismatch!"
        assert np.isclose(points_global[i][1], theta_trans), "Theta mismatch!"
        assert np.isclose(points_global[i][2], phi_trans), "Phi mismatch!"
    
    # 9. Mesafe hesaplama
    print("\nKüresel koordinatlar arasındaki mesafeleri hesaplama:")
    r = 6371  # Dünya yarıçapı
    for i in range(len(points_global) - 1):
        distance = great_circle_distance(r, points_global[i][1], points_global[i][2], points_global[i + 1][1], points_global[i + 1][2])
        print(f"Point {i + 1} ile Point {i + 2} arasındaki mesafe: {distance:.2f} km")
        
    # 10. 3D Kartesyen Mesafe Hesabı
    print("\n3D Kartesyen koordinatları arasındaki mesafeleri hesaplama:")
    for i in range(len(points_cartesian_from_global) - 1):
        distance = distance_3d_cartesian(points_cartesian_from_global[i][0], points_cartesian_from_global[i][1], points_cartesian_from_global[i][2],
                                          points_cartesian_from_global[i + 1][0], points_cartesian_from_global[i + 1][1], points_cartesian_from_global[i + 1][2])
        print(f"Point {i + 1} ile Point {i + 2} arasındaki mesafe: {distance:.2f} km")

    # 11. Kutupsal koordinatlar arasındaki mesafe
    print("\nKutup koordinatları arasındaki mesafeleri hesaplama:")
    for i in range(len(points_global) - 1):
        distance = distance_2d_polar(points_global[i][0], points_global[i][1], points_global[i + 1][0], points_global[i + 1][1])
        print(f"Point {i + 1} ile Point {i + 2} arasındaki kutupsal mesafe: {distance:.2f} km")
  

# Main function
if __name__ == "__main__":
        # Run the benchmark tests for distance calculation
    times = benchmark_distance_calculations() # Kodun çalıştırılması distance_2d_polar() fonksiyonu, 2D kutupsal koordinatlar arasındaki mesafeyi hesaplıyor.

    print_benchmark_results(times)    # Print the results
    plot_benchmark_results(times)  # Plot the performance results
       # Run tests
    test_spherical_coordinate_transformations()  # Fonksiyonu çağırın
    test_global_coordinate_transformations()

#Saving chart data
points_polar = np.array([(r, theta) for r, theta in [(5, np.pi / 6), (10, np.pi / 4), (7, np.pi / 3)]])
points_cartesian = np.array([polar_to_cartesian(r, theta) for r, theta in points_polar])

# Show Koordinatları 
plot_coordinates(points_cartesian, points_polar)

 # Conversion verification in console
verify_transformations(points_cartesian, points_polar)
