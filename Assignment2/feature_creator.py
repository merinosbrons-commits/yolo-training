import numpy as np


def get_width_depth_height(points):
    dims = np.ptp(points, axis=0)
    return dims[0], dims[1], dims[2]

# NOTE: Geef je feature een passende naam
# Stijn
def feature1():
    ...

def feature2():
    ...

# Merijn
def elongation(points):
# bounding box elongation, de verhouding tussen de langste en kortste zijde van de bounding box
    x_range = np.max(points[:,0]) - np.min(points[:,0])
    y_range = np.max(points[:,1]) - np.min(points[:,1])

    elongation = max(x_range, y_range) / (min(x_range, y_range) + 1e-5)

    return elongation

def density(points):

    n_points = len(points)

    x_range = np.max(points[:,0]) - np.min(points[:,0])
    y_range = np.max(points[:,1]) - np.min(points[:,1])
    z_range = np.max(points[:,2]) - np.min(points[:,2])

   
    volume = x_range * y_range * z_range

    density = n_points / (volume + 1e-5)

    return density

# Maarten
def feature5():
    ...

def feature6():
    ...


def load_features(objects: list[np.ndarray]) -> None:
    all_rows = []

    for i, points in enumerate(objects):
        width, depth, height = get_width_depth_height(points)
        # NOTE: Hier kan je dus je twee eigen features inladen
        elongation_laden = elongation(points)
        density_laden= density(points)
        row = [i, width, depth, height, elongation_laden, density_laden]
        all_rows.append(row)

    output_matrix = np.array(all_rows).astype(np.float32)

    # NOTE: vergeet 'm niet aan de header toe te voegen
    header = 'ID,label,width,depth,height,elongation,density'
    np.savetxt('own_data.txt', output_matrix, fmt='%10.5f', delimiter=',', header=header)
    
    print("Features succesvol opgeslagen in own_data.txt")