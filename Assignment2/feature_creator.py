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
    """
    Calculates the 2D bounding box elongation of a point cloud.
    
    This feature measures the ratio between the longest and shortest sides 
    of the object's footprint in the XY plane.
               
    Note:
        Unlike raw dimensions, this is scale-invariant. A 1x2m object 
        and a 10x20m object will both result in an elongation of 2.0.
    """
    # bounding box elongation, de verhouding tussen de langste en kortste zijde van de bounding box
    x_range = np.max(points[:,0]) - np.min(points[:,0])
    y_range = np.max(points[:,1]) - np.min(points[:,1])

    elongation = max(x_range, y_range) / (min(x_range, y_range) + 1e-5)

    return elongation

# @merinosbrons-commits: isn't this the same as height, width and depth? 
# I think that they are correlate 100% 
# Same counts for elongation
def density(width, depth, height, n_points):
    """
    Calculates the point density relative to the 3D bounding box volume.
    
    This feature represents how "packed" the points are within the 
    spatial bounds occupied by the object.

    Note:
        This is distinct from dimensions because it introduces the 
        variable of point count (N). Two objects with identical 
        height, width, and depth can have vastly different densities 
        (e.g., a solid wall vs. a sparse tree).
    """
    volume = width * depth * height
    return n_points / (volume + 1e-5)

# Maarten
def feature5():
    ...

def feature6():
    ...


def load_features(objects: list[np.ndarray]) -> None:
    all_rows = []
    n_points = len(objects)

    for i, points in enumerate(objects):
        width, depth, height = get_width_depth_height(points)
        # NOTE: Hier kan je dus je twee eigen features inladen
        elongation_laden = elongation(width, depth, height, n_points)
        density_laden= density(points)
        row = [i, width, depth, height, elongation_laden, density_laden]
        all_rows.append(row)

    output_matrix = np.array(all_rows).astype(np.float32)

    # NOTE: vergeet 'm niet aan de header toe te voegen
    header = 'ID,label,width,depth,height,elongation,density'
    np.savetxt('own_data.txt', output_matrix, fmt='%10.5f', delimiter=',', header=header)
    
    print("Features succesvol opgeslagen in own_data.txt")