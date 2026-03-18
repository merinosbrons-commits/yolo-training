import numpy as np


# TODO: check if we're getting the right order here. Are they indeed width, depth, height?
# This is a TODO since it is not 
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
def get_plane_elongations(width, depth, height):
    """
    Calculates elongation ratios for the three primary orthogonal planes.
    
    This provides a 'shape profile' by looking at the object from the 
    Top (XY), Side (YZ), and Front (XZ) perspectives.
    
    Returns:
        tuple: (elong_xy, elong_yz, elong_xz)
    """
    eps = 1e-5
    
    # Top-down perspective (Footprint)
    elong_xy = max(width, depth) / (min(width, depth) + eps)
    
    # Side perspective (Profile)
    elong_yz = max(depth, height) / (min(depth, height) + eps)
    
    # Front perspective (Cross-section)
    elong_xz = max(width, height) / (min(width, height) + eps)
    
    return elong_xy, elong_yz, elong_xz

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
        
        # Calculate features
        exy, eyz, exz = get_plane_elongations(width, depth, height)
        dens = density(width, depth, height, n_points)

        # Add row
        row = [i, width, depth, height, exy, eyz, exz, dens]
        all_rows.append(row)

    output_matrix = np.array(all_rows).astype(np.float32)

    # NOTE: vergeet 'm niet aan de header toe te voegen
    header = 'ID,width,depth,height,elong_xy,elong_yz,elong_xz,density'    
    np.savetxt('own_data.txt', output_matrix, fmt='%10.5f', delimiter=',', header=header)
    
    print("Features succesvol opgeslagen in own_data.txt")