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
def feature3():
    ...

def feature4():
    ...

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
        
        row = [i, width, depth, height]
        all_rows.append(row)

    output_matrix = np.array(all_rows).astype(np.float32)

    # NOTE: vergeet 'm niet aan de header toe te voegen
    header = 'ID,label,width,depth,height'
    np.savetxt('own_data.txt', output_matrix, fmt='%10.5f', delimiter=',', header=header)
    
    print("Features succesvol opgeslagen in own_data.txt")