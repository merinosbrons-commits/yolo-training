
import numpy as np


def get_width_depth_height(points):
    dims = np.ptp(points, axis=0)
    return dims[0], dims[1], dims[2]

# nog 5 features bouwen

def load_features(objects: list[np.ndarray]) -> None:
    """
    Exports own own_data.txt
    """
    for points in objects:
        width, depth, height = get_width_depth_height(points)

    print("iets")

def load_features(objects: list[np.ndarray]) -> None:
    """
    Exporteert de features naar own_data.txt
    """
    all_rows = []

    for i, points in enumerate(objects):
        width, depth, height = get_width_depth_height(points)
        
        row = [i, width, depth, height]
        all_rows.append(row)

    output_matrix = np.array(all_rows).astype(np.float32)

    header = 'ID,label,width,depth,height'
    np.savetxt('own_data.txt', output_matrix, fmt='%10.5f', delimiter=',', header=header)
    
    print("Features succesvol opgeslagen in own_data.txt")