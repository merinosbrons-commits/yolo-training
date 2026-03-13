import os

import numpy as np
from feature_creator import load_features
def main():
    data_path = 'pointclouds-500' 
    
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.xyz')])
    
    print(f"Start met het laden van {len(files)} objecten...")

    objects = []

    for filename in files:
        filepath = os.path.join(data_path, filename)
        
        points = np.loadtxt(filepath, dtype=np.float32)
        objects.append(points)

    load_features(objects)


    print("Alle data is succesvol benaderbaar!")

if __name__ == "__main__":
    main()