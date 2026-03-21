"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = {}

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        xyz_max = np.amax(self.points, axis=0)
        self.feature["width"] = xyz_max[0]
        self.feature["depth"] = xyz_max[1]
        self.feature["height"] = xyz_max[2]

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature["root_density"] = root_density

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature["hull_area"] = hull_area

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature["shape_index"] = shape_index

        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)

        # Get eigenvalues and sort descending: L1 >= L2 >= L3 as decriped in the paper
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1] 
        l1, l2, l3 = evals
        
        # Add a small epsilon to prevent division by zero
        self.eps = 1e-5

        # Apply formulas from Table 1 of the paper 
        self.feature.update({
            "sum_eigen":    np.sum(evals),
            "omnivariance": np.prod(np.maximum(evals, self.eps))**(1/3),
            "eigenentropy": -np.sum(evals * np.log(evals + self.eps)),
            "linearity":    (l1 - l2) / (l1 + self.eps),
            "planarity":    (l2 - l3) / (l1 + self.eps),
            "sphericity":   l3 / (l1 + self.eps),
            "curvature":    l3 / (np.sum(evals) + self.eps),
            "n_points":    int(self.points.shape[0]) 
        })

        # Our own (creative) functions
        self.elongation()
        self.bbox_density()
        self.density_profile_per_segment()
        
    # The functions we came up with ourselves (without papers, just creativity)
    def elongation(self):
        self.feature["elong_xy"] = max(self.feature["width"], self.feature["depth"]) / (min(self.feature["width"], self.feature["depth"]) + self.eps)
        self.feature["elong_yz"] = max(self.feature["depth"], self.feature["height"]) / (min(self.feature["depth"], self.feature["height"]) + self.eps)
        self.feature["elong_xz"] = max(self.feature["width"], self.feature["height"]) / (min(self.feature["width"], self.feature["height"]) + self.eps)

    def bbox_density(self):
        volume = self.feature["width"] * self.feature["height"] * self.feature["depth"]
        self.feature["volume"] = self.feature["n_points"] / (volume + self.eps)

    # NOTE: n_segments is a hyperparameter, for now it's 6 (no arguments)
    def density_profile_per_segment(self, n_segments=6):
        mins = np.min(self.points, axis=0)
        normalized_points = self.points - mins
        
        # Calculate histograms for each axis
        for axis, name in enumerate(['prof_x', 'prof_y', 'prof_z']):
            hist, _ = np.histogram(normalized_points[:, axis], bins=n_segments, density=True)
            for j in range(n_segments):
                self.feature[f"{name}_{j}"] = hist[j]


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    # if exists(data_file):
    #     return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + list(i_object.feature.values())
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    feature_names = ",".join(i_object.feature.keys())
    data_header = 'ID,label,' + feature_names
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def load_feature_names(data_file='data.txt'):
    """
    Read feature names from the header of the saved data file
        data_file: the local file that contains feature names in the header
    """
    with open(data_file, 'r') as f_input:
        header = f_input.readline().strip()

    if header.startswith('#'):
        header = header[1:].strip()

    columns = [col.strip() for col in header.split(',')]
    return columns[2:]


def compute_scatter_matrices(X, y):
    """
    Compute within-class scatter matrix S_W and between-class scatter matrix S_B
        X: input features with shape (n_samples, n_features)
        y: labels with shape (n_samples,)
    """
    classes = np.unique(y)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    mu = np.mean(X, axis=0)
    S_W = np.zeros((n_features, n_features), dtype=np.float64)
    S_B = np.zeros((n_features, n_features), dtype=np.float64)

    for cls in classes:
        X_k = X[y == cls]
        n_k = X_k.shape[0]
        mu_k = np.mean(X_k, axis=0)

        if n_k > 1:
            sigma_k = np.cov(X_k, rowvar=False, bias=False)
        else:
            sigma_k = np.zeros((n_features, n_features), dtype=np.float64)

        mean_diff = (mu_k - mu).reshape(-1, 1)

        S_W += (n_k / n_samples) * sigma_k
        S_B += (n_k / n_samples) * (mean_diff @ mean_diff.T)

    return S_W, S_B


def fisher_score_subset(X_subset, y, regularization=1e-8):
    """
    Score a feature subset using trace(inv(S_W) * S_B)
        X_subset: selected features with shape (n_samples, n_selected_features)
        y: labels
        regularization: small value for numerical stability
    """
    S_W, S_B = compute_scatter_matrices(X_subset, y)
    S_W = S_W + regularization * np.eye(S_W.shape[0])
    score = np.trace(np.linalg.pinv(S_W) @ S_B)
    return float(score)


def rank_features_by_fisher(X, y, feature_names):
    """
    Rank individual features using the Fisher criterion S_B / S_W in 1D
        X: input features
        y: labels
        feature_names: names of the features
    """
    ranking = []

    for idx, name in enumerate(feature_names):
        Xi = X[:, [idx]]
        score = fisher_score_subset(Xi, y)
        ranking.append((name, idx, score))

    ranking.sort(key=lambda item: item[2], reverse=True)
    return ranking


def select_features_greedy(X, y, feature_names, n_select=5):
    """
    Greedy forward feature selection using the Fisher subset score
        X: input features
        y: labels
        feature_names: names of all features
        n_select: number of features to select
    """
    n_select = min(n_select, X.shape[1])
    selected_indices = []
    remaining_indices = list(range(X.shape[1]))

    for _ in range(n_select):
        best_candidate = None
        best_score = -np.inf

        for idx in remaining_indices:
            candidate_indices = selected_indices + [idx]
            candidate_score = fisher_score_subset(X[:, candidate_indices], y)

            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = idx

        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)

    selected_names = [feature_names[idx] for idx in selected_indices]
    return selected_indices, selected_names


def feature_selection_report(X, y, feature_names, n_select=4):
    """
    Print a ranking of individual features and a greedy-selected subset
    """
    ranking = rank_features_by_fisher(X, y, feature_names)
    selected_indices, selected_names = select_features_greedy(X, y, feature_names, n_select=n_select)

    print("\nTop individual features based on Fisher score:")
    for rank_i, (name, idx, score) in enumerate(ranking[:10], start=1):
        print("%2d. %-20s idx=%2d score=%10.5f" % (rank_i, name, idx, score))

    print("\nGreedy selected feature subset:")
    for idx, name in zip(selected_indices, selected_names):
        print("idx=%2d  name=%s" % (idx, name))

    subset_score = fisher_score_subset(X[:, selected_indices], y)
    print("Subset Fisher score: %10.5f" % subset_score)

    return ranking, selected_indices, selected_names


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 3], X[100*i:100*(i+1), 4], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel('x1:root density')
    ax.set_ylabel('x2:area')
    ax.legend()


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    
    grid = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    best_clf = grid.best_estimator_
    print(f"Best SVM model: {grid.best_params_} ")
    
    preds = best_clf.predict(X_test)
    print(f"SVM Accuracy (on test set): {accuracy_score(y_test, preds):.2f}")
    return best_clf


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    print("\n--- RF Optimalisatie (Sectie 2.2) ---")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    best_clf = grid.best_estimator_
    print(f"Beste RF model: {grid.best_params_} ")
    
    preds = best_clf.predict(X_test)
    print(f"RF Accuracy (on test set): {accuracy_score(y_test, preds):.2f}")
    return best_clf

def plot_learning_curve(clf, X, y, title="Learning Curve"):
    """
    Handmatige implementatie van de learning curve.
    Slaat het resultaat op in de map 'results'.
    """
    train_ratios = np.linspace(0.1, 0.9, 9)
    train_scores = []
    test_scores = []
    
    print(f"\nBezig met genereren van {title}...")
    
    for ratio in train_ratios:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ratio, random_state=42
        )
        clf.fit(X_train, y_train)
        
        train_score = accuracy_score(y_train, clf.predict(X_train))
        test_score = accuracy_score(y_test, clf.predict(X_test))
        
        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.figure(figsize=(8, 5))
    plt.plot(train_ratios, train_scores, 'o-', label="Training Accuracy")
    plt.plot(train_ratios, test_scores, 's-', label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Ratio")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    filename = f"results/learning_curve_{title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.png"    
    plt.savefig(filename)
    print(f"Figuur opgeslagen: {filename}")
    plt.close()

def evaluate_model_performance(clf, X, y, model_name="Classifier"):
    """
    Analyseert resultaten en slaat confusion matrix op in 'results'.
    """
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Building', 'Car', 'Fence', 'Pole', 'Tree'] 
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Ware Label')
    plt.xlabel('Voorspeld Label')
    
    # Opslaan
    filename = f"results/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Figuur opgeslagen: {filename}")
    plt.close()
    
    print(f"\n--- {model_name} Rapport ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__=='__main__':
    path = 'pointclouds-500' # Pas aan naar jouw lokale pad

    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Voorbereiding & Laden
    feature_preparation(data_path=path)
    ID, X, y = data_loading()
    feature_names = load_feature_names()

    # 2. Feature Selectie: Selecteer exact 4 kenmerken 
    ranking, selected_indices, selected_names = feature_selection_report(X, y, feature_names, n_select=4)
    X_selected = X[:, selected_indices]

    # 3. Model Optimalisatie (SVM & RF)
    best_svm = SVM_classification(X_selected, y)
    best_rf = RF_classification(X_selected, y)

    # 4. Learning Curves genereren (Verplicht onderdeel)
    plot_learning_curve(best_svm, X_selected, y, title="Learning Curve: SVM (Top 4 Features)")
    plot_learning_curve(best_rf, X_selected, y, title="Learning Curve: Random Forest")

    # 5. Error Analyse & Confusion Matrices
    evaluate_model_performance(best_svm, X_selected, y, model_name="SVM")
    evaluate_model_performance(best_rf, X_selected, y, model_name="Random Forest")
