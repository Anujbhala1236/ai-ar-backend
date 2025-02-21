import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ✅ Load the dataset (pose landmarks stored as NumPy arrays)
data = np.load("pose_dataset.npy")  # Shape: (samples, 33*2)
labels = np.load("pose_labels.npy")  # Shape: (samples,)

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# ✅ Train a K-Nearest Neighbors model (can replace with deep learning model)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ✅ Save the trained model
with open("pose_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("✅ Model trained & saved as pose_model.pkl")
