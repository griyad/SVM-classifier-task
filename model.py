import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from PIL import Image

def load(data_dir):
    X = []
    y = []
    for file in os.listdir(os.path.join(data_dir, "cats")):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, "cats", file)
            img = Image.open(img_path).resize((64, 64)).convert("L")
            img_arr = np.array(img).flatten() / 255.0
            X.append([np.mean(img_arr), np.var(img_arr)])
            y.append(0)
    for file in os.listdir(os.path.join(data_dir, "dogs")):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, "dogs", file)
            img = Image.open(img_path).resize((64, 64)).convert("L")
            img_arr = np.array(img).flatten() / 255.0
            X.append([np.mean(img_arr), np.var(img_arr)])
            y.append(1)
    return np.array(X), np.array(y)
data = "/Users/riyad/Desktop/coding/ML/svm-project/test_set"
X, y = load(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

def predict(content):
    return model.predict(content)