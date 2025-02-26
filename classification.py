# --- IMPORT SECTION ---
import numpy as np
import pandas as pd # load the data into a DataFrame
import matplotlib.pyplot as plt # to visualize the data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.preprocessing import StandardScaler # to standardize(scale) the features (the X)
from sklearn.ensemble import RandomForestClassifier # to use the RandomForest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # to evaluate the model
import seaborn as sns # to visualize the confusion
from sklearn.datasets import load_iris # to load the Iris dataset
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---

# Importing the dataset: the Iris dataset contains data of three species of flowers
dataset = load_iris() #recupera il dataset direttamente da internet(if it's connected to the net)

# Creating the DataFrame
data = pd.DataFrame(data = dataset.data, columns = dataset.features_names) # the features and the target (a.k.a. the X and the y)
data['target'] = dataset.target # the target (a.k.a. the y)

# Visualizing the first rows of the dataset
print(f"\nHere are the first rows of the dataset:\n{dataset.head()}")

# Separate the data in the features and target
X = data.iloc[:, :-1].values # all the columns except the last one
# iloc = fa selezionare insieme di righe e colonne a seconda di quello che mettiamo tra [ ]
y = data['target'].values # the last column

# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101, stratify = y)
# stratify parameter ensures that the classes are well-balanced between train and test















# --- END OF MAIN CODE   ---