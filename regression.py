# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns

# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split # to perform train-test split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)
#print(data.head())

# Visualizing the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

# Separete the data in fetaures adn target
X = data["YearsExperience"]
y = data["Salary"]

# Using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(X, y, color="red") # actual plot
sns.regplot(data = data, x = "YearsExperience", y = "Salary") #regression line
plt.show() # renderize the plot to show it

# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# Checking the train and the test size to prove they are 80% and 20% respectively
print(f"\nThe total X size is: {X.shape[0]}")
print(f"The X_train size is: {X_train.shape[0]} and is the {X_train.shape[0] / X.shape[0] * 100}% of the total X")
print(f"The X_test size is: {X_test.shape[0]} and is the {X_test.shape[0] / X.shape[0] * 100}% of the total X")

# --- END OF MAIN CODE   ---
