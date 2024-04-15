# Import the pickle module for serializing and deserializing Python object structures
import pickle
# Import NumPy for array handling
import numpy as np
# Import machine learning models from scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Import Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression
# Import Support Vector Machine model from scikit-learn
from sklearn.svm import SVC
# Import utility to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import function to compute accuracy score
from sklearn.metrics import accuracy_score

# Load serialized data from file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Initialize lists to store filtered data and corresponding labels
filtered_data = []
filtered_labels = []

# Loop to filter entries with exactly 42 features
for i, entry in enumerate(data_dict['data']):
    if len(entry) == 42:
        # Append entries with exactly 42 features to the filtered lists
        filtered_data.append(entry)
        filtered_labels.append(data_dict['labels'][i])
    else:
        # Output the index of entries that do not meet the criterion
        print(i)

# Convert filtered lists to NumPy arrays for model compatibility
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Output the number of entries in the filtered data
print(len(data))

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define a dictionary of classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC()
}

# Loop through the classifiers to train and evaluate them
for name, model in classifiers.items():
    # Fit model on training data
    model.fit(x_train, y_train)
    # Predict on testing data
    y_predict = model.predict(x_test)
    # Calculate and print accuracy score
    score = accuracy_score(y_predict, y_test)
    print(f"{name}: {score * 100:.2f}% of samples were classified correctly!")

# Serialize and save the Random Forest model to file
with open('model.p', 'wb') as f:
    pickle.dump({'model': classifiers['Random Forest']}, f)
