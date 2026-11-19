# Import the CSV module
import csv

# Open the Iris dataset
with open("iris.csv", "r", newline="") as file:

    # Read the dataset using column names
    reader = csv.DictReader(file)

    # Store all records
    rows = list(reader)

# Store the four measurements as features (X)
X = [
    [
        float(row["sepal_length"]),
        float(row["sepal_width"]),
        float(row["petal_length"]),
        float(row["petal_width"])
    ]
    for row in rows
]

# Store the flower species as the target (y)
y = [row["species"] for row in rows]

# Display the first feature record
print("First feature record:", X[0])

# Display its corresponding species
print("First target:", y[0])

# Import the function used to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Divide the dataset: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the number of records in each set
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Import the K-Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

# Import tools for building a machine-learning pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Import tools for cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Create a pipeline that scales features before applying KNN
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# Divide the training data into 5 stratified folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

# Display the accuracy for each fold
print("\nCross-validation scores:", cv_scores)

# Display the average cross-validation accuracy
print("Mean cross-validation accuracy:", cv_scores.mean() * 100, "%")

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Display the first five actual and predicted species
print("\nActual species:", y_test[:5])
print("Predicted species:", y_pred[:5])

# Import evaluation metrics from Scikit-learn
from sklearn.metrics import accuracy_score, classification_report

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy as a percentage
print("\nModel Accuracy:", accuracy * 100, "%")

# Generate precision, recall, and F1-score for each species
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Import tools for creating and displaying the confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Create the confusion matrix using actual and predicted values
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Setosa", "Versicolor", "Virginica"],
    cmap="Blues"
)

# Add a title to the graph
plt.title("Iris Classification - Confusion Matrix")

# Adjust the graph layout
plt.tight_layout()

# Display the graph
plt.show()

# Create a new figure for the scatter plot
plt.figure(figsize=(8, 6))

# Define the flower species to display
species_list = ["setosa", "versicolor", "virginica"]

# Plot each species using its petal measurements
for species in species_list:

    # Select the records belonging to the current species
    species_data = [
        (X[i][2], X[i][3])
        for i in range(len(X))
        if y[i] == species
    ]

    # Separate petal lengths and petal widths
    petal_lengths = [point[0] for point in species_data]
    petal_widths = [point[1] for point in species_data]

    # Display the points for this species
    plt.scatter(petal_lengths, petal_widths, label=species)

# Add labels and a title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Species: Petal Length vs Petal Width")

# Display the species legend
plt.legend()

# Display the grid
plt.grid(True)

# Adjust the graph layout
plt.tight_layout()

# Display the scatter plot
plt.show()

# Create a bar chart comparing actual and predicted species
import numpy as np

# Count the actual and predicted flowers in each species
species_names = ["setosa", "versicolor", "virginica"]

actual_counts = [
    y_test.count(species) for species in species_names
]

predicted_counts = [
    list(y_pred).count(species) for species in species_names
]

# Set the positions of the bars
x = np.arange(len(species_names))
width = 0.35

# Create the graph
plt.figure(figsize=(8, 6))

# Display actual and predicted counts
plt.bar(x - width/2, actual_counts, width, label="Actual")
plt.bar(x + width/2, predicted_counts, width, label="Predicted")

# Add labels and title
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")
plt.title("Actual vs Predicted Iris Species")

# Display species names and legend
plt.xticks(x, species_names)
plt.legend()

# Display the graph
plt.tight_layout()
plt.show()

# Save the trained model for the web application
import joblib

joblib.dump(model, "iris_model.pkl")

print("\nTrained model saved successfully as iris_model.pkl")