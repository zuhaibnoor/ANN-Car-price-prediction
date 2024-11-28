import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Read the data
df = pd.read_csv("car_data.csv")
df.dropna(inplace=True)
print("No of data rows:", df.shape[0])

# Normalization
scalar = MinMaxScaler()
df[["Kms_Driven", "Present_Price"]] = scalar.fit_transform(
    df[["Kms_Driven", "Present_Price"]]
)

# Standardization
standard_scalar = StandardScaler()
df[["Year"]] = standard_scalar.fit_transform(df[["Year"]])

# One hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
df = pd.concat(
    [
        df.drop(columns=["Fuel_Type", "Seller_Type", "Transmission"]),
        one_hot_encoder.fit_transform(df[["Fuel_Type", "Seller_Type", "Transmission"]]),
    ],
    axis=1,
)

# Binning
df["price_category"] = pd.qcut(df["Selling_Price"], q=3, labels=False)

# Drop processed categorical columns
x = df.drop(["Car_Name", "Selling_Price", "price_category"], axis=1)
y = df["price_category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("Number of training rows:", X_train.shape[0])
print("Number of testing rows:", X_test.shape[0])

# Train the model
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
mlp_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp_classifier.predict(X_test)
print("Hidden layer size:", mlp_classifier.hidden_layer_sizes)
print("Number of layers:", mlp_classifier.n_layers_)
print("Number of iterations:", mlp_classifier.n_iter_)
print("Classes:", mlp_classifier.classes_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
