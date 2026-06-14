# Step 1 - Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Step 2 - Load the data
df = pd.read_csv('data.csv')
print("Data loaded successfully!")
print(df)

# Step 3 - Separate features and target
X = df[['size', 'bedrooms', 'age']]  # inputs the model learns from
y = df['price']                       # what we want to predict

# Step 4 - Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train)} houses, testing on {len(X_test)} houses")

# Step 5 - Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Step 6 - Test the model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Average prediction error: Rs {error:,.0f}")

# Step 7 - Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as model.pkl")
