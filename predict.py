import pickle

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded!")
print("Enter house details to predict price\n")

# Take input from user
size = int(input("Enter house size (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter age of house (years): "))

# Predict
predicted_price = model.predict([[size, bedrooms, age]])

print(f"\nPredicted House Price: Rs {predicted_price[0]:,.0f}")
