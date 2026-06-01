import pickle
import numpy as np

# Load saved model
model = pickle.load(open("churn_model.pkl", "rb"))

# Sample customer data
sample_data = np.array([[1,0,1,0,12,1,0,2,1,0,1,0,1,0,0,1,2,70.5,1200]])

# Predict
prediction = model.predict(sample_data)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("Customer may churn")
else:
    print("Customer will stay")