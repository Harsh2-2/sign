import pickle

with open("asl_mediapipe_rf.pkl", "rb") as f:
    data = pickle.load(f)

# Handle dictionary or direct estimator format
model = data["model"] if isinstance(data, dict) and "model" in data else data

print("=" * 45)
print(f"Loaded Model Type: {type(model)}")

# Check feature attributes in scikit-learn models
if hasattr(model, "n_features_in_"):
    print(f"✓ EXPECTED FEATURES: {model.n_features_in_}")
elif hasattr(model, "input_shape"):
    print(f"✓ EXPECTED INPUT SHAPE: {model.input_shape}")
else:
    print("Could not auto-detect features. Printing model attributes:")
    print(dir(model))

print("=" * 45)