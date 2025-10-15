from tensorflow import keras

# Load your existing .h5 model (from TF 2.13)
model = keras.models.load_model("models/model.h5", compile=False)

# Save in the new .keras format, disabling strict checks
model.save("models/model.keras", save_format="keras", safe_mode=False)

print("✅ Model reconverted successfully.")