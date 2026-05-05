import numpy as np

# Normalize values between 0 and 1
def normalize(value, min_val=0, max_val=1):
    return (value - min_val) / (max_val - min_val + 1e-8)


# Safe division
def safe_divide(a, b):
    return a / b if b != 0 else 0


# Convert numpy to float
def to_float(value):
    try:
        return float(value)
    except:
        return 0.0


# Calculate variance safely
def compute_variance(data):
    try:
        return float(np.var(data))
    except:
        return 0.0


# Generate unique ID (optional)
def generate_id():
    import uuid
    return str(uuid.uuid4())