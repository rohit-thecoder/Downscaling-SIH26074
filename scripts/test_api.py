import json
import urllib.request

import torch


# ============================================================
# LOAD ONE REAL TEST SAMPLE
# ============================================================

X_test = torch.load(
    "data/processed/tensors/X_test.pt",
    weights_only=False
)

# First unseen test timestamp
sample = X_test[0].tolist()


# ============================================================
# REQUEST
# ============================================================

payload = json.dumps({
    "data": sample
}).encode("utf-8")


request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=payload,
    headers={
        "Content-Type": "application/json"
    },
    method="POST"
)


# ============================================================
# CALL API
# ============================================================

with urllib.request.urlopen(request) as response:

    result = json.loads(
        response.read().decode("utf-8")
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("API MODEL TEST")
print("=" * 70)

print("\nOutput shape:")
print(result["shape"])

print("\nVariables:")
print(result["variables"])

print("\nResolution:")
print(result["resolution"])

print("\nPrediction generated successfully!")

print("\nExpected output:")
print("[1, 3, 61, 51]")