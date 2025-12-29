"""
Modal GPU test - PyTorch CUDA check and simple logistic regression
"""

import modal

app = modal.App("test-pytorch-gpu")

# Image with PyTorch and CUDA
gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
    "scikit-learn",
)


@app.function(image=gpu_image, gpu="T4")
def check_cuda():
    """Check if CUDA is available."""
    import torch
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    return info


@app.function(image=gpu_image, gpu="T4")
def train_logistic_regression(n_samples: int = 10000, n_features: int = 100):
    """Train a simple logistic regression on GPU."""
    import torch
    import torch.nn as nn
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    w_true = np.random.randn(n_features).astype(np.float32)
    y = (X @ w_true + np.random.randn(n_samples) * 0.1 > 0).astype(np.float32)
    
    # Convert to tensors
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    
    # Simple logistic regression model
    model = nn.Sequential(
        nn.Linear(n_features, 1),
        nn.Sigmoid()
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(X_t).squeeze()
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Final accuracy
    with torch.no_grad():
        preds = (model(X_t).squeeze() > 0.5).float()
        accuracy = (preds == y_t).float().mean().item()
    
    return {
        "device": str(device),
        "final_loss": loss.item(),
        "accuracy": accuracy,
        "n_samples": n_samples,
        "n_features": n_features,
    }


@app.local_entrypoint()
def main():
    print("="*50)
    print("MODAL GPU TEST")
    print("="*50)
    
    print("\n1. Checking CUDA availability...")
    cuda_info = check_cuda.remote()
    for k, v in cuda_info.items():
        print(f"   {k}: {v}")
    
    print("\n2. Training logistic regression on GPU...")
    result = train_logistic_regression.remote(n_samples=50000, n_features=200)
    print(f"   Device: {result['device']}")
    print(f"   Final loss: {result['final_loss']:.4f}")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("SUCCESS!")
    print("="*50)
