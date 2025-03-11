
import torch
import torch.nn as nn
import torch.optim as optim
from model import SAGEConv,GraphNeuralNetwork,Cheb_ploynomial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=1e-4)

# Early stopping parameters
early_stopping = 10
best_params = None
best_val_opa = -1
best_val_at_epoch = -1
epochs = 2

# Training loop
for epoch in range(epochs):
    gnn_model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output, edge_weights = gnn_model(features, edge_index, edgenet_input, random_walk_embeddings)
    loss_train = criterion(output[train_ind], labels[train_ind])
    
    # Backward pass
    loss_train.backward()
    optimizer.step()
    
    # Calculate training OPA metric (assuming a function `calculate_opa` is defined)
    train_opa = calculate_opa(output[train_ind], labels[train_ind])
    
    # Validation step
    gnn_model.eval()
    with torch.no_grad():
        val_output, _ = gnn_model(features, edge_index, edgenet_input, random_walk_embeddings)
        val_loss = criterion(val_output[val_ind], labels[val_ind]).item()
        val_opa = calculate_opa(val_output[val_ind], labels[val_ind])

    # Check for improvement
    if val_opa > best_val_opa:
        best_val_opa = val_opa
        best_val_at_epoch = epoch
        best_params = {name: param.clone() for name, param in gnn_model.named_parameters()}
        print(f" * [@{epoch}] Validation (NEW BEST): {val_opa}")
    elif early_stopping > 0 and epoch - best_val_at_epoch >= early_stopping:
        print(f"[@{epoch}] Best accuracy was attained at epoch {best_val_at_epoch}. Stopping.")
        break

# Restore best parameters
print("Restoring parameters corresponding to the best validation OPA.")
if best_params is not None:
    for name, param in gnn_model.named_parameters():
        param.data.copy_(best_params[name])
