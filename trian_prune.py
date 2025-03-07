import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from GNN_structure import GNNModel_1layer,GNNModel_2layer,GNNModel_4layer,GNNModel_3layer
from data_loader import MultiGraphDataset
from loss import ConnectionLoss

def train_model(train_loader, model, optimizer, criterion, epochs=15,save_path=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.edge_label.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":

    dataset_dir = ""
    save_path = ""
    random_data = True #50%
    dataset = MultiGraphDataset(dataset_dir,random_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GNNModel_2layer()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = ConnectionLoss()

    train_model(loader, model, optimizer, criterion, epochs=15,save_path=save_path)

