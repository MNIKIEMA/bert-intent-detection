import torch.nn as nn
import torch.optim as optim
import torch

def perf(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = num = correct = 0
    for x, mask, y in loader:
      x = x.to(device)
      y = y.to(device)
      mask = mask.to(device)
      with torch.no_grad():
        y_scores = model(x, mask)
        loss = criterion(y_scores, y)
        y_pred = torch.max(y_scores, 1)[1]
        correct += torch.sum(y_pred == y).item()
        total_loss += loss.item()
        num += len(y)
    return total_loss / num, correct / num



def fit(model,train_loader, valid_loader, epochs, device, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, mask, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            y_scores = model(x, mask)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        print(epoch, total_loss / num, *perf(model, valid_loader))