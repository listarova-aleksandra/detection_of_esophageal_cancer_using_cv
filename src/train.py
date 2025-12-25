import torch
from utils import compute_binary_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        all_labels.append(labels.detach())
        all_logits.append(logits.detach())

    epoch_loss = running_loss / len(loader.dataset)

    all_labels = torch.cat(all_labels).squeeze()
    all_logits = torch.cat(all_logits).squeeze()

    metrics = compute_binary_metrics(all_labels, all_logits)

    return epoch_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        all_labels.append(labels)
        all_logits.append(logits)

    epoch_loss = running_loss / len(loader.dataset)

    all_labels = torch.cat(all_labels).squeeze()
    all_logits = torch.cat(all_logits).squeeze()

    metrics = compute_binary_metrics(all_labels, all_logits)

    return epoch_loss, metrics