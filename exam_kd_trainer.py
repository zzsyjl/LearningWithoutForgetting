import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from utils import distillation_loss, DMER, distillation_bce_loss


def train(model, data_loader, step, device, args, previous_data_loader=None):
    pre_model = None
    # If step > 0, then new output nodes need to be added
    if step > 0:
        # If args.method is "lwf", then the model need to be copied before increasing classes
        if args.method == "lwf":
            pre_model = copy.deepcopy(model)

        # Add new output nodes
        model.increment_classes(args.model.incremental_class)

    # Move the model to the chosen device
    model.to(device)

    # Loss function for new classes
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss(reduction="none")
    
    optimizer = optim.SGD(model.parameters(),
                          lr=args.train.base_lr, momentum=args.train.optimizer.momentum,
                          weight_decay=args.train.optimizer.weight_decay)
    # LR scheduler
    scheduler = MultiStepLR(optimizer, args.train.scheduler_epochs, gamma=args.train.scheduler_gamma)

    for epoch in range(args.train.num_epochs):
        if step == 0:
            training_loss = train_epoch_for_base(model, data_loader, optimizer, bce, device)
        else:
            dl = DMER if args.distillation.use_mer else distillation_bce_loss
            training_loss = train_epoch_for_new(model, pre_model, data_loader, previous_data_loader, optimizer, bce, dl,
                                                step, device, epoch + 1, args)
        scheduler.step()
        print(f"[{(epoch + 1):>3d} / {args.train.num_epochs}], \tLoss: {training_loss:>0.4f}")


def accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze()
            probs, predictions = torch.max(torch.sigmoid(model(images)), dim=-1, keepdim=False)
            correct += (predictions == labels).type(torch.float).sum().item()
            total += images.shape[0]

    return correct, total


def train_epoch_for_base(model, data_loader, optimizer, loss1, device):
    model.train()
    running_loss = 0.0
    for data in data_loader:
        # images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        images, labels = data[0].to(device), data[1].to(device)
        labels = labels.squeeze()
        labels = F.one_hot(labels, num_classes=10).float()

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        loss = loss1(outputs, labels).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(data_loader)


def train_epoch_for_new(model, pre_model, data_loader, previous_data_loader, optimizer, loss1, loss2, step, device, epoch, args):
    model.train()
    running_loss = 0.0
    for data, previous_data in zip(data_loader, previous_data_loader):
        images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        labels = labels.squeeze()
        labels = F.one_hot(labels, num_classes=10 * (step + 1)).float()

        previous_images, previous_labels = previous_data[0].to(device), previous_data[1].to(device, dtype=torch.int64)
        previous_labels = previous_labels.squeeze()
        previous_labels = F.one_hot(previous_labels, num_classes=10 * step).float()

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        previous_outputs = model(previous_images)
        previous_outputs = torch.sigmoid(previous_outputs)

        # Compute the loss
        with torch.no_grad():
            pre_model.eval()
            soft_targets = pre_model(previous_images)
            soft_targets = torch.sigmoid(soft_targets)

        logit_dist = previous_outputs[:, :-args.model.incremental_class]
        # ce_loss = loss1(outputs[:, -args.model.incremental_class:], labels - args.model.incremental_class * step)
        # ce_loss = loss1(outputs[:, -args.model.incremental_class:], labels[:, -args.model.incremental_class:]).sum(dim=-1)
        ce_loss = loss1(outputs, labels).sum(dim=-1)
        mask = ce_loss <= 1/epoch
        ce_loss = (mask * (ce_loss - 1/epoch)).mean()

        dist_loss = loss2(logit_dist, soft_targets, args.distillation.temperature)
        loss = args.distillation.alpha * dist_loss + ce_loss
    
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(data_loader)
