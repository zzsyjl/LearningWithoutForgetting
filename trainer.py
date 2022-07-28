import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# from utils.distillation_loss import distillation_loss
from utils import distillation_loss, DMER


def train(model, data_loader, step, device, args):
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

    # Optimizer in different method
    if args.method == "ft":
        current_num_tasks = model.num_task
        # Freeze the parameters in classifier for previous tasks
        for i in range(current_num_tasks - 1):
            for param in model.fcs[i].parameters():
                param.requires_grad = False

        if args.train.optimizer.name == "sgd":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.train.base_lr, momentum=args.train.optimizer.momentum,
                                  weight_decay=args.train.optimizer.weight_decay)
        else:
            raise NotImplementedError
    elif args.method == "lwf":
        if args.train.optimizer.name == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.train.base_lr, momentum=args.train.optimizer.momentum,
                                  weight_decay=args.train.optimizer.weight_decay)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # LR scheduler
    scheduler = MultiStepLR(optimizer, args.train.scheduler_epochs, gamma=args.train.scheduler_gamma)

    for epoch in range(args.train.num_epochs):
        training_loss = train_epoch(model, pre_model, data_loader, optimizer, ce, DMER,
                                    step, device, args)
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
            predictions = model.classify(images)
            correct += (predictions == labels).type(torch.float).sum().item()
            total += images.shape[0]

    # acc = correct / total
    # return acc
    return correct, total


def train_epoch(model, pre_model, data_loader, optimizer, loss1, loss2, step, device, args):
    model.train()
    running_loss = 0.0
    for data in data_loader:
        images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        labels = labels.squeeze()
        optimizer.zero_grad()
        outputs = model(images)

        # Compute the loss
        if step == 0 or args.method == "ft":
            loss = loss1(outputs, labels)
        elif args.method == "lwf":
            with torch.no_grad():
                pre_model.eval()
                soft_targets = pre_model(images)

            logit_dist = outputs[:, :-args.model.incremental_class]
            ce_loss = loss1(outputs[:, -args.model.incremental_class:], labels - args.model.incremental_class * step)
            dist_loss = loss2(logit_dist, soft_targets, args.distillation.temperature)
            loss = args.distillation.alpha * dist_loss + ce_loss
        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(data_loader)
