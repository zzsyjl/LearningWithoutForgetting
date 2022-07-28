import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, Normalize, ToTensor
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter

# from utils.data_loader import BatchData
from utils import BatchData
from utils import save_previous_data
from trainer import train
from trainer import accuracy


# Arguments
parser = argparse.ArgumentParser(description="Class Incremental Learning")
# Load the config file
parser.add_argument("--config", type=str, default="./configs/lwf.yaml")
flags = parser.parse_args()
args_dict = OmegaConf.load(flags.config)


# Args -- dataset
if args_dict.dataset.name == "cifar100":
    # from utils.cifar100 import Cifar100 as Data_set
    from utils import Cifar100 as Data_set
else:
    raise NotImplementedError

# Args -- model
if args_dict.model.name.startswith("resnet"):
    from models.lwf_resnet import Model
else:
    raise NotImplementedError


def main(args):
    # Prepare the dataset
    data_set = Data_set(10)
    transforms_train = Compose([RandomHorizontalFlip(), RandomCrop(args.dataset.image_size, padding=4),
                                ToTensor(), Normalize(args.dataset.mean, args.dataset.std)])
    transforms_test = Compose([ToTensor(), Normalize(args.dataset.mean, args.dataset.std)])

    # The number of the task
    num_iter = data_set.num_task

    # Initialize the model
    model = Model(args)

    # Initialize the tensorboard writer
    writer = SummaryWriter("./runs/CIFAR100_LWF/kaiming_uniform")

    # print(model.feature_extractor)

    # Choose the device
    device = torch.device(args.device)

    accuracy_per_task = np.zeros((num_iter, num_iter))
    average_accuracy = np.zeros(num_iter)

    test_loader_list = []

    for step in range(num_iter):
        print(f"{'-' * 30} Task {step + 1} Training Begin {'-' * 30}\n")

        train_set, test_set = data_set.get_next_classes(step)
        print(f"The number of train set: {len(train_set)}. The number of test set: {len(test_set)}\n")

        train_x, train_y = zip(*train_set)
        test_x, test_y = zip(*test_set)

        train_loader = DataLoader(BatchData(train_x, train_y, transforms_train),
                                  batch_size=args.train.batch_size, shuffle=True)
        test_loader = DataLoader(BatchData(test_x, test_y, transforms_test),
                                 batch_size=args.eval.batch_size, shuffle=False)
        
        test_loader_list.append(test_loader)

        train(model, train_loader, step, device, args)

        accumulate_correct, accumulate_total = 0, 0
        for i in range(len(test_loader_list)):
            correct_each_task, total_each_task = accuracy(model, test_loader_list[i], device)

            accuracy_per_task[step, i] = correct_each_task / total_each_task
            accumulate_correct += correct_each_task
            accumulate_total += total_each_task

        average_accuracy[step] = accumulate_correct / accumulate_total
        print(f"\nThe accuracy of the 1st task is: {(100 * accuracy_per_task[step, 0]):>0.4f}%")
        print(f"The accuracy of current task {step + 1} is: {(100 * accuracy_per_task[step, step]):>0.4f}%")
        print(f"The average accuracy is: {(100 * average_accuracy[step]):>0.4f}%")
        print(f"\n{'-' * 30} Task {step + 1} Training End {'-' * 30}\n\n")

        writer.add_histogram("Distribution of 1st conv weight in layer1", model.feature_extractor.layer1[0].conv1.weight.flatten(), step)
        writer.add_histogram("Distribution of 1st conv weight in layer2", model.feature_extractor.layer2[0].conv1.weight.flatten(), step)
        writer.add_histogram("Distribution of 1st conv weight in layer3", model.feature_extractor.layer3[0].conv1.weight.flatten(), step)

    writer.close()

    np.save("./Records/accuracy_per_task_3.npy", accuracy_per_task)
    np.save("./Records/average_accuracy_3.npy", average_accuracy)

    np.savetxt("./Records/accuracy_per_task_3.csv", accuracy_per_task, delimiter=',')
    np.savetxt("./Records/average_accuracy_3.csv", average_accuracy, delimiter=',')


if __name__ == "__main__":
    main(args_dict)
