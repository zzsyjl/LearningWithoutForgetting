import torch
import torch.nn as nn


from models.cifar_resnet import *

resnet_zoos = {"resnet18": resnet18,
               "resnet32": resnet32,
               "resnet34": resnet34,
               "resnet50": resnet50,
               "resnet101": resnet101,
               "resnet152": resnet152}


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight)
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        # nn.init.kaiming_normal_(m.weight)
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # resnet_34 = ResNet32()
        classes = args.model.base_class
        resnet_name = args.model.name
        resnet = resnet_zoos[resnet_name]()
        resnet.apply(kaiming_normal_init)
        self.fc_in_features = resnet.classifier.in_features

        resnet.classifier = nn.Identity()
        self.feature_extractor = resnet

        fc = nn.Linear(self.fc_in_features, classes)
        self.fcs = nn.ModuleList()
        self.fcs.append(fc)
        self.num_task = 1

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = x.view(x.size(0), -1)
        outputs = self.fcs[0](x)
        for i in range(1, self.num_task):
            output = self.fcs[i](x)
            outputs = torch.cat((outputs, output), dim=1)
        return outputs

    def increment_classes(self, new_classes):
        # adding n classes in the final fc layer
        print("# new classes: ", new_classes)
        new_fc = nn.Linear(self.fc_in_features, new_classes)
        kaiming_normal_init(new_fc)
        self.fcs.append(new_fc)
        self.num_task += 1

    def classify(self, x):
        _, pred = torch.max(self.forward(x), dim=1, keepdim=False)
        return pred


if __name__ == "__main__":
    m = Model(20)
    m.increment_classes(20)
    print(m)
