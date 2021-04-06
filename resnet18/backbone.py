import torch
import torch.nn as nn
import torchvision

# code taken from https://github.com/tankche1/IDeMe-Net/blob/master/classification.py
class ClassificationNetwork(nn.Module):
    def __init__(self, params, dense=False):
        super(ClassificationNetwork, self).__init__()
        self.convnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.convnet.fc.in_features
        self.device = params.device
        self.dataset = params.dataset
        self.convnet.fc = nn.Linear(num_ftrs, params.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        if dense:
            self.conv1 = self.convnet.conv1
            self.bn1 = self.convnet.bn1
            self.relu = self.convnet.relu
            self.maxpool = self.convnet.maxpool
            self.layer1 = self.convnet.layer1
            self.layer2 = self.convnet.layer2
            self.layer3 = self.convnet.layer3
            self.layer4 = self.convnet.layer4
            self.avgpool = self.convnet.avgpool

    def forward(self, inputs):
        outputs = self.convnet(inputs)
        return outputs

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 100
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            #-----------Problem in training for CUB database
            #getting from 0-198 labels, converting them to 0-99
            if self.dataset == "CUB":
                y = y/2
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.forward(x)
            #print(outputs.shape)
            loss = self.criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))


# resnet18 without fc layer, used for testing to extract features
class EmbeddingNet(nn.Module):
    def __init__(self, params, dense=False):
        super(EmbeddingNet, self).__init__()
        self.resnet = ClassificationNetwork(params, dense=dense)
        ckpt = torch.load(params.file_path)
        self.resnet.load_state_dict(ckpt['model_state_dict'])

        self.conv1 = self.resnet.convnet.conv1
        self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
        self.bn1 = self.resnet.convnet.bn1
        self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
        self.relu = self.resnet.convnet.relu
        self.maxpool = self.resnet.convnet.maxpool
        self.layer1 = self.resnet.convnet.layer1
        self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
        self.layer2 = self.resnet.convnet.layer2
        self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
        self.layer3 = self.resnet.convnet.layer3
        self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.avgpool = self.resnet.convnet.avgpool

    def forward(self, x, feat_tensor = False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x) # (, 64L, 56L, 56L)
        layer2 = self.layer2(layer1) # (, 128L, 28L, 28L)
        layer3 = self.layer3(layer2) # (, 256L, 14L, 14L)
        layer4 = self.layer4(layer3) # (,512,7,7)
        if feat_tensor:
            return layer4
        else:
            x = self.avgpool(layer4)  # (,512,1,1)
            x = x.view(x.size(0), -1)
        return x