import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net_DemAI(nn.Module):
    def __init__(self):
        super(Net_DemAI, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=(2, 2))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=(2, 2))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 1 is number of channel, convert from feature shape (784,) to 28x28x1
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(-1, 7 * 7 * 32)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x
        activation1 = F.relu(self.fc1(x))
        x = self.fc2(activation1)
        # return F.log_softmax(x, dim=1) #not equivalent to sparse_softmax_cross_entropy from tensorflow
        return x, activation1


class Net_DemAI_Client(nn.Module):
    def __init__(self):
        super(Net_DemAI_Client, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=(2, 2))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=(2,2))
        # self.pool2 = nn.MaxPool2d(2,  stride=2)
        # self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 1 is number of channel, convert from feature shape (784,) to 28x28x1
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        # x = self.conv2(x)
        # x = nn.ReLU()(x)
        # x = self.pool2(x)
        # x = self.dropout2(x)
        x = x.view(-1, 14 * 14 * 32)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x
        activation3 = F.relu(self.fc1(x))
        x = self.fc2(activation3)
        # return F.log_softmax(x, dim=1) #not equivalent to sparse_softmax_cross_entropy from tensorflow
        return x, activation3


# activation = {}
# def get_activation(name):
#
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs


class DNN(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']
                            ]

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class DNN2(nn.Module):
    def __init__(self, input_dim=784, mid_dim_in=100, mid_dim_out=100, output_dim=10):
        super(DNN2, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim_in)
        self.fc2 = nn.Linear(mid_dim_in, mid_dim_out)
        self.fc3 = nn.Linear(mid_dim_out, output_dim)

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


#################################
##### Neural Network model #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)

        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        activation2 = F.relu(self.fc2(x))
        x = self.fc3(activation2)
        return x, activation2


class CNNCifar_Server(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar_Server, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],

                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16*16*16
        x = self.pool(F.relu(self.conv2(x)))  # 8*8*32
        x = self.pool(F.relu(self.conv3(x)))  # 4*4*64
        x = self.pool(F.relu(self.conv4(x)))  # 2*2*128
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        activation2 = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(activation2)
        return x, activation2


class CNNCifar_Server_3layer(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar_Server_3layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],

                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16*16*16
        x = self.pool(F.relu(self.conv2(x)))  # 8*8*32
        x = self.pool(F.relu(self.conv3(x)))  # 4*4*64
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        activation2 = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(activation2)
        return x, activation2


class CNNCifar_Server_4layer(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar_Server_4layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv4.weight', 'conv4.bias'],
                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16*16*16
        x = self.pool(F.relu(self.conv2(x)))  # 8*8*32
        x = self.pool(F.relu(self.conv3(x)))  # 4*4*64
        x = self.pool(F.relu(self.conv4(x)))  # 2*2*128
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        activation2 = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(activation2)
        return x, activation2


class CNNCifar100(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        activation2 = self.drop((F.relu(self.fc2(x))))
        x = self.fc3(activation2)
        return x, activation2


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias']]

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
