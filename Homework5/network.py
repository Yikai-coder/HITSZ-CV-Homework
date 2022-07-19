import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self):
        """
        Init network.

        The structure of network:
            * 2D convolution: output feature channel number = 6, kernel size = 5x5, stride = 1, no padding;
            * 2D max pooling: kernel size = 2x2, stride = 2;
            * 2D convolution: output feature channel number = 16, kernel size = 5x5, stride = 1, no padding;
            * 2D max pooling: kernel size = 2x2, stride = 2;
            * Fully-connected layer: output feature channel number = 120;
            * Fully-connected layer: output feature channel number = 84;
            * Fully-connected layer: output feature channel number = 10 (number of classes).

        Hint:
            1. for 2D convolution, you can use `torch.nn.Conv2d`
            2. for 2D max pooling, you can use `torch.nn.MaxPool2d`
            3. for fully connected layer, you can use `torch.nn.Linear`
            4. Before the first fully connected layer, you should have a tensor with shape (BatchSize, 16, 5, 5),
               later in `forward()` you can flatten it to shape `(BatchSize, 400)`, 
               so the `input_feature` of the first connected layer is 400.
        """
        super().__init__()
        ### YOUR CODE HERE
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=400, out_features=120), 
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Linear(in_features=84, out_features=10)
        )
        ### END YOUR CODE

    def forward(self, x):
        """
        Forwrad process.

        Hint:
            Before the first fully connected layer, you should have a tensor with shape (BatchSize, 16, 5, 5),
            you can flatten the tensor to shape `(BatchSize, 400)` here, you may find `torch.flatten` helpful.
        """
        ### YOUR CODE HERE
        out = self.sequential(x)
        ### END YOUR CODE
        return out



class resNet(nn.Module):
    def __init__(self) -> None:
        super(resNet, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=False)
        in_feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feat, 10)
    
    def forward(self, x):
        out = self.resnet(x)
        return out
        

