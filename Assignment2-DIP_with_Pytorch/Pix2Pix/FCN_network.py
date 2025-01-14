import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # 8 -> 16 channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32 -> 16 channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 16 -> 8 channels
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # 8 -> 3 channels (RGB)
            nn.Tanh()  # 使用Tanh激活函数将输出限制在[-1,1]范围内
        )

    def forward(self, x):
        # Encoder forward pass
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        # Decoder forward pass
        deconv1_out = self.deconv1(conv4_out)
        deconv2_out = self.deconv2(deconv1_out)
        deconv3_out = self.deconv3(deconv2_out)
        
        ### FILL: encoder-decoder forward pass

        output = self.deconv4(deconv3_out)
        
        return output
    
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        # Load VGG16 pretrained model
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        self.features = nn.Sequential(*features)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)
        self.upscore8 = nn.Sequential(nn.ConvTranspose2d(num_classes, 3, kernel_size=14, stride=8, padding=3, bias=False),
                                     nn.Tanh())  # Output size: (N, 3, 256, 256)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        # Input size: (1, 3, 256, 256)
        pool3 = None
        pool4 = None
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i == 16:  # pool3 layer
                pool3 = x  # Size after pool3: (1, 256, 32, 32)
            elif i == 23:  # pool4 layer
                pool4 = x  # Size after pool4: (1, 512, 16, 16)

        x = F.relu(self.fc6(x))  # Size after fc6: (1, 4096, 8, 8)

        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))  # Size after fc7: (1, 4096, 8, 8)
        x = F.dropout(x, 0.5, self.training)
        x = self.score_fr(x)  # Size after score_fr: (1, num_classes, 8, 8)

        x = self.upscore2(x)  # Size after upscore2: (1, num_classes, 16, 16)

        x = x + self.score_pool4(pool4)  # Size after adding score_pool4: (1, num_classes, 16, 16)

        x = self.upscore_pool4(x)  # Size after upscore_pool4: (1, num_classes, 32, 32)

        x = x + self.score_pool3(pool3)  # Size after adding score_pool3: (1, num_classes, 32, 32)
 
        x = self.upscore8(x)  # Size after upscore8: (1, num_classes, 256, 256)

        return x