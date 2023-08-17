import torch
import torch.nn as nn
from torchvision import models
from model.minifasnet import MiniFASNetV2SE


class ResNet_x128(nn.Module):
    def __init__(self):
        super(ResNet_x128, self).__init__()
        self.resnet_model = models.alexnet(num_classes=128)
        self.score_layer = nn.Linear(128, 1)

    def forward(self, x):
        features = self.resnet_model(x)
        features = torch.sigmoid(features)
        score = self.score_layer(features)
        if self.training:
            # features = torch.sigmoid(features)
            return features, score
        else:
            return score


class EnsembleNet(nn.Module):
    def __init__(self):
        super(EnsembleNet, self).__init__()
        # self.vgg_model = models.vgg11_bn(num_classes=128)
        self.vgg_model = models.alexnet(num_classes=128)
        self.alexnet_model = models.alexnet(num_classes=128)
        self.alex_score_layer = nn.Linear(128, 1)
        self.vgg_score_layer = nn.Linear(128, 1)
        self.output_layer = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        vgg_features = self.vgg_model(x)
        alexnet_features = self.alexnet_model(x)
        vgg_features = self.tanh(vgg_features)
        alexnet_features = self.tanh(alexnet_features)
        vgg_score = self.vgg_score_layer(self.relu(vgg_features))
        alex_score = self.alex_score_layer(self.relu(alexnet_features))
        combined_score = torch.cat((vgg_score, alex_score), dim=1)
        score = self.output_layer(combined_score)
        # score = torch.sigmoid(score)
        if self.training:
            # vgg_features = torch.tanh(vgg_features)
            # alexnet_features = torch.tanh(alexnet_features)
            return vgg_features, alexnet_features, vgg_score, alex_score, score
        else:
            return score


class EnsembleNet_c3(nn.Module):
    def __init__(self):
        super(EnsembleNet_c3, self).__init__()
        self.vgg_model = models.alexnet(num_classes=64)
        # self.vgg_model = models.resnet18(num_classes=64)
        self.alexnet_model = models.alexnet(num_classes=64)
        self.alex_score_layer = nn.Linear(64, 1)
        self.vgg_score_layer = nn.Linear(64, 1)
        self.output_layer = nn.Linear(2, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        vgg_features = self.vgg_model(x[:, :3, :, :])
        alexnet_features = self.alexnet_model(x[:, :3, :, :])
        vgg_score = self.vgg_score_layer(self.relu(vgg_features))
        alex_score = self.alex_score_layer(self.relu(alexnet_features))
        combined_score = torch.cat((vgg_score, alex_score), dim=1)
        score = self.output_layer(combined_score)
        if self.training:
            vgg_features = self.tanh(vgg_features)
            alexnet_features = self.tanh(alexnet_features)
            return vgg_features, alexnet_features, vgg_score, alex_score, score
        else:
            return combined_score, score


class EnsembleNet_c4(nn.Module):
    def __init__(self):
        super(EnsembleNet_c4, self).__init__()
        self.vgg_model = models.alexnet(num_classes=128)
        self.alexnet_model = models.alexnet(num_classes=128)
        self.alex_score_layer = nn.Linear(128, 1)
        self.vgg_score_layer = nn.Linear(128, 1)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        vgg_features = self.vgg_model(x)
        alexnet_features = self.alexnet_model(x)
        vgg_features = torch.sigmoid(vgg_features)
        alexnet_features = torch.sigmoid(alexnet_features)
        vgg_score = self.vgg_score_layer(vgg_features)
        alex_score = self.alex_score_layer(alexnet_features)
        combined_score = torch.cat((vgg_score, alex_score), dim=1)
        score = self.output_layer(combined_score)
        if self.training:
            return vgg_features, alexnet_features, vgg_score, alex_score, score
        else:
            return score


class PixelNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.Linear(27 * 27, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(26 * 26, num_classes),
        )

    def forward(self, img):
        map = self.features(img)
        map = torch.sigmoid(map)
        # x = self.avgpool(map)
        x = torch.flatten(map, 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return map, x


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, img_channel=3, num_classes=1, embedding_size=128, conv6_kernel=(5, 5)):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = MiniFASNetV2SE(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                    num_classes=num_classes, img_channel=img_channel, drop_p=0.8)
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()
        # self.vgg_model = models.vgg11(num_classes=1, dropout=0.9)
        # self.alexnet_model = models.alexnet(num_classes=1, dropout=0.9)
        # self.drop = torch.nn.Dropout(p=0.2)
        # self.output_layer = nn.Linear(3, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # alex_o = self.alexnet_model(x)
        # vgg_o = self.vgg_model(x)
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x2 = self.model.drop(x1)
        cls = self.model.prob(x2)
        # cls = self.drop(torch.cat((alex_o, vgg_o, fas_o), dim=1))
        # cls = self.output_layer(cls)

        if self.training:
            f_m = self.FTGenerator(x)
            return cls, x1, f_m
        else:
            return cls
