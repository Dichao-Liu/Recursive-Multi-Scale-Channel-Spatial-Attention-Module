from __future__ import print_function
import timm
from attention_module import *
from example_insert_into_state_of_the_art_PMG.model import *
from example_insert_into_state_of_the_art_PMG.Resnet import *
from timm.models.helpers import load_checkpoint

def load_model(model_name, attention, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200, attention)

    return net

def gem(x, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class Network_Warper(nn.Module):
    def __init__(self, layers_1, layers_2, feature_dim=2048, net_case="RMCSAM"):
        super().__init__()
        self.layers_1 = layers_1
        self.net_case = net_case
        if net_case == "baseline":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif net_case == "RMCSAM":
            self.RMCSAM = RMCSAM(feature_dim)
        else:
            print("unknown net_case")
        self.classifier = layers_2

    def forward(self, x):
        x = self.layers_1(x)
        if self.net_case == "baseline":
            x = self.pool(x).view(x.shape[0], -1)
        else:
            x = self.RMCSAM(x)
            x = gem(x).view(x.shape[0], -1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':

    model_name = "resnet50"
    model_ini = timm.create_model(model_name, pretrained=True)
    model_layers = list(model_ini.children())

    m1 = model_layers[:-2]
    m1 = nn.Sequential(*m1)

    m2 = model_layers[9]
    m2 = nn.Sequential(m2)

    model_ini = Network_Warper(m1, m2, net_case="RMCSAM")

    load_checkpoint(model_ini,
                    "<the path to the saved weights of the resnet50+RMCSAM pre-trained on the ImageNet dataset>.pth")
    model_layers = list(model_ini.children())
    model_layers_1 = model_layers[1]

    net = load_model(model_name='resnet50_pmg', attention=model_layers_1, pretrain=True, require_grad=True)

    random_bird_ims = torch.rand(8, 3, 448, 448)
    prediction_score1, prediction_score2, prediction_score3, prediction_concat = net(random_bird_ims)
    print(prediction_score1.shape)
    print(prediction_score2.shape)
    print(prediction_score3.shape)
    print(prediction_concat.shape)