from attention_module import *
import timm


def gem(x, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class Network_Warper(nn.Module):

    def __init__(self, layers_1, feature_dim=2048, num_class=200, net_case="RMCSAM"):
        super().__init__()
        self.layers_1 = layers_1
        self.net_case = net_case
        self.layers_2 = nn.Linear(feature_dim, num_class)

        if net_case == "baseline":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif net_case == "RMCSAM":
            self.RMCSAM = RMCSAM(feature_dim)
        else:
            print("unknown net_case")


    def forward(self, x):
        x = self.layers_1(x)
        if self.net_case == "baseline":
            x = self.pool(x).view(x.shape[0], -1)
        elif self.net_case == "RMCSAM":
            x = self.RMCSAM(x)
            x = gem(x).view(x.shape[0], -1)
        else:
            print("unknown net_case")

        x = self.layers_2(x)

        return x


if __name__ == '__main__':
    model = timm.create_model('resnet50', pretrained=True)
    model_layers = list(model.children())
    m = model_layers[:-2]
    m = nn.Sequential(*m)
    model = Network_Warper(m, feature_dim=2048, num_class=200, net_case="RMCSAM")
    random_bird_im = torch.rand(1, 3, 448, 448)
    prediction_score = model(random_bird_im)
    print(prediction_score.shape)
