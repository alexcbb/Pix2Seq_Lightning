import timm
from torch import nn


class Encoder(nn.Module):
    def __init__(
            self, 
            model_name='deit3_small_patch16_384_in21ft1k', 
            pretrained=False, 
            out_dim=256
        ):
        super().__init__()
        self.model = timm.create_model(model_name, num_classes=0, global_pool='', pretrained=pretrained)
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])