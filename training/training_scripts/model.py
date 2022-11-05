import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    """
    Defines the pytorch models for the classifier using pretrained efficientnet_v2_s module
    """

    def __init__(self, params):
        super(Classifier, self).__init__()
        self.base_model = eval(
            "models.{}(weights='{}')".format(params["base_model"], params["pretrained"])
        )
        if not params["finetune"]:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            # nn.Linear(in_features=1280, out_features=512, bias=True),
            # nn.ReLU6(),
            # nn.Linear(in_features=1280, out_features=1280, bias=True),
            # nn.ReLU6(),
            # nn.Linear(in_features=512, out_features=512, bias=True),
            # nn.ReLU6(),
            # nn.Linear(in_features=512, out_features=256, bias=True),
            # nn.ReLU6(),
            nn.Linear(in_features=1280, out_features=params["n_classes"], bias=True),
        )

    def forward(self, x):
        return self.base_model(x)
