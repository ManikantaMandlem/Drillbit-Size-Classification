import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.io as io
import torchvision.models as models
from prettytable import PrettyTable
from functorch import combine_state_for_ensemble, vmap
from argparse import ArgumentParser


class Classifier(nn.Module):
    """
    Defines the pytorch models for the classifier using pretrained efficientnet_v2_s module
    """

    def __init__(self):
        super(Classifier, self).__init__()
        self.base_model = models.mobilenet_v2()
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=9, bias=True),
        )

    def forward(self, x):
        return self.base_model(x)


def main_script(folder_path):
    ckpt_path1 = "./ckpts/drillbit_classification-v013_iteration8632.ckpt"
    ckpt_path2 = "./ckpts/drillbit_classification-v013_iteration9140.ckpt"
    ckpt_path3 = "./ckpts/drillbit_classification-v013_iteration11172.ckpt"
    ckpt_path4 = "./ckpts/drillbit_classification-v013_iteration11426.ckpt"
    ckpt_path5 = "./ckpts/drillbit_classification-v017_iteration19554.ckpt"
    ckpt_path6 = "./ckpts/drillbit_classification-v017_iteration23618.ckpt"
    ckpt_path7 = "./ckpts/drillbit_classification-v017_iteration26412.ckpt"
    ckpt_path8 = "./ckpts/drillbit_classification-v017_iteration25142.ckpt"

    crop = {"top": 400, "left": 300, "height": 200, "width": 250}
    class_list = [
        "2.0 mm x 26 mm",
        "2.0 mm x 28 mm",
        "2.8 mm x 22 mm",
        "3.5 mm x 19 mm",
        "3.5 mm x 22 mm",
        "3.5 mm x 28 mm",
        "3.5 mm x 30 mm",
        "4.2 mm x 22 mm",
        "4.2 mm x 30 mm",
    ]

    # ckpt_obj = [torch.load(eval('ckpt_path{}'.format(i+1)), map_location=torch.device('cpu')) for i in range(8)]
    ckpt_obj = []
    for i in range(8):
        ckpt_path = eval("ckpt_path{}".format(i + 1))
        obj = torch.load(ckpt_path, map_location=torch.device("cpu"))
        ckpt_obj.append(obj)

    ckpts = [ckpt_obj[i]["model_state_dict"] for i in range(8)]

    models_ = [Classifier() for i in range(8)]
    for model, ckpt in zip(models_, ckpts):
        model.eval()
        model.load_state_dict(ckpt)

    models_, params, buffers = combine_state_for_ensemble(models_)

    class_map = {}
    for i, class_ in enumerate(class_list):
        class_map[i] = class_

    class_counts = {}
    for path_ in os.listdir(folder_path):
        image_path = os.path.join(folder_path, path_)
        image = io.read_image(image_path)
        image = image.float()
        image = transforms.functional.crop(
            image, crop["top"], crop["left"], crop["height"], crop["width"]
        )
        image = torch.unsqueeze(image, 0)
        logit = vmap(models_, in_dims=(0, 0, None))(params, buffers, image)
        probs = torch.nn.Softmax(dim=1)(torch.squeeze(logit))
        pred = torch.argmax(probs, dim=1)
        pred, _ = torch.mode(pred)
        print(path_, "--->", class_map[pred.item()])
        class_counts[class_map[pred.item()]] = (
            class_counts.get(class_map[pred.item()], 0) + 1
        )

    table = PrettyTable()
    for class_, count in class_counts.items():
        table.add_row([class_, count])
    print(table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", "-p", type=str, help="Provide path to the images folder"
    )
    # parser.add_argument("--ckpt_path", "-c", type=str, help="Provide path to the model checkpoints")
    args = parser.parse_args()
    main_script(args.image_path)
