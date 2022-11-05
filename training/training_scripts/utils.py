import yaml
import pandas as pd
import random
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
import os


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config


def input_to_generators(csv_path, val_split):
    meta_df = pd.read_csv(csv_path)
    paths = list(meta_df["path"])
    image_ids = list(meta_df["image_id"])
    labels = list(meta_df["label"])
    comp = list(zip(paths, image_ids, labels))
    random.shuffle(comp)
    paths, image_ids, labels = zip(*comp)
    path_dict = {}
    label_dict = {}
    for image_id, path, label in zip(image_ids, paths, labels):
        path_dict[image_id] = path
        label_dict[image_id] = label
    partitions = {}
    partitions["train"] = image_ids[: int(len(image_ids) * (1 - val_split))]
    partitions["validation"] = image_ids[int(len(image_ids) * (1 - val_split)) :]
    return path_dict, partitions, label_dict


class earlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class clearmlLogger:
    def __init__(self, params, api_key_path, project_name, task_name, tags):
        self.params = params
        self.api_key_path = api_key_path
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags
        self.logger = self.get_logger()

    def get_logger(self):
        # with open(self.api_key_path, 'r') as f:
        #     _clearml_keys = yaml.safe_load(f)
        # for k, v in _clearml_keys.items():
        #     os.environ[k] = v
        task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            reuse_last_task_id=False,
            tags=self.tags,
        )
        task.connect(self.params)
        writer = SummaryWriter("Neocis Assessment Training")
        return writer
