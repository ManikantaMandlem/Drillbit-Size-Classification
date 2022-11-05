import cv2
import yaml
import os
from tqdm import tqdm
import pandas as pd

# class to process videos and save as images
class videoProcessing:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.video_path = config["input_path"]
        self.output_path = config["output_path"]
        self.meta_file_path = config["meta_file_path"]
        self.meta_data = {}
        self.meta_data["image_id"] = []
        self.meta_data["path"] = []
        self.meta_data["label"] = []
        self._extractFrames()
        self._createMetafile()

    def _extractFrames(self):
        """
        Function that will extract image frames from videos and store them in appropriate folders
        Also creates csv file with meta data about the images created
        Input: None
        Return: None
        """
        with tqdm(
            desc="Extracting frames as images from videos",
            total=len(os.listdir(self.video_path)),
        ) as pbar:
            for video in os.listdir(self.video_path):
                video_name = video.replace(".mp4", "")
                vid_obj = cv2.VideoCapture(os.path.join(self.video_path, video))
                # creating necessary folder structure to save images
                curr_dir = os.path.join(self.output_path, video_name)
                if not os.path.exists(curr_dir):
                    os.makedirs(curr_dir)
                frame_count = 1
                success, frame = vid_obj.read()
                # for each video, loop until all frames are covered, save the images into designated folders
                while success:
                    image_id = video_name + "_{}".format(frame_count)
                    image_path = os.path.join(curr_dir, image_id + ".png")
                    cv2.imwrite(image_path, frame)
                    # store the meta data of current image into the dictionary
                    self.meta_data["image_id"].append(image_id)
                    self.meta_data["path"].append(image_path)
                    self.meta_data["label"].append(
                        image_id[0]
                        + "."
                        + image_id[1]
                        + " mm x "
                        + image_id[3:5]
                        + " mm"
                    )
                    # read next image frame
                    success, frame = vid_obj.read()
                    frame_count += 1
                pbar.update(1)

    def _createMetafile(self):
        """
        Helper function to save the meta data dictionary into csv file
        """
        meta_df = pd.DataFrame.from_dict(self.meta_data)
        meta_df.to_csv(self.meta_file_path, index=False)


if __name__ == "__main__":
    config_path = "/home/mmandlem/Neocis_assessment/video_processing/video_config.yaml"
    videoProcessing(config_path)
