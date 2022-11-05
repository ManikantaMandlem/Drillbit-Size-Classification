# Drillbit-Size-Classification
To Classify images of drillbits attached to robotic arm based on these dimensions

## Testing:
- Test scripts are located in Inference folder
- All the necessary packages are included in requirements.txt file
- Below command can be used to setup the environment for testing
```
pip install -r test_requirements.txt
```
- Then go to inference folder and run the below command. Please include the path to folder that consists of testing images using -p flag as shown below
```
python inference.py -p <path/to/image/folder>
```

## Training:
- Training scripts are present in training folder
- To replicate training, below command can be used to the environment first
```
pip install -r train_requirements.txt
```
- Then, a configuration file with all necessary parameters needs to be saved in the train_config folder. (One of the templates can be used to create one)
- Below command can be used to start the training process
```
python training/run_scripts/train_run.py -p <path/to/config/file>
```

- For more details please refer to report.pdf
