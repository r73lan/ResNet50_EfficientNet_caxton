# Training Results of ResNet50 and EfficientNet on the Caxton Dataset

This repository contains attempts to train the ResNet50 and EfficientNet on the Caxton dataset.

The dataset includes photographs taken during 3D printing and evaluations of four key printing parameters: nozzle movement speed, vertical nozzle offset, filament feed rate, and plastic melting temperature (feed_rate_class, z_offset_class, flow_rate_class, hotend_class).

Each parameter can be evaluated as 0 (too low), 1 (normal), or 2 (too high).

The dataset was downloaded from the official website (see the link below). Archives print0...print182 were downloaded, while print183...print191 contain defective images (according to the dataset authors).

All work was performed using Python.

The previous_work directory contains the initial training versions, maintaining the original dataset structure. The file caxton_dataset_with_resnet.ipynb shows training results on 13k images. train3.py is the first successful script for training on the entire dataset, but the results were quite unconvincing. Additionally, one epoch took about 12 hours. It was decided to abandon this script and change the dataset structure.

The final_train_and_predict directory contains the final training version. Before training, scripts were written to crop these photographs around the printer nozzle to a smaller size (first from 1920x1080 to 320x320, then from 320x320 to 224x224) and to combine all photographs into one directory. A single annotation file (.csv) is created.

**Article where the dataset was found**

Brion D. A. J., Pattinson S. W. Generalisable 3D Printing Error Detection and Correction via Multi-Head Neural Networks // Nature Communications. 2022. Vol. 13, No. 1. DOI: 10.1038/s41467-022-31985-y.

**Link to the repository by the dataset creators**

https://github.com/cam-cambridge/caxton

**Link to download the dataset**

https://www.repository.cam.ac.uk/handle/1810/339869
