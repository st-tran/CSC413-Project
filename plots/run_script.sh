#!/bin/bash

python3 gen_plots.py centercrop/loss "ResNet (cropped images)"
python3 gen_plots.py concat/loss "ResNet (concatenated images)"
python3 gen_plots.py efficientnetconcat/loss "EfficientNet (concatenated images)"
python3 gen_plots.py efficientnetcrop/loss "EfficientNet (cropped images)"
python3 gen_plots.py efficientnetnocrop/loss "EfficientNet (unaugmented images)"
python3 gen_plots.py resnetnocrop/loss "ResNet (unaugmented images)"
python3 gen_plots.py transformernocrop/loss "VIT (unaugmented images)"
python3 gen_plots.py transformercrop/loss "VIT (cropped images)"

