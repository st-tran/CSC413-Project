# CaNetDa: Deep-Learning Approach to GeoGuessr on a Canada Dataset
In this paper, we look at pinpointing an image location in Canada. Previous papers such as DeepGeo uses convolutional neural networks to predict the location of a given picture in the United States. We explore how well these methods can be extended given a dataset of Canadian locations. We compare results of previously explored methods such as convolutional neural networks to newer image classification techniques such as the Vision Transformer model. Furthermore, we applied data-augmentations to the image to attempt to improve the performance. Due to time-constraints we were not able to fully train our networks but utilized transfer learning techniques instead. We found that our transfer-learned EfficientNet and ViT models work much better than our transfer-learned ResNet.

## Report
See `./report.pdf`

## Data and Models
All supporting data and models are available in this shared Google Drive folder; they can be run on Colab as-is since there is code to download the appropriate dependencies and data: [link](https://drive.google.com/drive/folders/198KNZXeaM06WJU45tJ-iQQ9pzcitSZsl?usp=sharing)

## References
See `./report.pdf`
