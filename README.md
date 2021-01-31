# The Bredele hackathon
  by dotnetmobile@gmail.com
---

## Proof of concept based on a [Raspberry pi 4 (8GB RAM)](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) and a [Coral USB device](https://coral.ai/products/accelerator) for detecting [Bredele](https://en.wikipedia.org/wiki/Bredele) using the [TensorFlow Machine Learning platform version 2](https://www.tensorflow.org) and the [Faster RCNN Inception model](https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1).

All credits go to the authors (@EdjeElectronics + @tensorflow) of the following remarkable references:

* [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
* [How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow (GPU) on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
* [TensorFlow](https://github.com/tensorflow)
* [TensorFlow models](https://github.com/tensorflow/models)
* [TensorFlow model configurations](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

---
## Context

I successfully deployed on my Raspberry Pi all the installation steps of the excellent tutorial [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi).<br>
I noticed that the objects detection trained model was time to time mismatching objects like apples, oranges and balls.<br>
Therefore I decided to retrain the model in order to see if I can get a better object detection.<br>
As it was the Christmas period and we cooked quite a lot of Bredele, I trained the Faster RCNN Inception model using our own cookies.
It was a good starting point for using TensorFlow :wink:

---

## Objects

The trained model should be able to detect 16 different object types:

|  Id  | Name                                 | Comment                      |
|:----:|--------------------------------------|------------------------------|
|1     | <img title="Christmas tree" src="./images-small/training/bredelehackathon_88_of_511.jpeg" width="50" height="50"/>  | |
|2     | <img title="flake" src="./images-small/training/bredelehackathon_137_of_511.jpeg" width="50" height="50" /> | |
|3     | <img title="heart #1" src="./images-small/training/bredelehackathon_385_of_511.jpeg" width="50" height="50" /> | commercial |
|3     | <img title="heart #2" src="./images-small/training/bredelehackathon_153_of_511.jpeg" width="50" height="50" /> | |
|4     | <img title="king mage" src="./images-small/training/bredelehackathon_136_of_511.jpeg" width="50" height="50" /> | |
|5     | <img title="lamb" src="./images-small/training/bredelehackathon_157_of_511.jpeg" width="50" height="50" /> | |
|6     | <img title="Santa Claus #1" src="./images-small/training/bredelehackathon_140_of_511.jpeg" width="50" height="50" /> | |
|6     | <img title="Santa Claus #2" src="./images-small/training/bredelehackathon_133_of_511.jpeg" width="50" height="50" /> | |
|7     | <img title="snowball" src="./images-small/training/bredelehackathon_294_of_511.jpeg" width="50" height="50" /> | |
|8     | <img title="snowman" src="./images-small/training/bredelehackathon_132_of_511.jpeg" width="50" height="50" /> | |
|9     | <img title="shooting star" src="./images-small/training/bredelehackathon_156_of_511.jpeg" width="50" height="50" /> | |
|10    | <img title="squirrel" src="./images-small/training/bredelehackathon_151_of_511.jpeg" width="50" height="50" /> | |
|11    | <img title="star #1" src="./images-small/training/bredelehackathon_383_of_511.jpeg" width="50" height="50" /> | commercial |
|11    | <img title="star #2" src="./images-small/training/bredelehackathon_165_of_511.jpeg" width="50" height="50" /> | |
|12    | <img title="weird #1" src="./images-small/training/bredelehackathon_406_of_511.jpeg" width="50" height="50" /> | forgot baker's yeast :joy: |
|13    | weird #2 | forgot baker's yeast :joy: |
|14    | <img title="Christmas shortbread with jam" src="./images-small/training/bredelehackathon_290_of_511.jpeg" width="50" height="50" /> | |
|15    | <img title="angel" src="./images-small/training/bredelehackathon_148_of_511.jpeg" width="50" height="50" /> | |
|16    | <img title="bretzel" src="./images-small/training/bredelehackathon_384_of_511.jpeg" width="50" height="50" /> | commercial |

---

## Labels

| labels collection |
| ------------------- |
| <img title="weird #1, Christmas shortbread with jam" src="./doc/bredekehackathon_405_of_511.png" /> |
| <img title="Santa Claus, shooting star, heart, squirrel, lamb, star" src="./doc/bredelehackathon_22_of_511.png" /> |
| <img title="heart, star" src="./doc/bredelehackathon_77_of_511.png" /> |
| <img title="star, Christmas shortbread with jam" src="./doc/bredelehackathon_287_of_511.png" /> |
| <img title="Christmas tree, star, snowball" src="./doc/bredelehackathon_327_of_511.png" /> |
| <img title="heart, Christmas shortbread with jam, snowball" src="./doc/bredelehackathon_352_of_511.png" /> |
| <img title="Christmas shortbread with jam, snowball, bretzel, heart, star" src="./doc/bredelehackathon_370_of_511.png" /> |
| <img title="star, bretzel, heart" src="./doc/bredelehackathon_386_of_511.png" /> |


---
## Environment

* OS: [MacOS Big Sur](https://www.apple.com/uk/macos/big-sur/)
* [Anaconda 1.10.0](https://www.anaconda.com) with Python 3.8
* [TensorFlow Object Detection Git repository](https://github.com/tensorflow/models)
* [LabelImg](https://github.com/tzutalin/labelImg) for objects annotation
* [EdgeElectronics git repository](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/archive) instructions for training the model

## Setup the bredele hackathon environment

Follow instructions described in [INSTALL.md](https://github.com/dotnetmobile/bredele-hackathon/blob/main/INSTALL.md)

Please note that all bredele images used for the training and the testing have the size of 756x1008 (width x heigh). <br>
So the original <br>
```
faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.config
```
has been cloned in <br>
```
faster_rcnn_inception_resnet_v2_756x1008_coco17_tpu-8.config
```
and adapted to fit with the corresponding images size.



___
