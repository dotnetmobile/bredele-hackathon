# The Bredele hackathon
  by dotnetmobile@gmail.com
---

## Proof of concept based on a [Raspberry pi 4 (8GB RAM)](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) and a [Coral USB device](https://coral.ai/products/accelerator) for detecting [Bredele](https://en.wikipedia.org/wiki/Bredele) using the [TensorFlow Machine Learning platform version 2](https://www.tensorflow.org) and the [Faster RCNN Inception model](https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1).

All credits go to the authors (@EdjeElectronics + @tensorflow) of following remarkable references:

* [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
* [How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow (GPU) on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
* [TensorFlow](https://github.com/tensorflow)
* [TensorFlow models](https://github.com/tensorflow/models)
* [TensorFlow model configurations](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

---
## Context

I successfully deployed on my Raspberry Pi all the installation steps of the excellent tutorial [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi).
I noticed that the objects detection trained model was time to time mismatching objects like apples, oranges and balls.
Therefore I decided to retrain the model in order to see if I can get a better object detection.
As it was the Christmas period and we cooked quite a lot of Bredele, I trained the Faster RCNN Inception model using our own cookies.
It was a good starting point for using TensorFlow :wink:

---

## Objects

The trained model should be able to detect 16 different object types:

|  Id  | Name                                 | Comment                      |
|:----:|--------------------------------------|------------------------------|
|1     | [Christmas tree](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_88_of_511.jpeg)                   |                              |
|2     | [flake](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_137_of_511.jpeg)                            |                              |
|3     | [heart #1](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_385_of_511.jpeg)                            | commercial                   |
|3     | [heart #2](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_153_of_511.jpeg)                            |                              |
|4     | [king mage](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_136_of_511.jpeg)                        |                              |
|5     | [lamb](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_157_of_511.jpeg)                             |                              |
|6     | [Santa Claus #1](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_140_of_511.jpeg)                      |                              |
|6     | [Santa Claus #2](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_133_of_511.jpeg)                      |                              |
|7     | [snowball](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_294_of_511.jpeg)                         |                              |
|8     | [snowman](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_132_of_511.jpeg)                          |                              |
|9     | [shooting star](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_156_of_511.jpeg)                    |                              |
|10    | [squirrel](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_151_of_511.jpeg)                         |                              |
|11    | [star #1](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_383_of_511.jpeg)                             | commercial               |
|11    | [star #2](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_165_of_511.jpeg)                             |                              |
|12    | [weird #1](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_406_of_511.jpeg)                         | forgot baker's yeast :joy:     |
|13    | [weird #2]()                         | forgot baker's yeast :joy:     |
|14    | [Christmas shortbread with jam](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_290_of_511.jpeg)    |                              |
|15    | [angel](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_148_of_511.jpeg)                            |                              |
|16    | [bretzel](https://github.com/dotnetmobile/bredele-hackathon/blob/main/images-small/training/bredelehackathon_384_of_511.jpeg)                          | commercial                  |

---

## Environment

* OS: [MacOS Big Sur](https://www.apple.com/uk/macos/big-sur/)
* [Anaconda 1.10.0](https://www.anaconda.com) with Python 3.8
* [TensorFlow Object Detection Git repository](https://github.com/tensorflow/models)
* [LabelImg](https://github.com/tzutalin/labelImg) for objects annotation
* [EdgeElectronics git repository](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/archive) instructions for training the model

## Setup the bredele bredele hackathon environment

Follow instructions described in [INSTALL.md](https://github.com/dotnetmobile/bredele-hackathon/INSTALL.md)




___
