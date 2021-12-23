Run the docker as following:
```
sudo docker run -it \
--mount type=bind,source=/Users/Dotnetmobile/Documents/bredele-hackathon/images-small/training,\
target=/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/images/train \
--mount type=bind,source=/Users/Dotnetmobile/Documents/bredele-hackathon/images-small/test,\
target=/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/images/test\
bredele-hackathon bash
```
