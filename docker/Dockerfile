FROM continuumio/anaconda3

RUN apt-get update -y \
    && apt-get install wget unzip git gcc diffutils -y

# Step 1 - create your Conda virtual Environment named tensorflow2
WORKDIR /home/Dotnetmobile/Applications

RUN conda create -n tensorflow2 pip python=3.8

# Override default shell and use bash
SHELL ["/bin/bash","-login","-c"]

RUN activate tensorflow2

# Step 2 - install all TensorFlow python packages
RUN pip install tensorflow tensorflow-gpu \
    && conda install -c anaconda protobuf \
    && pip install pillow \
    && pip install lxml \
    && pip install Cython \
    && pip install contextlib2 \
    && pip install jupyter \
    && pip install matplotlib \
    && pip install pandas \
    && pip install opencv-python \
    && pip install lvis \
    && pip install pyyaml \
    && pip install scipy \
    && pip install tf_slim \
    && pip install pycocotools

# Step 3: create environment variable MYHOME
WORKDIR /home/Dotnetmobile/Documents

ENV MYHOME=/Users/Dotnetmobile/Documents

# Step 4: create the tensorflow folders environment based on Git source code repository
WORKDIR $MYHOME/tensorflow2

RUN cd $MYHOME/tensorflow2

RUN git clone https://github.com/tensorflow/models.git

# Step 5: extend your existing environment variable PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:$MYHOME/tensorflow2/models/:$MYHOME/tensorflow2/models/research/:$MYHOME/tensorflow2/models/research/slim/

# Step 6: download Edge Electronic tutorial
WORKDIR $MYHOME/tensorflow2/models/research/object_detection

RUN wget https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/archive/master.zip \
    && unzip master.zip \
    && rm master.zip \
    && mv  ./TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master/* .

# Step 7: process files cleaning before training a model from scratch
WORKDIR $MYHOME/tensorflow2/models/research/object_detection/images/test/
RUN rm -r *

WORKDIR $MYHOME/tensorflow2/models/research/object_detection/images/train/
RUN rm -r *

WORKDIR $MYHOME/tensorflow2/models/research/object_detection/images/
RUN rm *.csv

WORKDIR $MYHOME/tensorflow2/models/research/object_detection/training/
RUN rm -r *

WORKDIR $MYHOME/tensorflow2/models/research/object_detection/inference_graph
RUN rm -r *

# Step 9: compile Protobufs and execute setup.py
WORKDIR $MYHOME/tensorflow2/models/research

RUN protoc --python_out=. ./object_detection/protos/anchor_generator.proto \
./object_detection/protos/argmax_matcher.proto \
./object_detection/protos/bipartite_matcher.proto \
./object_detection/protos/box_coder.proto \
./object_detection/protos/box_predictor.proto \
./object_detection/protos/eval.proto \
./object_detection/protos/faster_rcnn.proto \
./object_detection/protos/faster_rcnn_box_coder.proto \
./object_detection/protos/grid_anchor_generator.proto \
./object_detection/protos/hyperparams.proto \
./object_detection/protos/image_resizer.proto \
./object_detection/protos/input_reader.proto \
./object_detection/protos/losses.proto \
./object_detection/protos/matcher.proto \
./object_detection/protos/mean_stddev_box_coder.proto \
./object_detection/protos/model.proto \
./object_detection/protos/optimizer.proto \
./object_detection/protos/pipeline.proto \
./object_detection/protos/post_processing.proto \
./object_detection/protos/preprocessor.proto \
./object_detection/protos/region_similarity_calculator.proto \
./object_detection/protos/square_box_coder.proto \
./object_detection/protos/ssd.proto \
./object_detection/protos/ssd_anchor_generator.proto \
./object_detection/protos/string_int_label_map.proto \
./object_detection/protos/train.proto \
./object_detection/protos/keypoint_box_coder.proto \
./object_detection/protos/multiscale_anchor_generator.proto \
./object_detection/protos/graph_rewriter.proto \
./object_detection/protos/calibration.proto \
./object_detection/protos/flexible_grid_anchor_generator.proto \
./object_detection/protos/fpn.proto \
./object_detection/protos/center_net.proto

# Step 10: generate the training records
# It creates a train_labels.csv and test_labels.csv file in the /object_detection/images folder
# $MYHOME/tensorflow2/models/research/object_detection/images

WORKDIR $MYHOME/tensorflow2/models/research/object_detection

RUN python xml_to_csv.py

# Step 11: update generate_tfrecord.py
WORKDIR $MYHOME/tensorflow2/bredele-hackathon
WORKDIR $MYHOME/tensorflow2

COPY ./generate_tfrecord.py ./models/research/object_detection

# Then generate the TFRecord files by issuing these commands from the /object_detection folder:
WORKDIR $MYHOME/tensorflow2/models/research/object_detection

# Steps below generate a train.record and a test.record file in /object_detection will be used to train the new object detection classifier.

RUN python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

RUN python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

# We skip the following steps as they folders are mounted once the docker is running executing docker run as:
# sudo docker run -it \
# --mount type=bind,source=/Users/Dotnetmobile/Documents/bredele-hackathon/images-small/training,\
# target=/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/images/train \
# --mount type=bind,source=/Users/Dotnetmobile/Documents/bredele-hackathon/images-small/test,\
# target=/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/images/test\
# bredele-hackathon bash

# Step 12: create label map + configure training
# to skip

# Step 13: change default configurations
# to skip
