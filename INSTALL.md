# The Bredele hackathon installation guide
  by dotnetmobile@gmail.com
---

## Quick installation
This installation steps have been done for TensorFlow 2.4 + Anaconda with Python 3.8

[Download and install Anaconda](https://www.anaconda.com/products/individual)

### Step 1 - create your Conda virtual Environment named *tensorflow2*

```
cd /Users/Dotnetmobile/Applications

conda create -n tensorflow2 pip python=3.8

conda activate tensorflow2
```

### Step 2 - install all TensorFlow python packages

```
pip install tensorflow tensorflow-gpu

conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python
pip install lvis
pip install pyyaml
pip install scipy
```

### Step 3: create environment variable MYHOME

```
export MYHOME=/Users/Dotnetmobile/Documents
```

### Step 4: create the tensorflow folders environment based on Git source code repository

```
mkdir $MYHOME/tensorflow2

cd $MYHOME/tensorflow2

git clone https://github.com/tensorflow/models.git
```

### Step 5: extend your existing environment variable PYTHONPATH

```
export PYTHONPATH=$PYTHONPATH:$MYHOME/tensorflow2/models/:$MYHOME/tensorflow2/models/research/:$MYHOME/tensorflow2/models/research/slim/
```

### Step 6: download Edge Electronic tutorial

```
cd $MYHOME/tensorflow2/models/research/object_detection

wget https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/archive/master.zip

unzip master.zip

rm master.zip

mv  ./TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master/* .
```

### Step 7: process files cleaning before training a model from scratch

```
cd $MYHOME/tensorflow2/models/research/object_detection/images/test/
rm -r *

cd $MYHOME/tensorflow2/models/research/object_detection/images/train/
rm -r *

cd $MYHOME/tensorflow2/models/research/object_detection/images/
rm *.csv

cd $MYHOME/tensorflow2/models/research/object_detection/training/
rm -r *

cd $MYHOME/tensorflow2/models/research/object_detection/inference_graph
rm -r *
```

### Step 8: retrieve the bredele images

```
cd $MYHOME/tensorflow2

git clone https://github.com/dotnetmobile/bredele-hackathon.git

cp -r ./bredele-hackathon/images-small/training/*  ./models/research/object_detection/images/train

cp -r ./bredele-hackathon/images-small/test/*  ./models/research/object_detection/images/test
```

### Step 9: compile Protobufs and execute setup.py

```
cd $MYHOME/tensorflow2/models/research

protoc --python_out=. ./object_detection/protos/anchor_generator.proto \
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
```

### Step 10: generate the training records

```
cd $MYHOME/tensorflow2/models/research/object_detection
python xml_to_csv.py
```

It creates a **train_labels.csv** and **test_labels.csv** file in the **/object_detection/images** folder

### Step 11: update generate_tfrecord.py

```
cd $MYHOME/tensorflow2
cp bredele-hackathon/generate_tfrecord.py ./models/research/object_detection
```

Then generate the TFRecord files by issuing these commands from the **/object_detection** folder:

```
cd $MYHOME/tensorflow2/models/research/object_detection

python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

It generates a **train.record** and a **test.record** file in **/object_detection** will be used to train the new object detection classifier.

### Step 12: create label map + configure training

```
cd $MYHOME/tensorflow2

cp -r bredele-hackathon/training/* ./models/research/object_detection/training/
```

### Step 13: change default configurations

```
cd $MYHOME/tensorflow2/models/research/object_detection/training
```

Edit hyper parameters <bf>

```
vi faster_rcnn_inception_resnet_v2_756x1008_coco17_tpu-8.config
```
<br>

and change the following lines:


#### 13.1: Check that lines *103+104* are defined as

```
fine_tune_checkpoint: ""
fine_tune_checkpoint_type: "detection"
```

#### 13.2: adapt path in lines *133* to *135* with your own $MYHOME path; here you see mine by default

```
train_input_reader: {
  label_map_path: "/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/training/labelmap.pbtxt"
  tf_record_input_reader {
    input_path: "/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/train.record"
  }
}
```

#### 13.3: adapt lines *146* to *150* with your own $MYHOME path; here you see mine by default

```
eval_input_reader: {
  label_map_path: "/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/training/labelmap.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/Users/Dotnetmobile/Documents/tensorflow2/models/research/object_detection/test.record"
  }
}
```

### Step 14 - start the training

```
cd $MYHOME/tensorflow2/models/research/object_detection
```

#### 14.1 create the output folder **bredele_output** which will contains all checkpoints and training outputs

```
mkdir bredele_output
```

#### 14.2 create the first run1 folder

```
mkdir bredele_output/run1
```

#### 14.3 start the training (first run = run1) with 2500 iterations using the hyper parameters values stored in <br>
> *faster_rcnn_inception_resnet_v2_756x1008_coco17_tpu-8.config* <br>
#### and save intermediate checkpoints every 500 iterations

```
python model_main_tf2.py --model_dir=bredele_output/run1 \
--num_train_steps=2500 \
--sample_1_of_n_eval_examples=100 \
--pipeline_config_path=training/faster_rcnn_inception_resnet_v2_756x1008_coco17_tpu-8.config \
--alsologtostderr --checkpoint_every_n=500
```

### Monitoring training and looking into previous trainings using TensorBoard

**Prerequisite:** ensure that you are running the virtual environment tensorflow2

```
cd /Users/Dotnetmobile/Applications

conda activate tensorflow2

EXPORT MYHOME=/Users/Dotnetmobile/Documents

export PYTHONPATH=$PYTHONPATH:$MYHOME/tensorflow2/models/:$MYHOME/tensorflow2/models/research/:$MYHOME/tensorflow2/models/research/slim/

cd $MYHOME/tensorflow2/models/research/object_detection
```

Start TensorBoard

```
tensorboard --logdir=bredele_output --bind_all
```

Finally open your Internet browser and enter the tensorboard URL provided by the previous command line output.

:thumbsup: That's it !

### Retrain the model with new hyper parameters values in a new output folder like **run2**
This is a short way once all previous steps have been executed (step 1 -> Step 13)

```
cd /Users/Dotnetmobile/Applications

conda activate tensorflow2/

EXPORT MYHOME=/Users/Dotnetmobile/Documents

export PYTHONPATH=$PYTHONPATH:$MYHOME/tensorflow2/models/:$MYHOME/tensorflow2/models/research/:$MYHOME/tensorflow2/models/research/slim/

cd $MYHOME/tensorflow2/models/research/object_detection

mkdir bredele_output/run2

python model_main_tf2.py --model_dir=bredele_output/run2 --num_train_steps=2500 --sample_1_of_n_eval_examples=1 --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.config --alsologtostderr --checkpoint_every_n=500
```

:thumbsup: That's it !
