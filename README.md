# VisemeNet infer

CPU Inference version of VisemeNet-tensorflow https://github.com/yzhou359/VisemeNet_tensorflow

Original VisemeNet_tensorflow requires CUDA 8.0 + TensorFlow 1.1.0 environment, which is outdated and quite difficult to setup such environment.

VisemeNet freezes tensorflow graph so that it runs in recent TensorFlow and also without GPU(CUDA).

## Requirements

* TensorFlow 1.12(pip installed CPU version recommended)
* Python 3.5 or 3.6 recommended

## How to freeze graph

First you need to build TensorFlow 1.1 to get `freeze_graph` tool for freezing graph.

### Bazel 0.4.5 for Tensorflow 1.1

```
$ curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh
$ chmod +x bazel-0.4.5-installer-linux-x86_64.sh
$ ./bazel-0.4.5-installer-linux-x86_64.sh --user
$ PATH=$HOME/bin/$PATH
```

### Build Tensorflow 1.1

Note: Python 3.7 is not supported. Plase use 3.6.

```
$ git clone https://github.com/tensorflow/tensorflow
$ git checkout r1.1
$ ./configure
```

#### Build pip package
```
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD/tensorflow_pkg
$ sudo pip3 install tensorflow_pkg/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
```


#### Build freeze tool
```
$ bazel build tensorflow/python/tools:freeze_graph
```

### Create graphdef file
In the directory of `VisemeNet_tensorflow`, run the following Python code.

```
import tensorflow as tf

from src.model import model
from src.utl.load_param import model_dir

if __name__ == '__main__':

    model_name='pretrain_biwi'

    with tf.Graph().as_default() as graph:

        init, net1_optim, net2_optim, all_optim, x, x_face_id, y_landmark, \
        y_phoneme, y_lipS, y_maya_param, dropout, cost, tensorboard_op, pred, \
        clear_op, inc_op, avg, batch_size_placeholder, phase = model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        max_to_keep = 20
        saver = tf.train.Saver(max_to_keep=max_to_keep)

        OLD_CHECKPOINT_FILE = model_dir + model_name + '/' + model_name +'.ckpt'

        saver.restore(sess, OLD_CHECKPOINT_FILE)
        print("Model loaded: " + model_dir + model_name)

        tf.train.write_graph(sess.graph_def, '.', 'graphdef.pbtxt')
        print("Graph def is output")
```

### Create frozen graph file

```
$ ./bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=../VisemeNet_tensorflow/graphdef.pbtxt \
  --input_checkpoint=../VisemeNet_tensorflow/data/ckpt/pretrain_biwi/pretrain_biwi.ckpt \
  --output_graph=visemenet_frozen.pb \
  --output_node_names=net2_output/add_1,net2_output/add_4,net2_output/add_6
```

#### NOTE: Node correspondance

- net2_output/add_1 : v_cls
- net2_output/add_4 : v_reg
- net2_output/add_6 : jali

## Inference

Put `use_fronzen.py` to `VisemeNet-tensorflow` directory.

Edit file path in `use_frozen.py`, then simply run

```
$ python use_frozen.py
```

You may need to pip install `scipy`, `python_speech_features`, etc if required.

You'll get maya animation parameter file as done in original `VisemeNet-tensorflow`.

## License

Python script is licensed under MIT license.

### VisemeNet license

`use_frozen.py` uses some python code from `VisemeNet-tensorflow`. It is unclear that what is the license of VisemeNet-tensorflow.
