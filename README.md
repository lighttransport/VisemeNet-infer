# VisemeNet Code Readme

## Environment

+ Python 3.5 
+ Tensorflow 1.1.0 
+ Cudnn 5.0

## Python Package

+ numpy
+ scipy
+ python_speech_features
+ matplotlib

## Input/Output

+ Input audio needs to be 44.1kHz, 16-bit, WAV format
+ Output visemes are applicable to the JALI-based face-rig, see [HERE](http://www.dgp.toronto.edu/~elf/jali.html)

## JALI Viseme Annotation Dataset

+ BIWI dataset with well-annotated JALI viseme parameters. [[DATASET](https://www.dropbox.com/sh/oj13tvq9ggf2puz/AADBPyRUcyisFtKgCoDmNhLHa?dl=0)]   [[README](VisemeNet_Annotation_README.md)]

## At test time:

1. **Create and install required envs and packages**
```
conda create -n visnet python=3.5
  
# take care of your OS and python version, here is a Linux-64bit with Python3.5 link
pip install --ignore-installed --upgrade https://download.tensorflow.google.cn/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl
  
pip install PYTHON_PACKAGE_REQUIRED
```
2. **Download this repository to your local machine:**  
```
git clone https://github.com/yzhou359/VisemeNet_tensorflow.git  

cd VisemeNet_tensorflow 
```
3. **Prepare data and model:**  
   * convert your test audio files into WAV format, put it to the directory data/test_audio/   
   * download the public face rig model from [HERE](https://www.dropbox.com/sh/7nbqgwv0zz8pbk9/AAAghy76GVYDLqPKdANcyDuba?dl=0), put all 4 files to data/ckpt/pretrain_biwi/  

4. **Forward inference:**  
   * put your test audio file name in file 'main_test.py', line 7. 
   * Then run command line
```
python main_test.py
```  
   The result locates at:  
```
data/output_viseme/[your_audio_file_name]/mayaparam_viseme.txt
```
5. **JALI animation in Maya:**
   * put your test audio file name in file 'maya_animation.py', line 4.
   * Then run 'maya_animation.py' in Maya with JALI environment to create talking face animation automatically. (If using different version of JALI face rig, the name of phoneme/co-articulation variable might varies.)
   * UPDATE: 'maya_animation.py' has been updated with the [public face rig](http://www.dgp.toronto.edu/~elf/jali.html) annotations. Feel free to play with it!


## tensorflow2.0 support

※miniconda used
pip install tensorflow==2.0
pip install scipy
pip install python_speech_features
pip install matplotlib

python v2_use_frozen.py



https://www.tensorflow.org/guide/upgrade
Automatically upgrade code to TensorFlow 2


usage: tf_upgrade_v2 [-h] [--infile INPUT_FILE] [--outfile OUTPUT_FILE]
                     [--intree INPUT_TREE] [--outtree OUTPUT_TREE]
                     [--copyotherfiles COPY_OTHER_FILES] [--inplace]
                     [--reportfile REPORT_FILENAME] [--mode {DEFAULT,SAFETY}]
                     [--print_all]


Add any file name to the option according to nootbook


