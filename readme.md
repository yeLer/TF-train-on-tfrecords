# Train mnist AlexNet with Tensorflow from tfrecords

**Update 05.24.2019**


**Note**: Please keep in mind that the network architecture is just same as the AlextNet, not is the models in the paper.

## Requirements

- Python 3
- TensorFlow >= 1.9.0
- Numpy


## Usage
please download the mnist data, and then put them in the folder mnist_data,in this folder need to include thses:
+ t10k-images-idx3-ubyte.gz
+ t10k-labels-idx1-ubyte.gz
+ train-images-idx3-ubyte.gz
+ train-labels-idx1-ubyte.gz

Then you need to run tfrecords.py to get the tfrecords.
```bash
python tfrecords.py
```

I strongly recommend to take a look at the entire code of this repository. In the `alex.py` script you will find a section of configuration settings 
you have to adapt on your problem.You can just run the command by:
```bash
python train.py
```


