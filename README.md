# Character(Digits + Upper Case Letters) Recognition Using Tensorflow 

This project performs character recognition using deep learning concepts. It can classify image into 36(0...9 + A...Z) classes. It was initially built to perform character recognition from number plate which contains only digit and capital alphabet letters. This is totaly seperate module to number plater recognition. we can intigrate this project with number plate localizetion and segmentation. Here I provide a neural network implementation to perform character recognition. It implements a simple but efficient convolution neural network using most popular library tensorflow.

# Specification of this project
1. Tensorflow model (graph + parameters) which is created by model.py is self saved and self stored, you don't need to 			provide any external code to save and restore model.
2. You can run training process multiple times. Each time you train your model it starts training your model from where it 		was stoped last training. In other words it loads your model from last(latest) checkpoint and resume training process.
3. In each training, your plot of training accuracy is inserted or appended into logdir file for tensorboard visualization.
4. Training in large model takes many days so you require a model that can be stoped and resumed. In case of power failure 		you don't want to train your whole model again from initialization of variables, So you need a model that start training 	from last checkpoint.
5. This project also use early stopping. Early stopping means when your model accuracy drops in many contiguous iterations,		then it stops training.


# Prerequisites
* Tensorflow version latest by 1.1, see how to [install](https://www.tensorflow.org/install/)
* csv lib
* numpy
* pickle liberary
* Knowledge of deep learning concepts, if you don't feel comfortable working with cnn then you can use [online book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/index.html).
* Emnist data set must be available on your system, [download here](https://www.kaggle.com/crawford/emnist).


# Data Sets
Here we use balanced data sets to train model.
Although downloaded data set also contains small alphabets letters. I provided here preprocess_data.py to remove all small alphabet latter from original data sets.
If you visualize original images in given data sets you see images are flipped or rotated.
So in order to get normal image each image must be flipped vertically and then rotated 90 degree anti-clock-wise. I have provided code to flip and rotate image in [preprocess_data.py](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/preprocess_data.py). After processing file, train file contains 86400 number of images and test file contains 14400 number of image.
# The Model
	It uses csv python module to open given csv file into appropriate csv module. Here we use 5 layers.
		1. Convolutional layer 
			Input  : 4d tensor, dim:[N, w, h, Number of input channel = 1], where N is batch size.
			Output : 4d tensor, dim:[N, w/2, h/2, Number of filters at cnn layer-1]

		2. Convolutional layer 
			Input  : 4d tensor, dim:[N, w/2, h/2, Number of filters at cnn layer-1]
			Output : 4d tensor, dim:[N, w/4, h/4, Number of filters at cnn layer-2]

			Now this output 4d tensor is flattened inorder to provide input to fully connected layer-1.

		3. Fully connected layer
			Input  : 2d tensor, dim:[N, Flattened size]
			Output : 2d tenser, dim:[N, Number of neurons at fully connected layer-1]

		4. Fully connected layer
			Input  : 2d tensor, dim:[N, Number of neurons at fully connected layer-1]
			Output : 2d tenser, dim:[N, Number of neurons at fully connected layer-2]

		5. Output layer.
			Input  : 2d tensor, dim:[N, number of neurons at fully connected layer-2]
			Output : 2d tenser, dim:[N, Number of classes]

# How to run
1. To prepare data sets run appropriate method from preprocess_data.py. Discription to each method being used has given as comments in [preprocess_data.py](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/preprocess_data.py) file. Once you prepared your data sets, you are all set to train.
2. Run [train_model.py](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/train_model.py) as many times as you want, As everything is being stored in "store/..." directory to you don't worry about consistancy of training process. It will ask you about number of epochs for training model. Although Training process may be interruped by [early stopping strategy](https://en.wikipedia.org/wiki/Early_stopping).
3. Although we perform testing along with training, But if you want to test model seperately then run [evaluation.py](https://github.com/Devbishnoi29/https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/evaluation.py) to evaluate model on testing data.

# Model graph
![graph goes here](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/images/graph.png)

# Plot of Cost while training
![cost plot](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/images/cost.png)

# Plot of Training Accuracy while training
![Train accuracy](https://github.com/Devbishnoi29/Digits-and-Capital-Alphabet-Recognition-using-Tensorflow/blob/master/images/accuracy.png)

# Efficiency
This such a small model gains 81% accuracy on testing data sets. Although this may differ with your testing accuracy because of random initialization of biases and weights. Because of computational power limitations we are limited to such small convolutional neural network. But testing accuracy may increase when we increase number of parameters (biases and weights).

# About me
I am a computer programmer who loves to solve programming problems and exploring the exciting possibilities using deep learning. I am interested in solving real life problems using efficient algorithms and computer vision that creates innovative solutions to real-world problems. I hold a B.Tech degree in computer Engineering From Nit kurukshetra. You can reach me on [LinkedIn](https://www.linkedin.com/in/devi-lal-468596126/).