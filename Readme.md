 # Training and Deployment of Bitcoin Price Prediction Model 

 ## Overview:

 This repo explains how we can train and deploy of __Bitcoin price Prediction__ using _Tensorflow's  Neural Network Model_ 

Bitcoin: Basic explanation ????


 ### Requirments:
 - Pandas
 - Numpy
 - matplotlib
 - Tensorflow

 ## Steps:
 - Setup the Environment and Libraries
 - Choosing the Right Model Architecture
 - Exploring the Bitcoin Dataset and Preparing Data for a Model
 - Training a Neural Network with Different Hyperparameters
 - Assembling a Deep Learning System
 - Optimizing a Deep Learning Model
 - Deploying a Deep Learning Application

<hr>

## Setup the Environment and Libraries

Before we starting to training neural network, let's install and configure all the required libraries that we need are available.

Create a Virtual Environment and Install required Libraries, which helpfull on managing project dependencies.
  
``` sh
"A python virtual environment can be created by using the following"

$ python -m venv venv
$ source venv/bin/activate
```

A `requirement.txt` has attached in this repo, run it CMD will install all required lib for this project . 

``` sh
$ pip install –r requirements.txt
```

<p align = "center">
<img src = ".\images\pip_install.png"> </img> </p>


This will install the libraries used in this project in that virtual environment. It will do nothing if they are already available. If the library is getting installed, a progress bar will be shown, else it will notify that 'requirement is already specified'. 

To check the available libraries installed, please use the following command:
        
```sh 
$ pip list
```

<p align = "center">
<img src = ".\images\pip_list.png"> </img> </p>

### Dataset:

- The network that used here is trained to recognize numerical digits (integers) using `images of handwritten digits`. It uses the `MNIST dataset`, a classic dataset frequently used for exploring pattern recognition tasks.
  
- The `Modified National Institute of Standards and Technology (MNIST)` dataset contains a training set of `60,000` images and a test set of `10,000` images. Each image contains a single handwritten number. 
  
- This dataset, which is derived from one created by the US Government is very useful for understanding how `neural networks models` can achieve a `high level of accuracy with great efficiency`. 

<hr>

## Neural Network Model Architecture

Considering the available architecture possibilities, there are two popular architectures that are often used as starting points for several applications: 
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs). 

We will be implementing a special-purpose neural network called a Convolutional Neural Network(CNN)

Our complete network contains three hidden layers: `two fully connected layers` and `a convolutional layer`. The model is defined using Python - TensorFlow:

```py
model = Sequential()
model.add(Convolution2D(filters = 10, kernel_size = 3, \
                        input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

```

<hr>

## Exploring the Bitcoin Dataset and Preparing Data for a Model

Import `lib` mentioned in requirments, read dataset into variable `bitcoin` by load file from `.\data\bitcoin_dataset.csv` on jupyter nootbook `.nootbook\Exploring_Bitcoin_Dataset.ipynb`

```py
bitcoin.set_index('date')['close'].plot(linewidth=2, \
                                        figsize=(14, 4),\
                                        color='#d35400')
plt.plot(bitcoin['date'], bitcoin['close'])

```

<p align = "center">
<img src = ".\images\bitcoin_chart.png"> </img> </p>

### Preparing Dataset for Model
Neural networks typically work with either [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)) or [tensors](https://en.wikipedia.org/wiki/Tensor). Our data needs to fit that structure before it can be used by either `keras` (or `tensorflow`). 

Also, it is common practice to normalize data before using it to train a neural network. We will be using a normalization technique the evaluates each observation into a range between 0 and 1 in relation to the first observation in each week.

`.\src\normalizations.py` is attached is imported for normalization the dataset

It contains three functions
- `z_score`, `maximum_and_minimum_normalization` and `point_relative_normalization`


<p align = "center">
<img src = ".\images\norm.png"> </img> </p>

After the normalization procedure, our variables close and volume are now relative to the first observation of every week. We will be using these variables -- close_point_relative_normalization and volume_point_relative_normalization, respectivelly -- to train our LSTM model.

<p align = "center">
<img src = ".\images\norm_img.png"> </img> </p>


### Training and Test Dataset

Let's divide the dataset into a training and a test set. In this case, we will use 80% of the dataset to train our LSTM model and 20% to evaluate its performance.

Given that the data is continuous, we use the last 20% of available weeks as a test set and the first 80% as a training set.


```py
boundary = int(0.9 * bitcoin_recent['iso_week'].nunique())

train_set_weeks = bitcoin_recent['iso_week'].unique()[0:boundary]

test_set_weeks = bitcoin_recent[~bitcoin_recent['iso_week'].isin(train_set_weeks)]['iso_week'].unique()

```
### Storing Output

```py

bitcoin_recent.to_csv('data/bitcoin_recent.csv', index=False)
train_dataset.to_csv('data/train_dataset.csv', index=False)

```


<hr>

## Training a Neural Network with Different Hyperparameters

1. Using your Terminal / CMD, navigate to the directory cloned from this repo and execute the following command to start TensorBoard:
   
    ``` sh
        $ tensorboard --logdir logs/fit
    ```

2. Now, open the URL provided by TensorBoard in your browser. You should be able to see the TensorBoard SCALARS page:

    <p align = "center">
    <img src = ".\images\scalar.png"> </img> </p>

3. On the TensorBoard page, click on the SCALARS page and enlarge the epoch_accuracy graph. Now, move the smoothing slider to 0.6.

    The accuracy graph measures how accurately the network was able to guess the labels of a test set. 

    At first, the network guesses those labels completely incorrectly. This happens because we have initialized the weights and biases of our network with random values, so its first attempts are a guess. 

    The network will then change the weights and biases of its layers on a second run; the network will continue to invest in the nodes that give positive results by altering their weights and biases and will penalize those that don't by gradually reducing their impact on the network

 4. For improving neural network efficiency try to change `epochs` and `learning_rate` in `.\src\train_nn.py`

    ```py 

    learning_rate=0.001
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test),callbacks=[tensorboard_callback])
   
