 # Training and Deployment of Tensorflow's Prediction Neural Network Model 

 ## Overview:

 This repo explains how we can train and deploy of Tensorflow's Prediction Neural Network Model 

 ## Steps:
 - __Setup the Environment and Libraries__
 - __Training a Neural Network with Different Hyperparameters__
 - __Assembling a Deep Learning System__
 - __Optimizing a Deep Learning Model__
 - __Deploying a Deep Learning Application__

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

- The `Modified National Institute of Standards and Technology (MNIST)` dataset contains a __training set__ of `60,000` images and a __test set__ of `10,000` images. Each image contains a single handwritten number. 
- This dataset, which is derived from one created by the __US Government__ is very useful for understanding how `neural networks models` can achieve a `high level of accuracy with great efficiency`. 

<hr>


## Training a Neural Network with Different Hyperparameters

1. Using your Terminal / CMD, navigate to the directory cloned from this repo and execute the following command to start TensorBoard:
   
    ``` sh
        $ tensorboard --logdir logs/fit
    ```
    <p align = "center">
    <img src = ".\images\tensorboard.png"> </img> </p>


2. Now, open the URL provided by TensorBoard in your browser. You should be able to see the TensorBoard SCALARS page:

    <p align = "center">
    <img src = ".\images\scalar.png"> </img> </p>

3. On the TensorBoard page, click on the SCALARS page and enlarge the epoch_accuracy graph. Now, move the smoothing slider to 0.6.

    The accuracy graph measures how accurately the network was able to guess the labels of a test set. 

    At first, the network guesses those labels completely incorrectly. This happens because we have initialized the weights and biases of our network with random values, so its first attempts are a guess. 

    The network will then change the weights and biases of its layers on a second run; the network will continue to invest in the nodes that give positive results by altering their weights and biases and will penalize those that don't by gradually reducing their impact on the network

 4. For improving neural network efficiency try to change `epochs` and `learning_rate` in __`.\src\train_nn.py`__

    ```py 

    learning_rate=0.001
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test),callbacks=[tensorboard_callback])
   
