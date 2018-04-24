# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
if ou want to record the images to make video later , use the code like:
```sh
python drive.py model.h5 image_folder_path
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model consists of a convolution neural network with Convolution 5x5 and Convolution 3x3 and fully connected

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer . 

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.


Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .Just put them into train dataset together , and I found the result is great 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to whatever let the car driving safty.

My first step was to use a convolution neural network model similar to the Lenet I thought this model might be appropriate because the goal of my work just read a picture and output a steering angle

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the dropout is using and the result is not bad.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


The final model architecture  consisted of a convolution neural network with the following layers and layer sizes:
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)


To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

I didn't need to augment the data set, because the result is quite good. But if we did it, the resut seems better

![alt text][image6]
![alt text][image7]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the loss decent very slow after this epoch . I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Result
Here comes my result , I did the smae thing in both track :
[track1](https://www.youtube.com/watch?v=3gycqGv_ogk)
[track2](https://www.youtube.com/watch?v=3bcAyncIUnE&t=40s)
I think maybe i can try train the self-driving car in track 1 and test it in track 2 next time
### Reference
- https://www.youtube.com/watch?v=rpxZ87YFg0M&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&t=0s&index=3
- https://github.com/patdring/CarND-Term1-Behavioral-Cloning-P3/blob/master/model.py
- https://medium.com/waymo/with-waymo-in-the-drivers-seat-fully-self-driving-vehicles-can-transform-the-way-we-get-around-75e9622e829a
