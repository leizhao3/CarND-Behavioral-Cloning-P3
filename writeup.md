# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images_output/1.CarDidding.png "1.CarDidding.png"
[image2]: ./test_images_output/2.Before_After_ImgAugmentation.jpg "2.Before_After_ImgAugmentation.jpg"
[image3]: ./test_images_output/2.Before_After_ImgCropping.jpg "2.Before_After_ImgCropping.jpg"
[image4]: ./test_images_output/2.BehaviorCloning_NN.svg "2.BehaviorCloning_NN.svg"
[image5]: ./test_images_output/4.Angles_Distribution.png "4.Angles_Distribution.png"
[image6]: ./test_images_output/5.CenterLaneDriving.jpg "5.CenterLaneDriving.jpg"
[image7]: ./test_images_output/5.RecoverFromSide_1.jpg "5.RecoverFromSide_1.jpg"
[image8]: ./test_images_output/5.RecoverFromSide_2.jpg "5.RecoverFromSide_2.jpg"
[image9]: ./test_images_output/5.RecoverFromSide_3.jpg "5.RecoverFromSide_3.jpg"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **P3_01_01_003.py**: containing the script to create and train the model
* **drive.py**: for driving the car in autonomous mode
* **model_01_01_003.h5**: containing a trained convolution neural network
* **writeup.md**: summarizing the results
* **run3.mp4**: the final video for model drive car autonomously.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_01_01_003.h5
```

#### 3. Submission code is usable and readable

The P3_01_01_003.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 & 5x5 filter sizes and depths between 24 and 64 (P3_01_01_003.py lines 205-260)

The model includes RELU layers to introduce nonlinearity (code line 215), and the data is normalized in the model using keras.utils.normalize (code line 173). In addition, a cropping layer is used to delete unnecessary image area(code line 209).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (P3_01_01_003.py lines 217).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 279). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (P3_01_01_003.py line 273).

Epoches(line 278), corrections in left/right images(line 123), dropout ratio (line 192), are also trailed in different parameter to minimize the valid loss.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination listed below:

    * 2x center lane driving, clockwise
    * 1x center lane driving, counter-clockwise
    * 1x recovered from side

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find existing model with similar application.

My first step was to use a convolution neural network model similar to the the paper *End to End Learning for Self-Driving Car* by NVIDIA I thought this model might be appropriate because it trains the model by using multiple camera setup & also output the steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my **first model** had 1) a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 2) car is diddling around the center line, shown below:
![alt text][image1]

To combat the overfitting, I modified the model so that I tried different dropout ratio.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I played with different corrections in left/right images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (P3_01_01_003.py line 204-259) consisted of a convolution neural network with the following layers and layer sizes.

![alt text][image4]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to center. These images show what a recovery looks like starting from right lane to center:

![alt text][image7]
![alt text][image8]
![alt text][image9]

To augment the data sat, I also flipped images and angles thinking that this would have symmetric data distribution on positive & negative steering angle. For example, here is an image that has then been flipped:

![alt text][image2]

Before feed into the training, I also cropped out the unecessary area such as sky, so that the NN would focus on the lane line. Example pictures is shown below:
![alt text][image3]

After the collection process, I had 13,000 number of data points. I then preprocessed this data by normalizing them. Below shows the means, stds, mins, and maxs of data after normalizing:
Data After Normalizing & Center
Means: [[ 0.1024  0.1050 -0.01381]]
stds:  [[0.0986  0.0533 0.1483]]
Mins:  [[-0.5 -0.5 -0.5]]
Maxs:  [[0.5 0.5 0.5]]


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by most of the case the valid loss stop decreasing around 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At the end of the training, I also investigate the distribution of the steering angle to make sure that majority of the data lays around center. For the final model, steering angle distribution is shown below:
![alt text][image5]
