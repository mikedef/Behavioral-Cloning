# Behavioral-Cloning

## Michael DeFilippo

#### Please see my [project code](https://github.com/mikedef/Behavioral-Cloning/blob/master/behavioral-cloning-submission/model.py) for any questions regarding implimentation.

## Project Goals:
  1. Develop a NN in Keras that can predict stearing angles from images
  2. Use a simulator to collect and store images and associated steering angles to train the NN on.
  3. Train and validate the model in Keras
  4. Autonomously drive around the track without leaving the road or hitting the sides of the track.
  
### Data Gathering 
Udacity provided a training simulator that collects images from 3 cameras mounted on a vehicle at the left, right, and center. I first gathered my own data based on following the progression of the lesson and the recomendations of the lecturer. After much frustration with not sucessfully training a network to drive a vehicle in the simulator autonomusly, I look up many forum questions and found that the simulator is flawed when using a keyboard. After much time I decided to train my network using the provided dataset from the project.

![png](behavioral-cloning-submission/simulator_home.jpg)
![png](behavioral-cloning-submission/simulator_track1.jpg)
  
### Data Set Summary & and Preparation
#### Simulator Images
First lets look at a subset of randomly selected data from the center camera as shown below.

![png](behavioral-cloning-submission/example_dataset.png)

It is easy to see that there are many pixels in the image that will not provide any useful information to the NN, such as the hood of the car or anything above the horizon line. 

![png](behavioral-cloning-submission/cropped_dataset.png)

Cropping off the bottom 20 pixels and the top 50 pixels show that this is a much more useful image. This will also cuts down on the amount of pixels to process in each image. 

Lets also look at a corrolated set of images from the left, center, and right cameras of the vehicle.

![png](behavioral-cloning-submission/left_center_right_camera_view.png)

#### Metadata
The dataset shows that most of the steering angles are close to zero, which indicates mostely straight driving. 

![png](behavioral-cloning-submission/NN_loss.png)

#### Augmentation Techniques
Augmenting the dataset is a technique to increase the amount of data available to train the network. Since the provided images were limited and I could not collect any extra data due to a lack of a video game control, I decided to augment the images. 
Below is an image from the center camera of the vehicle. 

![png](behavioral-cloning-submission/Center Camera Image, steering 0.06.png)
