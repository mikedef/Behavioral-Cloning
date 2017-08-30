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
  
### Data Set Summary & Exploration
