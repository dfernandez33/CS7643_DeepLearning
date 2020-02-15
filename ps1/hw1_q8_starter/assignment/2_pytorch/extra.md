# CS 7643 Deep Learning - Homework 1 Extra Credit

**Name:** David Jose Fernandez

**Email:** dfernandez33@gatech.edu

**Best Accuracy:** 77%

For the extra credit section I decided to make a deep convolutional 
nueral network with the following architecture:

**conv-relu-dropout-conv-relu-pool**x3 - **affine**x3 - **softmax(indirectly implemented in the loss function)**

A couple of interesting things can be pointed out with regards to the main block of convolutions used in the network.
For one, I found it very useful to add padding to the convolution layers in order to prevent dimensionality reduction
when applying the convolutions. This allowed me to control the dimensionality reduction using the pooling layer at
the end of every convolution block. After 3 of these [conv-relu-dropout-conv-relu-pool] I realized that I was convoluting
over only a 4x4 "image" which wasn't helping the model learn anything by performing on such small inputs. This is why I
decided to then proceed to add a set of 3 affine layers to the end of the net. Since substantial dimensionality reduction
had been done at his point, computing fully connected layers was less computationally expensive than at the raw input.
Another interesting feature about the network is that a dropout layer is embedded in the middle of every [conv-relu-dropout-conv-relu-pool]
block. I decided to add this towards the end of my experimenting since I started to notice some overfitting on the training 
data and thought that adding a small dropout percentage would allow the model to better generalize to the testing set.
Lastly, I realized that the model was initially learning very slowly with a batch size of 512 and a learning rate of .0001.
In order to speed up the training process and prevent the model from getting stuck at a local minimum I decided to lower the batch
size down to 64 and increase the learning rate to .01. Since such a drastic change might cause the model to now learn too quickly
and not converge I also decided to add learning rate decay to the optimizer with a gamma of 0.95 every 25 epochs.


---

&#169; 2019 Georgia Tech
