# CarND-Traffic-Sign-Classifier

---

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png     "Grayscaling"
[image3]: ./architecture.png           "Architecture diagram"
[image4]: ./data/web/1.jpeg            "Traffic Sign 1"
[image5]: ./data/web/2.jpeg            "Traffic Sign 2"
[image6]: ./data/web/3.jpeg            "Traffic Sign 3"
[image7]: ./data/web/4.jpeg            "Traffic Sign 4"
[image8]: ./data/web/5.jpeg            "Traffic Sign 5"

## Rubric Points
I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view)
individually and describe how I addressed each point in my implementation.

---

Here is a link to my [project code](https://github.com/ricardosllm/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

A visualisation of the dataset can be found in the notebook,
in the `Exploratory Visualization of the Dataset` section.

Here's an example of the distribution of the number of images by classes

![alt text][image1]

### Design and Test a Model Architecture


As a first step, I decided to convert the images to grayscale
to remove light effects, this is specially important in the
case of traffic signs images due to large diference between
signs in direct sun ligh and signs in the shadow or even
eluminated by artfictial ligh in night conditions.

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so that the data has mean zero and equal variance.

I decided not to generate additional even though it would be useful here.
I might decide to expand this dataset in a future time.

Here's a diagram of the Model used, LeNet-5

![alt text][image3]

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 32x32x1   | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 14x14x6   | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x16  				|
| Flatten               | input 10x10x16, outputs 5x5x16                |
| Fully connected 		| input 400, ouputs 120        					|
| RELU                  | activation   									|
| Fully connected 		| input 120, ouputs 84        					|
| RELU                  | activation   									|
| Fully connected 		| input 84, ouputs 43        					|


#### Training

To train the model, I used tensorflow's `softmax_cross_entropy_with_logits`
function and `AdamOptimizer` for the optimizer.

I've used a batch size of `256`, this was enough to achieve the
required validation accuracy and it's small enough that it can
be trained in a above average laptop.

I've used a learning rate of `0.0015`, this value was achieved
by experimentation and turned out to be the best in its vicinity
considering the rest of the hyperparameters.

I've chosen to use `25` epochs for this training.
The reason I did not decide for a higher value is that,
with these hyperparameters, the accuracy was not really increasing
after 20 epochs. This can potentially be tuned in the future if other
hyperparameters like learning rate and batch size are changed.

#### Accuracy

My final model results were:
* validation set accuracy of **95.4%**
* test set accuracy of **92.5%**

I've taken the suggestion from the class to use the LeNet-5
architecture and achieved an accuracy of 89% on the first approch.

I then improved the data preprocessing and the accuracy increased
to around 91%, 92%

Finally I experimented with hyperparameters to acheive the desired
accuracy of > 93%

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]
