# Adaptive-Neural-Network-from-scrach
Adaptive Neural Network from scratch: Built using Gradient Descent, fundamentals of Backpropagation and Oriented Object Programming.

The python script contains all the classes that define the neuron class, the layer class, the network class and the main 
part that excecutes the code. All you should do is to keep this file with the generated data and run the code.
The excecution takes several minutes to finish since the complexity of the execution is important. This is due to the number 
of neurons and the large numbre of samples utilized during the training.
# Data:
Mnist dataset
# Parameters:
By comparing different learning rates, I decided to choose η=0.008 and the number of epochs to be 500. Choosing this value for η engenders an adequate learning rate. Consequently, the weights don’t change fast which makes the learning slow. At the beginning the algorithm did a lot of mistakes. However, in the following epochs the learning becomes more efficient and effective (error fraction of the test and training converge to a very small value). The number of epochs chosen is 500 because at this stage the model converges. In addition, the number of epochs should not be too big to avoid overfitting since above 500 epochs I noticed an overfitting.  For the weights, I initialized them to random values between -a and a with a is √(3/n)  with n is the number of inputs to the neuron. I tried Xavier initialization too, it gives good results but this initialization gives better results.
For the momentum I used  α=0.8 to make the stochastic gradient converge and avoid the local minimums. The alpha values give better results when they are in the range of 0.75 and 0.9 and 0.8 is the best that I found.
For the output thresholds H=0.95 and L=0.05 since I wanted to allow 5% of error to the neuron.
# Results:
![image](https://user-images.githubusercontent.com/74180896/168212142-f219a113-e53a-4609-9913-4bf63ccb7c03.png)
![image](https://user-images.githubusercontent.com/74180896/168212055-52d26477-6a1c-496f-9d90-a14221fea4cc.png)
![image](https://user-images.githubusercontent.com/74180896/168212067-e689ca70-e2d9-4eb6-bbdc-94dca1aa08b4.png)

