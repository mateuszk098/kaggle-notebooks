### **Training Linear Models Questions**

1. **Which linear regression model should we use if the training set contains millions of features?**  
   We can use stochastic gradient descent or mini-batch gradient descent. We should not use the normal equation because, with the feature number increase, computational complexity increases very rapidly.

2. **Suppose features in the training set have different scales. For which algorithms can it be harmful? How to handle this?**  
   When the features have different scales, the cost function will be stretched (expanded). Therefore gradient descent algorithms will need a lot of time to reach convergence. To solve this problem, we can scale these features before model learning. 

3. **Can gradient descent get stuck in the local minimum while learning the logistic regression model?**  
   The gradient descent algorithm cannot get stuck in the local minimum for logistic regression because the cost function is convex.

4. **Do all gradient descent algorithms lead to the same model if we give them enough time?**  
   When the optimization problem has a convex nature (linear or logistic regression) and the learning rate is not too big, then gradient descent algorithms reach a global minimum and will lead to similar models. But when the learning rate is relatively large, stochastic gradient descent and mini-batch gradient descent may not achieve real convergence.

5. **Suppose we use batch gradient descent algorithm and create a plot of validation error as a function of epoch. The error rate continues to rise. What does it mean? How to handle this?**  
   That is probably caused by a too large learning rate through which the algorithm becomes divergent. If the learning error rises too, then this is certainly that problem. We can decrease the learning rate, but if the learning error does not rise, this means the model is overfitted, and we should break learning.

6. **Is it a good idea to break mini-batch gradient descent algorithm immediately after the detection of the validation error increase?**  
   Both stochastic gradient descent and mini-batch gradient descent do not provide progress with each learning epoch because of randomness. Therefore it is not a good idea to break learning immediately. A better solution is to save the model in regular time intervals. Thus if the performance not rises for a long time, we can load the best model we saved earlier.

7. **Which gradient descent algorithm reaches the convergence fastest? Which of them actually reaches the convergence? How exactly can we provide a convergence in the remaining algorithms?**  
   Stochastic gradient descent is the fastest because, in each epoch, only one sample is analysed. Therefore, it reaches the global minimum the fastest. A similar situation is for mini-batch gradient descent if we have small mini-groups there. Nevertheless, the actual convergence will reach the batch gradient descent if we give it enough time.

8. **Suppose we use a polynomial regression. We draw a learning curve plot and notice a significant gap between learning and validation data errors. What does it mean? How to handle this?**  
   When the validation error is much larger than the learning error, this probably means we have an overfitted model. To handle this, we can decrease the polynomial degree, perform a regularization and add additional samples for the training set.

9. **Suppose we use the ridge regression method and notice the errors for the training set and validation set are almost equal and have a significant value. Does such a model has a big bias or variance? Should we increase or decrease the $\alpha$ parameter?**  
    When the errors for the training and validation sets are almost equal and large, this probably means we have an unfitted model with significant bias. We should reduce the $\alpha$ hyperparameter.

10. **Why should we use:**  
    a. **ridge regression instead of standard linear regression?**  
    b. **LASSO regression instead of ridge regression?**  
    c. **elastic net instead of LASSO?**  

    a. Usually, a model which contains some regularization works better than a not regularized model.  

    b. LASSO method uses $l_1$ regularization, which seeks to zero weights. This way, we get a sparse model with only the most significant weights. That is a good method if we suspect there only some features are essential. If we are not sure, we should use ridge regression.

    c. The elastic net is preferred over the LASSo because the second can behaves unpredictably in some situations (e.g. when features are correlated or when the dataset has more features than samples). 

11. **Suppose we want to classify photos taken day/night and indoors/outdoors. Should we use two logistic regression classifiers or one softmax classifier?**  
    We can use two logistic regression classifiers because the features are not mutually exclusive. There are four possible combinations.