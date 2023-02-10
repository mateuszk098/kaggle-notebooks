# **THE MACHINE LEARNING LANDSCAPE - QUESTIONS**

1. **Define what is Machine Learning.**  

   Machine Learning is the creation of systems that can "learn" from data. Through the "learn" process, we mean obtaining better and better results in something assignment (problem), considering some efficiency measures simultaneously.

2. **Specify four different problems with which the machine learning process performs best.**  

    Machine Learning is the best solution for complex problems for which algorithmic solutions don't exist. For problems for which we have to define a long list of conditions. For problems where the solution has to be matched to data that change over time. For teaching people (e.g. through data mining).

3. **What is a labeled training set?**  

    The labeled training set is a data set which contains the predicted solution (label) for each sample.

4. **Specify the two most popular applications of supervised learning.**  

    The two most popular applications of supervised learning are regression and classification.

5. **Specify four popular applications of unsupervised learning.**  

    These are clustering, visualisation, dimensionality reduction and association rule learning.

6. **What kind of machine learning algorithm do you use to create a robot that traverses unfamiliar areas?**  

    The best is probably reinforcement learning. This strategy is best adapted for such problems.

7. **What kind of machine learning algorithm do you use to split consumers into several different groups?**

    When we don't know what types of groups we want, we can use clustering (unsupervised learning) to split people, for example, by hobbies. When we know what groups we want, we can provide people from different groups and use a classifier (supervised learning) to place them in appropriate groups.

8. **Is spam classification a supervised or unsupervised mechanism of machine learning?**  

    Spam classification is a classic problem of supervised learning. We provide messages with labels (spam or not).

9. **What is an online learning?**  

    In online learning process, the system is trained up to date by sequentially providing data. Thanks to that, the system may quickly adapt to changing data.

10. **What is an out-of-core learning?**  

    In out-of-core learning, we train the system with a huge data set which doesn't fit in the device memory. In this algorithm, the data set is split into several mini-sets, and the online learning process is used for these sets.

11. **In which machine learning algorithm the similarity measure is required?**  

    The similarity measure is required in instance-based learning. The system learns instances by heart. Then, the algorithm uses similarity measure for a new sample to find similar samples and calculate prediction.

12. **Explain the difference between the model's parameter and the algorithm's hyperparameter.**  

    The model contains at least one parameter, which specifies the predicted factor considering a new sample (e.g. slope of the linear model). The algorithm tries to find the optimal values of these parameters. In contrast, the hyperparameter is the parameter of the algorithm itself (e.g. degree of regularisation).

13. **What are model-based learning algorithms searching for? What is the most often strategy used by them? How do they carry predictions?**  

    Model-based learning algorithms search for optimal model parameter values so that the model will predict new samples well. Usually, models learn through minimalising the cost function. To get predictions, we provide features of a new sample to the prediction function.

14. **Specify four main problems related to machine learning.**  

    There problems are, e.g. lack of data, low quality of data, useless features, excessively simplified models, and too complex models.

15. **What does it mean that the model handles train data very well but incorrectly generalises new samples?**  

    It means that the model is probably overfitted. To handle that, we can provide more training data and simplify the model (by choosing different algorithms, decreasing the number of using parameters or features, and carrying a regularisation) or reduce training samples noise.

16. **What is the test set, and why should we use it?**  

    The test set is used to estimate the generalisation error in the case of new samples.

17. **What is a validation set?**  

    The validation set is used to compare different models and adjust hyperparameters.

18. **What is a train-dev set? When do we need such a set? What to use it?**

    The train-dev set is used in the case of the risk of a mismatch of the train set and validation or test sets. The train-dev set is a training subset which was not used during learning. The model is tested using both the train-dev set and the validation set. When the model handles with training set well but not the train-dev set, that probably means the model is overfitted. If the model handles with the training set and train-dev set but not with the validation set, this probably means that data are unfitted.

19. **What is the risk of adjusting the hyperparameters using a test set?**  

    If we adjust hyperparameters using the test set, we risk overfitting the model toward this set, and the generalisation error will be optimistically small.
