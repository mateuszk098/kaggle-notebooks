# **SUPPORT VECTOR MACHINES - QUESTIONS**

### :small_orange_diamond: **1. What is the fundamental idea behind Support Vector Machines?**

The fundamental idea behind Support Vector Machines (SMV) is to find the widest possible "street" between classes. In other words, the crucial concept is to find such a decision boundary that the instances from different classes are as far as possible from that boundary. This "street" is called a margin. We distinguish between hard margin classification and soft margin classification. In hard margin classification, none sample cannot exceed the margin. On the other hand, in soft margin classification, the margin may be exceeded by some samples. The second idea behind SVM is to use kernels when training on a nonlinear dataset. SVMs may also be used for regression and novelty detection.

### :small_orange_diamond: **2. What is a support vector?**

A support vector is a sample which is located on the "street", including its border. These vectors entirely determine the decision boundary. All instances that are not the support vectors have not to influence on the decision boundary. We can add new samples and remove some, but the decision boundary remains the same as long as these samples are not support vectors. Computing the predictions with a kernelized SVM only involves the support vectors.

### :small_orange_diamond: **3. Why do we need to scale input data before we use the SVMs?**

SVM tries to fit the widest possible "street" between classes. When samples are not scaled, that task is more complicated. SVM may neglect small features compared to large ones.

### :small_orange_diamond: **4. Can confidence scores be calculated with SVM when classifying an instance? Can we calculate the probability that the sample belongs to a specific class?**

We can use the `decision_function()` method to get confidence scores. These scores represent the distance between the instance and the decision boundary. Nevertheless, confidence scores cannot be directly converted into probabilities. If we set `probability=True` in the `SVC()` constructor, then at the end of the training, it will use cross-validation to generate out-of-sample scores for the training samples. Next, it will train a `LogisticRegression` model to map these scores to estimated probabilities. The `predict_proba()` and `predict_log_proba()` methods will then be available.

### :small_orange_diamond: **5. Suppose we train the SVM model for a dataset composed of millions of examples and hundreds of features. Should we use a primal problem solution or a dual problem solution?**

We should use the primal problem solution. The dual problem solution is faster when the number of samples is smaller than the number of features.

### :small_orange_diamond: **6. Suppose we trained SVM classifier with an RBF kernel. We suspect that the model is underfitted. What should we do with gamma and C parameters?**

If an SVM classifier trained with an RBF kernel underfits the training set, there might be too much regularization. To decrease it, you need to increase `gamma` or `C` (or both).
