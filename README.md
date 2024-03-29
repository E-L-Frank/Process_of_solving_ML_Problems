# Process of Solving ML Problems
Assignment for Code Op.
Based on questions and answers from https://www.springboard.com/blog/data-science/machine-learning-interview-questions/.

# Machine Learning Interview Questions: Algorithms/Theory
Machine learning interview questions about ML algorithms will test your grasp of the theory behind machine learning.

### Q1: What’s the trade-off between bias and variance?
__Bias-Variance Tradeoff__ balances the two--increasing one decreases the other. Finds the best model with the least amount of error.
* __Bias__ the difference between average predictions and true values
    * Are the changes between the model and reality due to the model not having enough data or is it due to undue influence on the model? 
    * *i.e. is the simplifying assumptions made by the model to make the target function easier to approximate.
* __Variance__ the variability of our predictions (how spread out your model predictions are)
    * If given more data does the model change or does it stay the same? 
    * *i.e. is the amount that the estimate of the target function will change, given different training data.
* __Underfitting__ (overly simplistic) 
    * the model has a bias problem
    * underfits
    * high error on both test and train data
* __Overfitting__ (too complicated) 
    * the model has too much variance
    * models the noise from the input data
    * low error on the training data and high on the test
![alt text](https://scontent-iad3-1.xx.fbcdn.net/v/t1.6435-9/189902238_306440417637742_8564280170340073765_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=9267fe&_nc_ohc=VxIKKACl4lUAX9LEWcR&_nc_ht=scontent-iad3-1.xx&oh=00_AfCOLQr-O3CWGnBLahuCmGEONvoK8lGnf-nNadGkgn1hYg&oe=63B850BD)


### Q2: What is the difference between supervised and unsupervised machine learning?
* **Supervised learning** includes the correct results (targets) during training
    * precategorized data
* **Unsupervised learning** does not include the correct result during the training phase
![alt text](https://miro.medium.com/max/1030/1*zWBYt9DQQEf_XxXWLA2tzQ.webp)
* Classification: target is discrete (binary or categorical)
* Regression: target is con't and numerical
* Clustering: grouping unlabeled examples
* Dimensionality reduction: reducing the amount of random variables in a problem by obtaining a set of principal variables

### Q3: How is KNN different from k-means clustering?
* KNN Algorithm is based on feature similarity and K-means refers to the division of objects into clusters (such that each object is in exactly one cluster, not several).

* **K-Nearest Neighbors** is a supervised classification algorithm
   * need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part)
![alt text](https://miro.medium.com/max/1400/0*34SajbTO2C5Lvigs.webp)
         _https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4_
* **K-means clustering** is an unsupervised clustering algorithm
   * requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points
![alt text](https://lh5.googleusercontent.com/rhGbEMw6SBxZCarC2ewbhYIK9HbjMDWxcPbSx8MlAOoVp9jDaT5-vTLWyR7qiXlW3iBQ2GOty8HFPCVPTqtGPptuy4GVxbXH5ePvZP8IbVHtwfKb7iaCQQGuHiM3yVHuC24QO2cH)

### Q4: Explain how a ROC curve works.
![alt text](https://upload.wikimedia.org/wikipedia/commons/1/13/Roc_curve.svg)

* The ROC (receiver operating characteristic) curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).
* Created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.

![alt text](https://upload.wikimedia.org/wikipedia/commons/5/5a/Sensitivity_and_specificity_1.01.svg)

### Q5: Define precision and recall.
* **Precision**: how *valid* the results are
   * More false negatives 
* **Recall**: how many *relevant* elements are retrieved
   * More false positives 
![alt text](https://static.wixstatic.com/media/6f3565_48ca6fb7925d46608e4845254dd2bd65~mv2.png)
*N.B. the bottom chart is a confusion matrix*

### Q6: What is Bayes’ Theorem? How is it useful in a machine learning context
The probability of an event, based on prior knowledge of conditions that might be related to the event
* It lets you take the test results and correct for the “skew” introduced by false positives. You get the real chance of having the event

![alt text](https://miro.medium.com/max/1400/1*CnoTGGO7XeUpUMeXDrIfvA.webp)

### Q7: Why is “Naive” Bayes naive?
**Naive Bayes classifiers** are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) *independence assumptions* between the features 
* As a Quora commenter put it whimsically, a Naive Bayes classifier that figured out that you liked pickles and ice cream would probably naively recommend you a pickle ice cream
* Has applications ex. int text mining

### Q8: Explain the difference between L1 and L2 regularization.
The main intuitive difference between the L1 and L2 regularization is that L1 regularization tries to estimate the median of the data while the L2 regularization tries to estimate the mean of the data to avoid overfitting.
* **Regularization** controls the model complexity by penalizing higher terms in the model. If a regularization terms is added, the model tries to minimize both loss and complexity of model.
* **Loss function** (cost/error function): a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. Many different types
* **L1 regularization** (LASSO-least absolute shrinkage and selection operator-Regression): adds the “absolute value of magnitude” of the coefficient as a penalty term to the loss function
   * **L1 regularization penalizes |weight|**
   * L1 regularization forces the **weights** of uninformative features to be zero by substracting a small amount from the weight at each iteration and thus making the weight zero, eventually.
* **L2 Regularization** (Ridge Regression): Ridge regression adds the “squared magnitude” of the coefficient as the penalty term to the loss function.
   * **L2 regularization penalizes (weight)²**
   * L2 regularization forces weights toward zero but it *does not make them exactly zero.* L2 regularization acts like a force that removes a small percentage of weights at each iteration. Therefore, weights will never be equal to zero.
![alt text](https://miro.medium.com/max/1400/0*tATGj-F5jlQU80GE.webp)

![alt text](https://miro.medium.com/max/1100/1*-LydhQEDyg-4yy5hGEj5wA.webp)

lambda value | alpha value | Result
--- | --- | ---
lambda == 0 | alpha = any value | No regularization. alpha is ignored.
lambda > 0 | alpha == 0 | Ridge Regression
lambda > 0 | alpha == 1 | LASSO
lambda > 0 | 0 < alpha < 1| Elastic Net Penalty

**Good Sources**
* *https://towardsdatascience.com/l1-and-l2-regularization-explained-874c3b03f668*
* *https://medium.com/analytics-vidhya/l1-vs-l2-regularization-which-is-better-d01068e6658c*
* Explination of regularization paramaters alpha and lambda: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/lambda.html

### Q9: What’s your favorite algorithm, and can you explain it to me in less than a minute?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
Answer: Interviewers ask such machine learning interview questions to test your understanding of how to communicate complex and technical nuances with poise and the ability to summarize quickly and efficiently. While answering such questions, make sure you have a choice and ensure you can explain different algorithms so simply and effectively that a five-year-old could grasp the basics!

### Q10: What’s the difference between Type I and Type II error?
* **Type I error**: is a false positive i.e. claiming something has happened when it hasn’t.
   * Rejects a null hypothesis that is *true* in the population
* **Type II error** is a false negative i.e. claiming nothing is happening when in fact something is.
   * Accepts (fails to reject) a null hypothesis that is *false* in the population
![alt text](https://static.wingify.com/gcp/uploads/sites/3/2020/12/type-1-and-type-2-errors.png?tr=w-1366)

### Q11: What’s a Fourier transform?
A Fourier transform (FT) is a mathematical transform that decomposes functions into frequency components, which are represented by the output of the transform as a function of frequency.
   * The Fourier Transform takes a specific viewpoint: What if any signal could be filtered into a bunch of circular paths?
   * Converts a function in the amplitude vs. time (f(t)) domain to the amplitude vs. frequency domain (F(w))
#### Example
![alt text](https://insightincmiami.org/wp-content/uploads/2019/06/fourier-transform-car-engine-example.jpg)
#### Math
![alt text](https://betterexplained.com/wp-content/webp-express/webp-images/uploads/images/fourier-explained-20121219-224649.png.webp)
* N = number of time samples we have
* n = current sample we're considering (0 .. N-1)
* xn = value of the signal at time n
* k = current frequency we're considering (0 Hertz up to N-1 Hertz)
* Xk = amount of frequency k in the signal (amplitude and phase, a complex number)
* The 1/N factor is usually moved to the reverse transform (going from frequencies back to time). This is allowed, though I prefer 1/N in the forward transform since it gives the actual sizes for the time spikes. You can get wild and even use  on both transforms (going forward and back creates the 1/N factor).
* n/N is the percent of the time we've gone through. 2 * pi * k is our speed in radians / sec. e^-ix is our backwards-moving circular path. The combination is how far we've moved, for this speed and time.
* The raw equations for the Fourier Transform just say "add the complex numbers". Many programming languages cannot handle complex numbers directly, so you convert everything to rectangular coordinates and add those.

**Amazing website**: https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/

### Q12: What’s the difference between probability and likelihood?
**Probability** 

Probability = P(data| distribution)-->measures how probable the data come from the specific distribution
   * Measures the fitness of data given a specific distribution
   * Probability is used to estimate how probable a sample or groups of samples are from a distribution based on a given distribution.
   * Probability refers to the area under curve on the distribution curve. The higher the value, the more probable that the data come from this distribution.  

**Likelihood**

Likelihood = L(distribution| data)-->measures how probable a specific distribution fits the given data
   * Measures the fitness of a model given some data (i.e. how well a model fits the data)
   * refers to a specific point on the distribution curve
   * The lower the likelihood, the worse the model fits the data.
![alt text](https://miro.medium.com/max/1400/1*pKLjnStE9odh6oePDoLCdA.webp)
### Q13: What is deep learning, and how does it contrast with other machine learning algorithms?
* **Neural Network**: Composed of node layers (1 input, 1 or more hidden layers, and 1 out put). Each node connects to another and has an associated model with four main components (inputs, weights, a bias or threshold, and an output). Supposed to mimic human brains (as if the nodes were neurons).
![alt text](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png)
*Helful link*: https://www.ibm.com/in-en/cloud/learn/neural-networks
* **Deep learning**: A subset of machine learning-->a neural network with three or more layers.
   * The "deep" in deep learning refers to the depth of layers in a neural network 
![alt text](https://1.cms.s81c.com/sites/default/files/2021-04-22/Russian%20Nesting%20Dolls.png)
### Q14: What’s the difference between a generative and discriminative model?
* **Generative model**: learns *categories* of data
* **Discriminative model**: learns the *distinction* between different categories of data *Discriminative models will generally outperform generative models on classification tasks*
![alt text](https://d2mk45aasx86xg.cloudfront.net/Supervised_Learning_Cheatsheet_290e086e75.webp)
### Q15: What cross-validation technique would you use on a time series dataset?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
Answer: Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data—it is inherently ordered by chronological order. If a pattern emerges in later time periods, for example, your model may still pick up on it even if that effect doesn’t hold in earlier years!

You’ll want to do something like forward chaining where you’ll be able to model on past data then look at forward-facing data.

Fold 1 : training [1], test [2]
Fold 2 : training [1 2], test [3]
Fold 3 : training [1 2 3], test [4]
Fold 4 : training [1 2 3 4], test [5]
Fold 5 : training [1 2 3 4 5], test [6]

### Q16: How is a decision tree pruned?
* **Decision Tree**: a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.
* A nonlinear ML model that can be very interpretable
![alt text](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm.png)
* **Pruning**: remove branches (nodes/neurons) that aren't critical/ are redundant in order to increase the accuracy and reduce the complexity of the model.
   * Pruning can happen bottom-up and top-down
   * Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease predictive accuracy, keep it pruned. While simple, this heuristic actually comes pretty close to an approach that would optimize for maximum accuracy.
![alt text](https://upload.wikimedia.org/wikipedia/commons/2/23/Before_after_pruning.png)
### Q17: Which is more important to you: model accuracy or model performance?
* **Accuracy paradox**: the paradoxical finding that accuracy (proportion of correctness) is not a good metric for predictive models when classifying in predictive analytics (because a simple model may have a high level of accuracy but be too crude to be useful). Precision and recall are better measures in such cases.
* Model accuracy is only a subset of model performance, and at that, a sometimes misleading one.

### Q18: What’s the F1 score? How would you use it?
The harmonic mean between precision and recall. A classification metric that combines both recall and precision. The closer to 1 the better.
![alt text](https://hasty.ai/media/pages/docs/mp-wiki/metrics/f-beta-score/55c23cb495-1654855011/snimok-ekrana-2022-06-10-v-12-51-17.webp)
### Q19: How would you handle an imbalanced dataset?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
Answer: An imbalanced dataset is when you have, for example, a classification test and 90% of the data is in one class. That leads to problems: an accuracy of 90% can be skewed if you have no predictive power on the other category of data! Here are a few tactics to get over the hump:

Collect more data to even the imbalances in the dataset.
Resample the dataset to correct for imbalances.
Try a different algorithm altogether on your dataset.
What’s important here is that you have a keen sense for what damage an unbalanced dataset can cause, and how to balance that.

*Helpful Link*: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

### Q20: When should you use classification over regression?
* **Classification**: Output is discrete and has categorical/class labels
* **Regression**: Output is continuous and numerical

### Q21: Name an example where ensemble techniques might be useful.
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
**NEED EXAMPLES OF TYPES OF THESE TECHNIQUES AND HOW THEY WORK**

* **Ensemble techniques**: Ensemble techniques use a combination of learning algorithms to optimize better predictive performance. They typically reduce overfitting in models and make the model more robust (unlikely to be influenced by small changes in the training data).

### Q22: How do you ensure you’re not overfitting with a model?
This is a simple restatement of a fundamental problem in machine learning: the possibility of overfitting training data and carrying the noise of that data through to the test set, thereby providing inaccurate generalizations.

There are three main methods to avoid overfitting:

* Keep the model simpler: reduce variance by taking into account fewer variables and parameters, thereby removing some of the noise in the * training data.
* Use cross-validation techniques such as k-folds cross-validation.
   * k-folds cross-validation: https://machinelearningmastery.com/k-fold-cross-validation/   
* Use regularization techniques such as LASSO that penalize certain model parameters if they’re likely to cause overfitting.

### Q23: What evaluation approaches would you work to gauge the effectiveness of a machine learning model?
1. Split the dataset into training and test sets/use cross-validation techniques to further segment the training and test sets 
2. Implement a choice selection of performance metrics (ex. the F1 score, the accuracy, and the confusion matrix) 
   * Performance metrics: https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/

### Q24: How would you evaluate a logistic regression model?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
Demonstrate an understanding of what the typical goals of a logistic regression are (classification, prediction, etc.) and bring up a few examples and use cases.

### Q25: What’s the “kernel trick” and how is it useful?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)

# Machine Learning Interview Questions: Programming
These machine learning interview questions test your knowledge of programming principles you need to implement machine learning principles in practice. Machine learning interview questions tend to be technical questions that test your logic and programming skills: this section focuses more on the latter.

### Q26: How do you handle missing or corrupted data in a dataset?
Drop the rows/columns (isnull(), dropna()) or replace with a different value (fillna())

### *Q27: Do you have experience with Spark or big data tools for machine learning?
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
### *Q28: Pick an algorithm. Write the pseudo-code for a parallel implementation.
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)

### Q29: What are some differences between a linked list and an array?
![alt text](https://1.bp.blogspot.com/-SKU5oSgKyXg/V5DC9yymoMI/AAAAAAAABMo/farel9q9Uo0WZIOFLZ2Z_2gxbaXqdxoqwCLcB/s1600/array-linked%2Blist.PNG)
### Q30: Describe a hash table.
* **Hash Table**: a data structure that produces an associative array. A key is mapped to certain values through the use of a hash function. They are often used for tasks such as database indexing. It is an abstract data type that maps keys to values. A hash table uses a hash function to compute an index.
*Example*
![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Hash_table_3_1_1_0_1_0_0_SP.svg/1920px-Hash_table_3_1_1_0_1_0_0_SP.svg.png)
### Q31: *Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?*
Seaborn, Folium, PlyPlot, Matplotlib
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
### Q32: *Given two strings, A and B, of the same length n, find whether it is possible to cut both strings at a common point such that the first part of A and the second part of B form a palindrome.*
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
### Q33: How are primary and foreign keys related in SQL?
* **Primary key**: consists of one or more columns whose data contained within are used to uniquely identify each row in the table
* **Foreign key**: a set of one or more columns in a table that refers to the primary key in another table and provides a link between data in two tables
![alt text](https://www.thecrazyprogrammer.com/wp-content/uploads/2019/04/Difference-between-Primary-Key-and-Foreign-Key-1024x672.gif)
### Q34: How does XML and CSVs compare in terms of size?
XML>CSV
* **Extensible Markup Language (XML)**: a markup language and file format for storing, transmitting, and reconstructing arbitrary data.
* **Comma-separated values (CSV) file**: a delimited text file that uses a comma to separate values. 

In practice, XML is much more verbose than CSVs are and takes up a lot more space. CSVs use some separators to categorize and organize data into neat columns. XML uses tags to delineate a tree-like structure for key-value pairs. You’ll often get XML back as a way to semi-structure data from APIs or HTTP responses. In practice, you’ll want to ingest XML data and try to process it into a usable CSV. This sort of question tests your familiarity with data wrangling sometimes messy data formats. 

### Q35: What are the data types supported by JSON? 
There are six basic JSON (JavaScript Object Notation) datatypes you can manipulate: strings, numbers, objects, arrays, booleans, and null values. 

### *Q36: How would you build a data pipeline?*
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
# Machine Learning Interview Questions: Company/Industry Specific
These machine learning interview questions deal with how to implement your general machine learning knowledge to a specific company’s requirements. You’ll be asked to create case studies and extend your knowledge of the company and industry you’re applying for with your machine learning skills.
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
### *Q37: What do you think is the most valuable data in our business? 
### Q38: How would you implement a recommendation system for our company’s users?
### Q39: How can we use your machine learning skills to generate revenue?
### Q40: What do you think of our current data process?

# Machine Learning Interview Questions: General Machine Learning Interest
This series of machine learning interview questions attempt to gauge your passion and interest in machine learning. The right answers will serve as a testament to your commitment to being a lifelong learner in machine learning.
![alt text](https://miro.medium.com/max/1400/1*EWl_Flr1FKtD42270Iosog.webp)
### *Q41: What are the last machine learning papers you’ve read?
### Q42: Do you have research experience in machine learning?
### Q43: What are your favorite use cases of machine learning models?
### Q44: How would you approach the “Netflix Prize” competition?
### Q45: Where do you usually source datasets?
### Q46: How do you think Google is training data for self-driving cars?
### Q47: How would you simulate the approach AlphaGo took to beat Lee Sedol at Go?
### Q48: What are your thoughts on GPT-3 and OpenAI’s model?
### Q49: What models do you train for fun, and what GPU/hardware do you use?
### Q50: What are some of your favorite APIs to explore? 
### Q51: How do you think quantum computing will affect machine learning?
