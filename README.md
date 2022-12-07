# Process of Solving ML Problems
Assignment for Code Op.
Based on questions and answers from https://www.springboard.com/blog/data-science/machine-learning-interview-questions/.

# Machine Learning Interview Questions: Algorithms/Theory
Machine learning interview questions about ML algorithms will test your grasp of the theory behind machine learning.

### Q1: What’s the trade-off between bias and variance?
* __Bias-Variance Tradeoff__ balances the two--increasing one decreases the other. Finds the best model with the least amount of error.
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

### Q6: What is Bayes’ Theorem? How is it useful in a machine learning context

### Q7: Why is “Naive” Bayes naive?

### Q8: Explain the difference between L1 and L2 regularization.

### Q9: What’s your favorite algorithm, and can you explain it to me in less than a minute?

### Q10: What’s the difference between Type I and Type II error?

### Q11: What’s a Fourier transform?

### Q12: What’s the difference between probability and likelihood?

### Q13: What is deep learning, and how does it contrast with other machine learning algorithms?

### Q14: What’s the difference between a generative and discriminative model?

## Q15: What cross-validation technique would you use on a time series dataset?

## Q16: How is a decision tree pruned?

## Q17: Which is more important to you: model accuracy or model performance?

## Q18: What’s the F1 score? How would you use it?

## Q19: How would you handle an imbalanced dataset?

## Q20: When should you use classification over regression?

## Q21: Name an example where ensemble techniques might be useful.


## Q22: How do you ensure you’re not overfitting with a model?

## Q23: What evaluation approaches would you work to gauge the effectiveness of a machine learning model?

## Q24: How would you evaluate a logistic regression model?

## Q25: What’s the “kernel trick” and how is it useful?

# Machine Learning Interview Questions: Programming
These machine learning interview questions test your knowledge of programming principles you need to implement machine learning principles in practice. Machine learning interview questions tend to be technical questions that test your logic and programming skills: this section focuses more on the latter.

## Q26: How do you handle missing or corrupted data in a dataset?

## Q27: Do you have experience with Spark or big data tools for machine learning?

## Q28: Pick an algorithm. Write the pseudo-code for a parallel implementation.

## Q29: What are some differences between a linked list and an array?

## Q30: Describe a hash table.

## Q31: Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?

## Q32: Given two strings, A and B, of the same length n, find whether it is possible to cut both strings at a common point such that the first part of A and the second part of B form a palindrome.

## Q33: How are primary and foreign keys related in SQL?

## Q34: How does XML and CSVs compare in terms of size?

## Q35: What are the data types supported by JSON? 

## Q36: How would you build a data pipeline?

# Machine Learning Interview Questions: Company/Industry Specific
These machine learning interview questions deal with how to implement your general machine learning knowledge to a specific company’s requirements. You’ll be asked to create case studies and extend your knowledge of the company and industry you’re applying for with your machine learning skills.

## Q37: What do you think is the most valuable data in our business? 

## Q38: How would you implement a recommendation system for our company’s users?

## Q39: How can we use your machine learning skills to generate revenue?

## Q40: What do you think of our current data process?

# Machine Learning Interview Questions: General Machine Learning Interest
This series of machine learning interview questions attempt to gauge your passion and interest in machine learning. The right answers will serve as a testament to your commitment to being a lifelong learner in machine learning.

## Q41: What are the last machine learning papers you’ve read?

## Q42: Do you have research experience in machine learning?

## Q43: What are your favorite use cases of machine learning models?

## Q44: How would you approach the “Netflix Prize” competition?

## Q45: Where do you usually source datasets?

## Q46: How do you think Google is training data for self-driving cars?

## Q47: How would you simulate the approach AlphaGo took to beat Lee Sedol at Go?

## Q48: What are your thoughts on GPT-3 and OpenAI’s model?

## Q49: What models do you train for fun, and what GPU/hardware do you use?

## Q50: What are some of your favorite APIs to explore? 

## Q51: How do you think quantum computing will affect machine learning?
