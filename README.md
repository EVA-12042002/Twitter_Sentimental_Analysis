# Twitter_Sentimental_Analysis

                                         ![image](https://github.com/EVA-12042002/Twitter_Sentimental_Analysis/assets/129527829/6ca88935-2543-414b-90b1-1263bf61360b)

Overview
This project aims to perform sentiment analysis on Twitter data using various Naive Bayes classifiers: Naive Bayes, Complement Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes. Additionally, the project includes Receiver Operating Characteristic (ROC) curves to visualize the performance of each model.

Sentiment analysis is a crucial natural language processing (NLP) task that classifies text data into positive, negative, or neutral sentiments. Understanding public sentiment on social media platforms like Twitter has applications in brand monitoring, customer feedback analysis, and more.


Contents
1]sentiment_analysis.ipynb: Jupyter Notebook containing the code for sentiment analysis using various Naive Bayes classifiers and ROC curve visualization.

2]twitter_data.csv: The dataset used for this project, containing Twitter text data and corresponding sentiment labels.

3]twitter.jpg: An image related to Twitter, used for illustration purposes in this README.

Dependencies
To run the code in this repository, you will need the following Python libraries:

*Pandas
*NumPy
*Matplotlib
*Scikit-learn
*NLTK (Natural Language Toolkit)

Usage
i]Open and run the Jupyter Notebook sentiment_analysis.ipynb to explore the code, perform sentiment analysis using Naive Bayes classifiers, and visualize the ROC curves.

ii]The key steps involved in this project include:

*Data Preprocessing: Cleaning and preparing the Twitter data for analysis, including text tokenization and feature extraction.

*Naive Bayes Classifiers: Implementing Naive Bayes, Complement Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes classifiers for sentiment analysis.

*Model Evaluation: Evaluating the performance of each classifier using metrics like accuracy, precision, recall, and F1-score.

*ROC Curves: Visualizing the performance of each model using ROC curves and calculating the area under the ROC curve (AUC).

iii]Customize the code, experiment with different preprocessing techniques, or try other NLP models to improve sentiment analysis accuracy.

Results
The results of sentiment analysis using various Naive Bayes classifiers and ROC curves are summarized as follows:

Based on the analysis of the AUC-ROC curve for the three models, Complement Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes, it can be concluded thaMultinomial Naive Bayes and Bernoulli Naive Bayes are better models for predicting the sentiment of people on Twitter for Hate Speech Detection compared to Complement Naive Bayes.

The AUC-ROC curve is a graphical representation of the performance of a classification model. It shows the trade-off between the true positive rate (sensitivity) and the false positiverate (1 - specificity) for different classification thresholds. A higher AUC (Area Under the Curve) indicates better performance, where an AUC of 1 represents a perfect classifier. Based on the AUC-ROC curve, Multinomial Naive Bayes and Bernoulli Naive Bayes have higher AUC values compared to Complement Naive Bayes. This implies that they have better discriminatory power and can effectively differentiate between hate speech and non-hate
speech tweets.

Therefore, Multinomial Naive Bayes and Bernoulli Naive Bayes are recommended models for sentiment analysis of Twitter data for Hate Speech Detection. You can further evaluate these models by considering other evaluation metrics such as accuracy, precision, recall, and F1-score to get a comprehensive understanding of their performance.

Model Performance: -
Multinomial Naive Bayes: Achieved an accuracy score of 0.85 and an F1 score of 0.83.
Bernoulli Naive Bayes: Obtained an accuracy score of 0.81 and an F1 score of 0.79.
Complement Naive Bayes: Attained an accuracy score of 0.68 and an F1 score of 0.65.

Accuracy Comparison: -
Among the three models, Multinomial Naive Bayes demonstrated the highest accuracy,
correctly classifying 85% of the tweets in the test set. It outperformed both Bernoulli Naive
Bayes (81% accuracy) and Complement Naive Bayes (68% accuracy).

F1 Score Comparison: -
The F1 scores, which take into account both precision and recall, further support the
superiority of Multinomial Naive Bayes (F1 score of 0.83) over the other models. Bernoulli
Naive Bayes achieved an F1 score of 0.79, while Complement Naive Bayes lagged behind
with a score of 0.65.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

1. https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
  
2. https://medium.com/geekculture/how-to-deal-with-class-imbalances-in-python-
960908fe0425
   
3. https://medium.com/analytics-vidhya/how-to-improve-logistic-regression-b956e72f4492
   
4. https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
   
5. https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
   
6. https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

Contact
If you have any questions or suggestions, feel free to contact me at [evangelinpriyanka12@gmail.com].


   



