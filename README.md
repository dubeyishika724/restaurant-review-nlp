# Restaurant Review Sentiment Analysis (NLP)

This project implements a Natural Language Processing (NLP) based sentiment analysis system to classify restaurant customer reviews as Positive, Neutral, or Negative.

## Objective
To analyze customer feedback using NLP techniques and machine learning models in order to understand customer sentiment and support data-driven business decisions.

## Dataset
The dataset consists of restaurant customer reviews and ratings. Ratings are converted into sentiment labels:
- Rating ≥ 4 → Positive  
- Rating = 3 → Neutral  
- Rating ≤ 2 → Negative  

## Technologies Used
- Python  
- Pandas, NumPy  
- NLTK  
- TF-IDF Vectorization  
- Scikit-learn  
- Matplotlib, Seaborn  

## Methodology
- Text preprocessing (lowercasing, stopword removal, lemmatization)
- Feature extraction using TF-IDF
- Model training using Naive Bayes and Logistic Regression
- Model evaluation using accuracy and confusion matrix

## Results
Logistic Regression performed better than Naive Bayes in sentiment classification and achieved higher accuracy.

## Future Improvements
- Aspect-based sentiment analysis  
- Deep learning models (LSTM / BERT)  
- Real-time review analysis
