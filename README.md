# Sentiment-Analysis
Thank you Wisesight for the dataset for the sentiment Analysis

- The rough architecture in the notebook are 

	1. XGBoost Model
	
	Data --> Remove Punctuation --> Word Segmentation --> Word Embedding --> Gradient boosting --> Output
	
	This model is also use in the API (API.py) deply using flask.

	2. Deep Learning Models

	2.1 CNN

	Data --> Remove Punctuation --> Word Segmentation --> Word Embedding --> CNN Bigram  ---> Concat --> Dense --> Output
	  								     --> CNN Trigram

	2.2 LSTM CNN

	Data --> Remove Punctuation --> Word Segmentation --> Word Embedding --> Bidirectional LSTM --> CNN Bigram  ---> Concat --> Dense --> Output
	  								     			    --> CNN Trigram

	2.3 CNN LSTM

	Data --> Remove Punctuation --> Word Segmentation --> Word Embedding --> CNN Bigram  ---> Concat ---> BidirectionalLSTM --> Dense --> Output
	  								     --> CNN Trigram

