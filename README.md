# Bank Marketing Modeling

This README.md will act as technical documentation for the machine learning pipeline implemented on the Bank Marketing dataset from the UCI Machine Learning Repository. The data can be found [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). A Portuguese banking institution derived the dataset from a direct marketing campaign made via phone calls. This project aims to accurately predict the success of future campaigns based on the attributes listed in the dataset, implementing a high-preforming machine learning model. The generalized ML pipeline is outlined in the diagram below.

![A generalized Machine Learning Pipeline for this project.](https://github.com/bigredbayes/BankMarketingModeling/blob/main/Bank_Marketing_Pipeline.png)

This document will follow the steps outlined in the ML pipeline above, giving exaplanations for technical considerations and decisions made along the way. The following list summarizes each step in the process:

1) Problem Identification- Identifies and defines the problem that can be solved with machine learning techniques. This involves clearly understanding the objectives, goals, and requirements of the problem at hand.
2) Data Gathering- Gathers the relevant data required for training and evaluating the machine learning models. This includes accessing known databases and exploring potential new sources that can be added to the model.
3) Data Preprosseing- Preprocesses the data to prepare it for training the machine learning models. This step involves cleaning the data by handling missing values, dealing with outliers, and addressing inconsistencies or errors in the data. It may also include data normalization, feature scaling, or transformation to make the data suitable for the chosen machine learning algorithms.
4) Feature selection- Identifies and selects the most relevant features from the available dataset. This step helps in reducing dimensionality and focusing on the features that have the most impact on the target variable.
5) Model training- Trains the various machine learning models by choosing the appropriate algorithms, fitting the model to the training data, and optimizing the model hyperparameters.
6) Model evaluation- Evaluates the trained models to assess their performance and gauge their capabilities. This step involves using a separate testing dataset to measure the model's accuracy metrics, such as precision, recall, F1 score, brier score, and others.
7) Final model selection- Selects the best-performing model for the given problem based on the results of the model evaluation. This involves comparing multiple models, considering their performance metrics, interpretability, computational complexity, and other factors. The selected model can then be deployed in a real-world setting to make predictions or decisions based on new, unseen data.

### Problem Identification

