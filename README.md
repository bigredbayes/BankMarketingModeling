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

A Portuguese banking institution fails to easily identify people who are susceptible to direct marketing campaigns. This institution would like a higher proportion clients to subscribe to a term deposit as a result of these marketing campaigns. The solution is the create a predictive model that uses the institution's direct marketing campaign data to successfully identify the clients who are most likely to subscribe to a term deposit. A predictive algorithm would thus classify a client as subscribed or unsubscribed to a term deposit. Such a classification analysis would allow the institution to better target clients with a high probability of subscribing and/or not waste time on clients with a low probability of subscribing to a term deposit.

### Data Gathering

The data gathering for this project was relatively simply. As stated previously, the [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset from the UCI Machine Learning Repository contains all the features collected by the Portuguese banking instituion. These include variables specific to clients such as their age, job type, and marital status, in addition attributes related to previous marketing campaigns and macroeconomic indicators such as the number of previous contacts with the client, the employment variation rate, and the consumer price index. 

The potential for supplementary data was also explored at this point. Specifically, more macroeconomic indicators such as Portugal's [Real GDP]([https://archive.ics.uci.edu/dataset/222/bank+marketing](https://fred.stlouisfed.org/series/CLVMNACSCAB1GQPT)https://fred.stlouisfed.org/series/CLVMNACSCAB1GQPT) and [Bank Z-Score](https://fred.stlouisfed.org/series/DDSI01PTA645NWDB) were considered. The idea was that a country's Real GDP indicates spending levels and would be correlated with individuals' willingness to spend money, which would make them more likely to open a term deposit. Similarly, a bank's z-score would capture the public's general trust in financial institutions, as a higher probability of banks defaulting would dissuade clients from investing with the financial institution and there would be fewer new term deposits. However, both these variables were already highly correlated with other macroeconomic indicators found in the Bank Marketing dataset and thus left out of the analysis.

### Data Preprocessing

Data preprocessing began by reading in the bank-additional-full.csv file with all 41,188 data observations and 20 inputs. Next, the value counts of the dependpent variable were checked to determine the balance of the dataset. The "positive" class (clients who subscribed to a term deposit) only consisted of 4,640 observations, or about 11.27% of the datasets total observations. This indicates an unbalanced dataset. If more time was given for this project, additional preprocessing steps would have been taken such as synthetic data generation with ADASYN or SMOTE undersampling/oversampling in order to balance the dataset and improve overall model accuracy. However, for the purposes of this project models were built using the unbalanced data.

An outlier check was conducted to explore the shape of the data and identify any worrisome observations. 

Lastly, all categorical variables were one-hot encoded to conclude data preprocessing.
