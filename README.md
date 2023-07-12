# Bank Marketing Modeling

This README.md will act as technical documentation for the machine learning pipeline implemented on the Bank Marketing dataset from the UCI Machine Learning Repository. The data can be found [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). A Portuguese banking institution derived the dataset from a direct marketing campaign made via phone calls. This project aims to accurately predict the success of future campaigns based on the attributes listed in the dataset, implementing a high-performing machine learning model. The generalized ML pipeline is outlined in the diagram below with black boxes indicating problem/solution steps, yellow boxes referencing data-related steps, and blue boxes representing model building steps.

![A generalized Machine Learning Pipeline for this project.](https://github.com/bigredbayes/BankMarketingModeling/blob/main/Bank_Marketing_Pipeline.png)

This document will follow the steps outlined in the ML pipeline above, giving explanations for technical considerations and decisions made along the way. The following list summarizes each step in the process:

1) Problem Identification- Identifies and defines the problem that can be solved with machine learning techniques. This involves clearly understanding the objectives, goals, and requirements of the problem at hand.
2) Data Gathering- Gathers the relevant data required for training and evaluating the machine learning models. This includes accessing known databases and exploring potential new sources that can be added to the model.
3) Data Preprocessing- Preprocesses the data to prepare it for training the machine learning models. This step involves cleaning the data by handling missing values, dealing with outliers, and addressing inconsistencies or errors in the data. It may also include data normalization, feature scaling, or transformation to make the data suitable for the chosen machine learning algorithms.
4) Feature selection- Identifies and selects the most relevant features from the available dataset. This step helps in reducing dimensionality and focusing on the features that have the most impact on the target variable.
5) Model training- Trains the various machine learning models by choosing the appropriate algorithms, fitting the model to the training data, and optimizing the model hyperparameters.
6) Model evaluation- Evaluates the trained models to assess their performance and gauge their capabilities. This step involves using a separate testing dataset to measure the model's accuracy metrics, such as precision, recall, F1 score, Brier score, and others.
7) Final model selection- Selects the best-performing model for the given problem based on the results of the model evaluation. This involves comparing multiple models, considering their performance metrics, interpretability, computational complexity, and other factors. The selected model can then be deployed in a real-world setting to make predictions or decisions based on new, unseen data.

### Problem Identification

A Portuguese banking institution fails to easily identify people who are susceptible to direct marketing campaigns. This institution would like a higher proportion clients to subscribe to a term deposit as a result of these marketing campaigns. The solution is to create a predictive model that uses the institution's direct marketing campaign data to successfully identify the clients who are most likely to subscribe to a term deposit. A predictive algorithm would thus classify a client as subscribed or unsubscribed to a term deposit. This classification analysis would allow the institution to better target clients with a high probability of subscribing and avoid wasting resources on clients with a low probability of subscribing to a term deposit.

### Data Gathering

The data gathering for this project was relatively simple. As stated previously, the [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset from the UCI Machine Learning Repository contains all the features collected by the Portuguese banking institution. These include variables specific to clients such as their age, job type, and marital status, in addition attributes related to previous marketing campaigns and macroeconomic indicators such as the number of previous contacts with the client, the employment variation rate, and the consumer price index. 

The potential for supplementary data was also explored at this point. Specifically, more macroeconomic indicators such as Portugal's [Real GDP](https://fred.stlouisfed.org/series/CLVMNACSCAB1GQPT) and [Bank Z-Score](https://fred.stlouisfed.org/series/DDSI01PTA645NWDB) were considered. A country's Real GDP indicates spending levels and would be correlated with individuals' willingness to spend money, which would make them more likely to open a term deposit. Similarly, a bank's z-score would capture the public's general trust in financial institutions, as a higher probability of banks defaulting would dissuade clients from investing with the financial institution, resulting in fewer new term deposits. However, both these variables were already highly correlated with other macroeconomic indicators found in the Bank Marketing dataset and thus left out of the analysis.

### Data Preprocessing

Data preprocessing began by reading in the bank-additional-full.csv file with all 41,188 data observations and 20 inputs. Next, the value counts of the dependent variable were checked to determine the balance of the dataset. The "positive" class (clients who subscribed to a term deposit) only consisted of 4,640 observations, or about 11.27% of the datasets total observations. This indicates an unbalanced dataset. If more time was given for this project, additional preprocessing steps would have been taken such as synthetic data generation with ADASYN or SMOTE undersampling/oversampling in order to balance the dataset and improve overall model accuracy. However, for the purposes of this project, models were built using the unbalanced data.

An outlier check was conducted to explore the shape of the data and identify any worrisome observations. Only numeric, non-categorical variables were studied for potential outliers. Outliers were checked using interquartile ranges (IQRs) and boxplots. Utilizing IQRs identified any points far enough from the feature mean that allowed for further examination. Similarly, the boxplots allowed for visual examination of the numeric variables to identify any observations that might be considered outliers. Upon the examination of these tools, it was clear that all the outliers identified by these methods were not cause for concern. Most features are not normally distributed and skewed, but most individual data points were still close to at least a few other data points, indicating they wouldn't have undue leverage in the model. The "pdays" feature (the number of days that passed by after the client was last contacted from a previous campaign) had a concentrated cluster of outliers, but that was due to a value of 999 being assigned when clients had not been previously contacted. Ultimately, it was determined that there was no reason to remove outliers as most feature distributions were skewed and no singular data point was adding unnecessary weight to future models. Keeping these "outliers" could actually assist the model in drawing boundaries between classes. Additionally, a higher proportion of "positive" class data points were kept in the dataset since the proportion fell below 10% of all observations once all "outliers" are removed from the dataset.

Lastly, all categorical variables were one-hot encoded to conclude data preprocessing. Each categorical variable was a assigned a binary dummy variable for each category within the feature.

### Feature Selection

Two methods were used for feature selection in the dataset. The first method was univariate selection, which determined the strength of the relationship between each variable and the class. For this process all features were assigned numeric values so that the tests could be conducted. While a Chi-squared test would be appropriate here, some features contained negative values which made score calculations impossible. Instead, an F-score was used to assign significance to the relationship between each input and the output variable. The best scoring feature was "duration" (the number of seconds contact lasted) which is due to its highly impactful relationship with the output variable since a duration of zero immediately results in a "negative" output class. Shorter call durations also result in fewer "positive" classes than longer calls. The "loan" variable (whether or not the client has a personal loan) was determined to be the least significant feature. It was decided that only the top 10 features would be kept to reduce unnecessary noise in the model. This resulted in any feature with an F-score below 200 being dropped. 

The second feature selection methodology used for the Bank Marketing dataset was feature correlation. All numeric features that remained after univariate selection were put into a correlation matrix and plotted to look for multicollinearity. It was decided that any variables with a correlation over 0.9 would be removed as features. Two variables, "emp.var.rate" and "euribor3m" (employment variation rate and Euro Interbank Offered 3-Month Rate), were heavily correlated with each other and some additional variables. These variables were removed to prevent multicollinearity from affecting the models. This resulted in a total of eight features being included in the model building.

With more time, feature selection would have been done using feature importance after training the models. However, training and optimizing the models on a full dataset, adjusting for feature importance, and then training and optimizing the models again was too time consuming for this project's allotted timeframe. For these reasons, it was decided that feature selection prior to training the models was better under these circumstances. Once the data preprocessing and feature selection was finalized, the final dataset was split into training and testing subsets using a 80-20 split.

### Model Training

Three different model types were selected for training: random forest, logistic regression, and naive bayes. A neural network and support vector machine were also considered, but were not fully trained for different reasons. The neural network needed its own optimization framework using tensorflow and keras which would take hours to run- too much time for this project. Initial trainings of the support vector machine algorithm did not perform well, so it was discarded in favor of the other three working models.

Data prep and optimization functions were built to train the models. The data prep function split the training data into new training and validation data subsets using an 80-20 split. The optimization function assessed each models potential hyperparameters and selected the best performing ones. Optimization was done utilizing the BayesSearchCV from scikit-learn, a Bayesian search optimization. BayesSearchCV was chosen over other optimization functions because it was less computationally expensive than GridSearchCV, which tries all possible combinations of hyperparameters, and it is more structured in its sampling than RandomSearchCV. Each optimization was iterated 50 times and used 3-fold cross validation. Lastly, each model's Brier score ultimately determined which set of hyperparameters optimized the model. Brier score was selected over traditional metrics like accuracy, precision, and recall because it is better suited for assessing model performance on unbalanced datasets.

All three models were built and trained using their respective scikit-learn functions. Additionally, each model had its own defined class in the pipeline which was called upon to prep the data, build an initial model, optimize the model parameters, and return a final model with its validation performance. The final, optimized model for each algorithm was then passed to the model testing portion of the pipeline.

### Model Evaluation

Each model was evaluated on several criteria to compare their strengths and judge any potential weak points. Model accuracy was used as a baseline metric to determine if the model could beat random guessing. However, accuracy is not the best metric when assessing an unbalanced dataset. The model F1-score is included as an aggregate of the precision and recall of each model, determining the impact of true positives in the confusion matrix. Another traditional metric used to judge the final models was the area under the receiver operating characteristic curve (ROC-AUC), which is a hallmark of a robust model. Log loss was also included in the criteria to indicate how close the model predictions were to their true values. Brier scores do the same, but function especially well on unbalanced datasets such as the Bank Marketing one. Both log loss and the brier scores are better when lower, whereas the other three metrics should be maximized as much as possible.

After setting the evaluation metrics, each model was run using the test data. A model testing class returned all five metrics that were previously mentioned in addition to their respective confusion matrices.

### Final Model Selection

At the conclusion of the final model testing and evaluation, all metrics were compared and the random forest model was selected as the final model with hyperparameters of an entropy criterion, a max depth of 10, and 1,000 estimators. This model outperformed the logistic regression and naive bayes models in every metric of interest. It had a higher accuracy, F1-score, and ROC-AUC, and a lower log loss and brier score.

This selected model can be deployed in a real-world setting to make predictions or decisions based on a data feed about an institution's clients. A Portuguese banking institution could profile its clients based on this final random forest model to determine whether its resources are worth devoting to a direct marketing approach on a client-by-client basis. Beyond simple classification, the underlying probabilities assigned to each class by the model could be used for further assessment by the bank. For instance, the institution might decide that it is worth pursuing any clients for open term deposits if their chances are greater than 30% or 90%, rather than 50%. A cost-benefit analysis could be conducted at a probability threshold of getting new clients on the tradeoffs between the costs of the direct marketing campaigns and the potential for adding those new clients. Ultimately, this model can be deployed in many ways in addition to any customizing or tailoring of the final model to fit the banking institution's needs.
