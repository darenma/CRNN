# CRNN
Contextual Sequence Modeling for Recommendation with RNN 
– A Paper Implementation

Annette Lin, Daren Ma
## Background and motivation
The recommendation system is a vastly discussed and extremely important topic in the application of Deep Learning. Among the vast range of Deep Learning algorithms, Recurrent Neural Networks (RNN) are especially good at exhibiting temporal dynamic behavior in a dataset. In this project, we desire to use Deep Learning algorithms to solve a real-world recommendation problem on a large dataset. We selected the paper Contextual Sequence Modeling for Recommendation with Recurrent Neural Networks to implement its ideas. During this process, we gained a better understanding of both Recommendation Systems and RNN. 
To have a solid benchmark, we followed the same steps to preprocess the data and to develop the algorithms in the paper. Unfortunately, the authors didn’t provide their actual code, so we could only imitate the steps out of our own understanding.

## Technologies
We used Numpy and Pandas to preprocess the data. We developed the model in PyTorch.
Our notebook can be found on Github: https://github.com/darenma/CRNN.

## Data
We use the Yoochoose dataset for RecSys 2015 Kaggle Competition to compare with the benchmark provided by the authors. The description of this dataset can be found here (https://www.kaggle.com/chadgostopp/recsys-challenge-2015). Note that the dataset is divided into 2 subsets: one is the clicks dataset and the other is the buys dataset. We need to process and combine them before use.
The original Buys dataset contains 1.1 million rows, and the Clicks dataset contains over 30 million rows. To speed up the process, we sampled 30,000 unique sessions that had only clicks and 30,000 sessions having both clicks and buys logs. Then we applied the data preprocessing tricks talked in the next chapter. After preprocessing, we got a dataset of 0.25 million rows, among which we had 234,000 rows of the training data, 8,000 rows of validation data and 7,198 rows of testing data.

## Variables and Target
There are multiple approaches to frame the dataset for a Machine Learning problem. One general way is to make this a Binary Classification problem, where y is 0 (not buy) or 1 (buy). This notebook by an anonymous Kaggler is a good example. 
However, what we actually care about is whether our model can predict not only the buy event but also which item the user buys. Therefore, we followed the steps in the paper to generate a fixed-length time window for the buy/click log data of each session (user). Then the topic has been addressed as a Multi-Classification problem. We selected recall@10 to be our metric.

To be specific, we do the following steps to make the data fit our needs.

We union the buys and clicks datasets together and sort them by session_id and Timestamp in chronological order. 

For each session, we make a rolling window of 20 events and treated the last event in the window as our target. The minimum number of events of the window should be 2 (i.e. at least one event before our target event). We add paddings for those windows having fewer than 20 events to unify the length of our dataset.
This figure shows the head 5 rows of our dataset after step 2. Note that the index is the time index of these events, and padding rows of values -1 are added. 

We extract each 20-event time window as one record and transform the dataset long-to-wide. This is followed by saving the data into a CSV file where each row is an X for easier access. 

## Feature Engineering
We implemented the feature engineering of the paper. 
To consider both of the buys and clicks DataFrames, we dropped the columns that only occurred in one dataset. This includes price and quantity.
We also dropped the category column as it is not considered a contextual variable by the paper.
We treat the item_id as a categorical feature and train embeddings of it in the model.
We extract the contextual variables as the paper did. They’re all discretized as categorical variables and their embeddings are taken as input. This includes:
Hour, Month, DayOfWeek. These are extracted from Timestamp.
Event. It’s 0 if the event is Click, and 1 for Buy.
Timediff. It’s the time difference between two consecutive events in seconds. We performed a log2 transformation and binned it to be integers ranging from 0 to 20.
For NaN values, we filled them to be -1 which are later treated as the padding index in the modeling part.
In the training process, we reshaped the input variables to form a 20 * k matrix to fit into our model, where 20 is the length of the time window, and k is the total number of features. 

## The algorithm
The features are divided into 2 categories: the item history of the session and the context of each action. In every 20-event time range, we define the last event to be our y. Hence for each x and y, there is a corresponding item and context. To extract information from both, the RNN has 3 modules:
Input module. This module takes the item id and the context vector of x, makes them interact, and generates a context-dependent item representation.

The interaction is called an integration function. 3 integration functions are explored: concatenation, multiplicative interaction, and the combination of both. The experiment results in the paper prove that the combination performs the best, so we use it as the interaction when we run our model.
Recurrent module. This module updates the hidden state vector with the current input and the previous state. We use GRU as the paper does.
Output module. This module takes the last state vector, makes it interact with the context vector of y using the same integration function described in the input module, and applies a linear layer to predict the probability distribution over items.





## Model Evaluation
We use Recall@10 as our NorthStar metric. It is the metric the authors used in the paper so we can easily compare our results. Recall@10 is the proportion of purchased items found in the top-10 recommendations. Our best result, for now, is 0.423, which means 42.3% of the users did buy one of the top-10 items we predicted the user to be interested in.

## Responsibilities
Our team has two members, Annette (Zijun) Lin and Daren Ma.
Daren is responsible for the data preprocessing and feature engineering part. He applied Pandas tricks to clean, filter, reshape the datasets to create the chronological variables. Daren also wrote the report and maintained the team’s Github repository.
Annette is responsible for modeling and presentation. She understood the mathematics behind the original algorithm and implemented the CRNN from scratch using PyTorch. Annette conducted hyperparameter tuning in the training process. She also wrote the scripts for the presentation.

## References
Original Paper: Contextual Sequence Modeling for Recommendation with Recurrent Neural Networks
Discussion of the multiplicative interaction function described in the algorithm: A Multiplicative Model for Learning Distributed Text-Based Attribute Representations



