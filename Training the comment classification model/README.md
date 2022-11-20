# Training the comment classification model
*** 

### Task
***

`Wiki shop` online store launches a new service for editing and supplementing product descriptions, as in wiki communities.

The store needs a model that will search for toxic comments and send them to a store employee for moderation

> The task in this project is to train a model for classifying comments into positive and negative


### Data description
***
We have at your disposal a data set with markup on the toxicity of edits
- `toxic_comments.csv` - All data

### Description of fields
***
- `text` — comment
- `toxic` — toxicity flag

### Work plan
***

#### Dataframe analysis
- [x] Getting to know the data
- [x] `Class balance` analysis
- [x] Created a df to record the results

#### Data preprocessing
- [x] Convert the text to lowercase
- [x] Clearing of unnecessary characters
- [x] Processing duplicates
- [x] Creating features
- [x] Removed stop words
- [x] Lematized text
- [x] Splitting df into a training, valid and test sample 80-20
- [x] Vectorized text

#### Model Training
- [x] Creating a `Randomized Search CV` search function with `cross-qualification`
- [x] Setting the `hyperparameters` that we will `iterate` over the models
- [x] Training the `LogisticRegression` classifier
- [x] Training the `Dummy_Classifier` classifier
- [x] Training the `RandomForestClassifier` classifier
- [x] Training the `Cat_Boost_Classifier` classifier

#### Checking the best model
- [x] Building `confusion_matrix`
- [x] Outputting `classification_report`
- [x] `Draw a graph` of the ROC curve

### Conclusions
***
In this project, we have done a solid job of processing and preparing data for training models.
At the end of the project, we brought the training of the best model and compared the quality of training with the constant model.

The best `f1` metric that we managed to achieve is `TF-IDF+Catboost` - 0.758487

### Libraries
***
- `pandas`
- `numpy`
- `re`
- `wordnet`
- `WordNetLemmatizer`
- `time`
- `matplotlib`
- `sklearn`

