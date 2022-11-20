# Forecasting the outflow of Bank customers
*** 

### Task
***
At `Beta Bank`, every month, customers began to close accounts and go to competitors.

Marketers believe that retaining current customers is cheaper than attracting new ones.

The task of forecasting whether the client will close the account in the near future or not has been set.

### Data description
***
Historical data on the behavior of customers and the closure of accounts with the bank are presented
- `Churn.csv` - All historical data

### Description of fields
***
- `RowNumber` — index
- `CustomerId` — client ID
- `Surname` — surname
- `CreditScore` — credit rating
- `Geography` — country of residence
- `Gender` — gender
- `Age` — age
- `Tenure` — how many years has the client been a client of the bank
- `Balance` — account balance
- `NumOfProducts` — number of bank products
- `HasCrCard` — availability of a credit card
- `IsActiveMember` — client activity
- `EstimatedSalary` — estimated salary

> It is necessary to maximize the `F1` metric and `AUC-ROC`

### Work plan
***

#### Dataframe analysis
- [x] Getting to know the data
- [x] `Class balance` analysis

#### Data preprocessing
- [x] Checking for missed entries
- [x] Checking outliers in data
- [x] `Splitting` df into a training, valid and test sample 30-30-40
- [x] `Encode` the signs using the `OHE` method
- [x] `Normalize` the features

#### Model Training
- [x] Creating a `Randomized Search CV` search function with `cross-qualification`
- [x] Setting the `hyperparameters` that we will `iterate` over the models
- [x] Training the `GradientBoostingClassifier` classifier
- [x] Training the `RandomForestClassifier` classifier
- [x] Training the `DecisionTreeClassifier` classifier

#### Checking the best model
- [x] Building `confusion_matrix`
- [x] Outputting `classification_report`
- [x] `Draw a graph` of the ROC curve

### Conclusions
***
Our initial task was to predict the outflow of customers and achieve the `f1` metric greater than 0.75.
With the help of boosting, `f1` was 0.9353195, and `AUC-ROC` was 0.85

After the implementation of the MNP project, `Beta Bank` began to feel more confident in the market. 
And it also allowed him to retain key customers and offer them favorable terms on accounts.
### Libraries
***
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`

