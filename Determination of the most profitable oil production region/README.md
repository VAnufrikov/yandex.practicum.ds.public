# Determination of the most profitable oil production region
*** 

### Task
***

`Glavrosgosneft` needs to decide where to drill a new well.

Steps for choosing the location of the well:

1. Collect characteristics about the quality and volume of stocks in the region
2. Build a model for predicting the volume of reserves in new wells
3. Choose wells with the highest ratings
4. Determine the region with the maximum total profit

The project provides oil sample data in 3 regions, build a model for the region where production will bring the greatest profit.

And also analyze the profits and risks with the Bootstrap technique
### Data description
***
The data consists of three files. In order not to confuse the regions, we will call them conditionally three cities of Russia.

- `geo_data_0.csv` - Moscow
- `geo_data_1.csv` - Saint_petersburg
- `geo_data_2.csv` - Kazan


### Description of fields
***
- `id` — unique well ID
- `f0`, `f1`, `f2` — signs of the selected region
- `product` — the volume of well reserves (thousand barrels).

> Income per unit of product is 7 500 $

### Work plan
***

#### Dataframe analysis
- [x] Getting to know the data
- [x] Creating a lists to store the results of the `mse`, `rmse`, `r2` and `mae` metrics

#### Data preprocessing
- [x] Checking for missed entries
- [x] Checking outliers in data

#### Model Training
- [x] Creating a `Linear Regression` learning function
- [x] `Splitting` df into a training and test sample 75-25
- [x] `Normalize` the signs

#### Profit modeling
- [x] Create a profit calculation function
- [x] Setting the development budget
- [x] Analyze profits and risks using the `bootstrapping` method

### Conclusions
***
I downloaded and analyzed the data in the project, after modeling the profit from the task condition, the second Saint Petersburg region was selected.

Due to the fact that this region has the lowest percentage of losses and a very high `profit forecast
### Libraries
***
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `scipy`

