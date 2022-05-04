# ZillowHouses

“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. 
And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning.

Submissions are evaluated on Mean Absolute Error between the predicted log error and the actual log error. The log error is defined as logerror=log(Zestimate)−log(SalePrice) and it is recorded in the transactions training data.

Dataset is reported through PandasProfilingReport because is too big for github.

File descriptions:

properties_2016.csv - all the properties with their home features for 2016.

properties_2017.csv - all the properties with their home features for 2017.

train_2016.csv - the training set with transactions from 1/1/2016 to 12/31/2016.

train_2017.csv - the training set with transactions from 1/1/2017 to 9/15/2017.

zillow_data_dictionary.xlsx - features meaning


Performance:

MAE : 0.07507 

In order to have a comparison, this is performance of 1st in leaderboard: 0.07408
