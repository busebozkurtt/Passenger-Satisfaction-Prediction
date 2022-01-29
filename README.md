# Passenger Satisfaction Prediction

The dataset consists of eighteen numerical features and five categorical features that make up a total of twenty-three columns. The raw data set consists of 103,903 lines containing information about the airline passengers and the journey. The target feature in the dataset includes whether the passengers are satisfied with their journey.

- __Gender:__	Gender of the passengers (Female, Male)
- __Customer Type:__	The customer type (Loyal customer, disloyal customer)
- __Age:__	The age of the passengers
- __Type of Travel:__	Purpose of the flight of the passengers (Personal Travel, Business Travel)
- __Class:__	Travel class in the plane of the passengers (Business, Eco, Eco Plus)
- __Flight Distance:__	The flight distance of this journey
- __Inflight Wifi Service:__	Satisfaction level of the inflight Wi-Fi service (0: Not Applicable, 1-5)
- __Departure/Arrival Time Convenient:__	Satisfaction level of Departure/Arrival time convenient
- __Ease of Online Booking:__	Satisfaction level of online booking
- __Gate Location:__	Satisfaction level of Gate location
- __Food and Drink:__	Satisfaction level of Food and drink
- __Online Boarding:__	Satisfaction level of online boarding
- __Seat Comfort__	Satisfaction level of Seat comfort
- __Inflight Entertainment:__	Satisfaction level of inflight entertainment
- __On-board Service:__	Satisfaction level of On-board service
- __Leg room Service:__	Satisfaction level of Leg room service
- __Baggage Handling:__	Satisfaction level of baggage handling
- __Check-in Service:__	Satisfaction level of Check-in service
- __Inflight Service:__	Satisfaction level of inflight service
- __Cleanliness:__	Satisfaction level of Cleanliness
- __Departure Delay in Minutes:__	Minutes delayed when departure
- __Arrival Delay in Minutes:__	Minutes delayed when Arrival
- __Satisfaction:__	Airline satisfaction level (Satisfaction, neutral or dissatisfaction)

##  Libraries required for the project: 
- numpy
- pandas
- matplotlib
- seaborn
- lightgbm
- sklearn
- plotly

## Data Preprocess
One of the most important stages affecting the performance of machine learning algorithms is the data pre-processing stage. Optimizing the data greatly increases the performance. The data pre-processing part was examined and completed under 6 headings in the project. These operations are; Removing unnecessary columns, filling missing data, removing duplicate rows, encoding categorical variables, removing outlier and standardization stages.

### 	Drop Unnecessary Columns
First of all, it is important to understand the dataset. When the dataset is analyzed, some features are removed from the dataset. These properties are those that do not affect the accuracy of the model to be created. Looking at the dataset used in the project, the "Unnamed: 0" and "id" properties have been removed from the dataset. Because these features do not affect the target feature result.
After that, feature extraction was performed by examining the correlation matrix. Considering the correlation matrix of the used dataset, the properties with high correction have been removed because they may negatively affect the classification result created. The extracted feature is "Arrival Delay in Minutes" and has a 0.96 correlation with the "Departure Delay in Minutes" feature. In addition, features that do not affect the target feature and have a correlation matrix of zero are also removed. This feature is "Gate Location".

### 	Missing Values
When the dataset used in the project is examined, the "Arrival Delay in Minutes" feature has 310 missing values. Mean values are used to eliminate these missing values. In this method, the non-missing values in the "Arrival Delay in Minutes" column are averaged. The missing values were then replaced with this average value.

###  Categorical Variable Encoding
One hot encoding process is the most used. In the dataset, "Gender", "Customer Type", "Type of Travel", "Class" and "satisfaction" properties are categorical variables. One hot encoding is not a correct method for attributes that have two values. Because these columns can be expressed with 0 and 1. Therefore, if one hot encoding was done, there would be two different variables and they would be the opposite of each other. Therefore, the values are changed to 0 and 1. Only the Class attribute has 3 different values. These are "Eco", "Eco Plus" and "Business" values. Since these values can represent a ranking, one hot encoding is not done. Instead, it has been converted to numerical values such as 0,1,2 because it indicates the order.

### 	Outlier Detection
One of the most misleading situations for machine learning algorithms and also one of the important steps for pre-processing is outlier detection and their extraction from the data. If outlier detection is not made, machine learning models will try to generalize this outlier data as well. In generalizing them, in fact, the generalization of normal data will be affected and generalize them worse. Therefore, outliers in the data were determined by using the LocalOutlierFactor method and those lines were removed from the data.

###  Standardization
The fact that the range of numerical values is very different can greatly affect the performance of the algorithms. In particular, the performance of algorithms that use distance measurements is adversely affected. Therefore, it is important to translate all variables into a specific range. With the StandardScaler function to be used in the project, the values of all variables have been changed so that the mean is 0 and the standard deviation is 1.

##  Classification Models
__Logistic Regression__ is a statistical method and analyzes datasets with more than one variable. It performs classification on the data it analyzes. In logistic regression, the result is measured with a binary variable and the events are independent. Therefore, in this model, the result should be separate. Maximum Likelihood Estimation (MLE) is used to estimate parameters, so it relies on large sample approaches. Using too many features in logistic regression causes overfitting.

__Light-GBM__ algorithm has been developed in order to obtain faster and more successful results by optimizing the Gradient Boosting algorithm. LightGBM is a histogram-based algorithm, while doing this, it aims to reduce the computational cost by making the continuous variables discrete. In addition, unlike other tree algorithms, it uses the Leaf-wise learning method. It is critical to avoid the possibility of overfitting with parameter optimization (especially tree depth, number of leaves).

__Random Forest__ algorithm is a type of machine learning algorithm known as Ensemble learning. This algorithm has emerged for the overfitting problem, which is one of the weaknesses of the decision tree algorithm. It tries to avoid the overfitting problem by creating more than one decision tree and choosing different subsets from the data set. It selects different subsets from the data set with the Bootstrap method. In this way, the outlier effects in the data set are reduced. All trees generate a prediction result and the algorithm returns whatever result comes out the most. Since there is more than one decision tree, the prediction can be more accurate and the overfitting problem is avoided.

## Results of Models

The results of the algorithms used were compared with and without cross-validation. In addition, Random Forest and Light-GBM algorithms were evaluated by removing features with low feature importance.

The image below shows the accuracy results of all models on the test data.

![image](https://user-images.githubusercontent.com/50152584/151385957-58e5a279-291c-4ee6-b047-2234600ee9ab.png)

The image below shows the F1-Score results on the test data of all models.

![image](https://user-images.githubusercontent.com/50152584/151386177-123e6225-0c06-424c-bb4e-811d93877049.png)

The ROC results of the 3 selected models are shown in the image below.

![image](https://user-images.githubusercontent.com/50152584/151386429-206f2c42-09e0-4d32-92f7-66e8a0a72754.png)
