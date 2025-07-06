# Power-Consumption-in-Tetouan-City-Morrocow

## Objective
To develop a time series regression model that accurately forecasts Zone 1 power consumption based on atmospheric data and historical consumption trends using the Power Consumption of Tetouan City Dataset.
To provide better understanding of our dataset and its contents, a visual representation of raw data is being displayed below:

![image](https://github.com/user-attachments/assets/2205496e-ff78-4eba-a450-af00d46e477a)


After displaying the dataset and to better understand the structure of each feature, the column names with datatypes and non-null counts are displayed below:

![image](https://github.com/user-attachments/assets/b47528ea-ab6f-4c6c-8661-01627f58b291)



## EDA
Let’s proceed with Exploratory Data Analysis for our dataset to get some meaningful insights for our data as it ensures that our data is clean, consistent and ready for training before modelling.
### Handling/Analyzing missing or null values:
For the reliability of our data, it is necessary to find out null/missing values in our data and handle them.

![image](https://github.com/user-attachments/assets/b2dff9c0-7451-4b6a-b791-71376caa9b3f)



Fortunately, there are no missing values in our dataset which makes it consistent in terms of handling values. Next, describing the dataset or statistical aspects of the dataset is very crucial. Therefore, we will look into descriptive statistics summary of the dataset.


### Statistical Description:
For a better understanding for the central tendency and distribution of the numerical features of the dataset, statistical description is important. It provides basic statistical metrices such as count (total number of non-null datapoints), mean (sum of total datapoints/ total number of datapoints), standard deviation (how far away are data points from the mean). Also, to point out the outliers in the data we should look into quartiles (Q1, Q2, Q3) (25%, 50%, 75%). Quartiles also help us in figuring out how our data is distributed. 

![image](https://github.com/user-attachments/assets/e0e060f2-7210-4e11-b76c-a4b9a8174b77)

Q1(25%) is the threshold value which points out the datapoints that falls below 25% of the total number of datapoints. In the same way, Q3(75%) represents the point which basically has 75% of the data falling below this point and Q2(50%) is the median value or the middle point of the distributed values. If any datapoint falls below Q1 and falls above Q3, it is considered as an outlier, when we are using the concept called inter-quartile range (IQR). Therefore, the datapoints falling between Q1 and Q3 will be considered as consistent and we have to focus on those datapoints only. Also, we are able to figure out symmetry and skewness of the data as well in the process.
Note: Statistical features can only be applied on numerical data.

### Feature Engineering with Date-time
As we saw in the Figure 1., which displays the raw data, we witnessed that there was a column named as “DateTime” but it becomes difficult to utilize these datapoints given in the DateTime column given the context of the data. Therefore, to extract something meaningful or something valuable from this feature, we are going to perform engineering on this feature. Basically, we are going to break the components of this feature and split it into different columns named as day, month, day of the week, hour, minute, quarters, quarter of the year, day of the year. We are basically extracting datapoints from the DataTime column and distributing them in the columns as mentioned. After splitting the data, we will also be removing the original DateTime column from our dataset so that data repetition can be avoided. Below is an image of how our data looks after feature engineering.

![image](https://github.com/user-attachments/assets/a36327ff-3f5f-445a-b55e-ae43bc127e07)

### Correlation Analysis among variables:
Correlation is nothing but finding the relationship between two variables and by relationship we mean the direction that two variables are heading towards. The name in itself defines its meaning, where co meaning two (used with related things) and relationship meaning how they are related. It is a statistical measure which describes strength of two variables and in which directing they are moving. Furthermore, it tells how a change in one variable can bring a change to another variable. It is only applicable to numeric values. To understand the concept better or to get something meaningful we need something which is defined when it comes to understanding the correlation. For this we need to look into coefficient of correlation. Coefficient is nothing but a statistical measure that defines the magnitude of linear relationship among two variables. It ranges from (-1 to 1) on a scale. If coefficient is (-1), this indicates a negative correlation among the variables. Unlike, if the coefficient is (1) indicates a positive correlation. If correlation is (0), it means they have no relationship. For example, we have two variables m and n, if the correlation is negative it means with every increasing value of m, n is decreasing and if the correlation is positive, it means with every increasing value of m, n is also increasing. Negative correlation reflects the inverse effect and positive correlation shows the same effect among the variables.


![image](https://github.com/user-attachments/assets/78a973c4-8bf4-49ac-9733-ce231c360a06)

In context to our project, correlation heatmap helps us in figuring out how different factors such as humidity, wind speed, diffuse flows, temperature, etc. are related to each other when it comes to finding out the consumption of electricity in Tetouan city. This is important to understand how different factors affect the electricity consumption or helps in energy forecasting understanding different patterns and insights. Also, it can help us to formulate strategies by looking into datapoints for better solutions for optimum electricity usage.

### Time-Basis EDA:
“Any set of data that consists of numerical measurements of the same variable collected and organized according to regular time intervals may be regarded as time series data”. Basically, it is the process of keeping a check on the data for over a period of time. It helps in pointing out certain patterns or trends, outliers etc. over a period of time. It is helpful in a time-series based datapoints like we have in our dataset- “DateTime” column which we have transformed with feature engineering into months, days, day of the week, hours, etc. It helps better decision making by understanding energy consumption trends. For example, what was the consumption initially, how it increased or decreased during monthly intervals over a period of one year. It also enlightens our understanding of energy demand over the region. Here, we will be displaying energy consumption of Zone 1, Zone 2, Zone 3.

![image](https://github.com/user-attachments/assets/d4561f12-9bba-411c-85e9-ed1a0e410dd6)

### Analyzing Density-Plot:
The Kernel Density plot (kde) helps in understanding the distribution of electricity consumption across different zones (zone 1, zone 2, zone 3) of the city. The histogram does help in clear visualization of the distribution of datapoints but kde plots provides a more clear picture than histogram alone. The plot helps us in visualizing patterns of electricity utilization across the three zones in the dataset.

![image](https://github.com/user-attachments/assets/e97becad-4720-4c6b-9048-ad8f15048df9)

From the figure above, it can be observed that every plot has a peak point which indicates most common electricity consumption range in that for the specific zone. The plots which are more broad and flatter indicates more variability in the datapoints or electricity consumption, while the plots which are pointed and have sharper peaks reveals more consistency in electricity usage. This visual helps us in figuring out the demand of energy across different zones and helps us look into our analysis and formulate strategies for our objective.

### Pair-plot Analysis:
Pair-plot is nothing but a matrix of multiple features that represents a relationship among them on an individual basis. We can visualize how each independent features like temperature, humidity, windspeed, diffuse-flows, etc. are related to the output variable which is Zone 1. In this, the scatter-plots shows how strong a linear and non-linear datapoints are with respect to target variable, which can be really helpful indicators in a regression model. For example., if we visualize upward or downward plotting trend between humidity and zone 1 consumption, it reflects a possibility of correlation. Also, it helps in finding out multi-collinearity between variables or independent features, which is good for before model training because if two independent variables are highly correlated, it will be difficult for the model to determine how each of them impacts the model.

![image](https://github.com/user-attachments/assets/cba25e89-89b3-4d16-8978-6e04734cda70)

### Data Scaling/Data Standardization:
According to Geron,A -"Standardization (also called Z-score normalization) rescales the features so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one." Standardization is used to standardize the range of values of each feature to a similar scale. It is important because different features have different units or ranges, as it can help some features from dominating others in the model. 

![image](https://github.com/user-attachments/assets/a5be902c-32a2-4084-9f72-074ebce59820)

As we see in the figure above, the blue colored diagrams represent the distribution of datapoints before scaling and orange color graphs represent after scaling effects of the datapoints. Basically, what we have contracted/shrink the datapoints here after standardization. One thing to observe here, if you look at the shape of the histograms on both sides, it remains same i.e. the shape is not affected or the diagrams looks same on both sides. But if you check the values on x-axis, it has been standardized. Also, the mean and standard deviation is also shrinked after standardization that is the mean becomes 0 and standard deviation becomes 1 after Standard-scaling/Standardization.

### Data Preparation for Regression:
The datapoints as we have already scaled them with Standard-Scaler are now contracted. The dataset comprises of two parts, training and testing data. The testing data will be 20% of the total datapoints making it a ratio of 80:20, which makes it 80% of training data and 20% of testing data. The method which will be used here is going to be train_test_split from sklearn.model_selection. This division is of data is important because we have to ensure that the model is also being bought to the alien/unseen data which will provide a more clear results of the model and is going to help us in future for optimization of our model as well.


## Methodology & Model Selection

In this project, our goal is to predict the consumption of electricity usage in the zone 1 of Tetuan city which depends upon variety of factors such as temperature, humidity, diffuse-flows, wind-speed, etc. 

*Preprocessing the Dataset: It includes if missing values were checked, working on datetime column into meaningful columns like hours, minutes, day, day of the week, etc., z-score or standard scaling applied to the required columns.
  
*Train-test-split: The are approximately 5000 datapoints that we have. We have to train the model with 80:20 ratio. 80% will be the training data and 20% will be the testing data.

*Model-selection: We need to evaluate two models – Linear Regression & Random-forest regressor. Linear regression assumes a linear relationship between input and output features. Random-forest handle non-linearities and interaction between the input features.

*Evaluation-metrics: We will be using variety of metrics like R-square score, Mean Absolute error, Root mean squared error.

## Model Implementation & Evaluation

In this chapter, we are going to look into the practical aspect of the of machine learning. Basically, we are going to implement our models and we will be evaluating them based upon their performances. In order to predict electricity consumption in Zone 1, we worked with two regression models: Linear Regression & Random Forest Regression.
These were the libraries used:

![image](https://github.com/user-attachments/assets/044d4ffb-8dc4-4d59-b453-83b2f58d3a02)


### 1.	Linear Regression:
The independent variables in the analysis are temperature, humidity, wind-speed, general diffuse flows, diffuse flows, zone 2 power consumption, zone 3 power consumption. We have other time-based features as well which are day, month, hour, minute, day of the week, quarter of the year, day of the year. The dependent/target variable here is Zone 1 consumption. From scikit-learn library, we have had access to the Linear-Regression model or object of LinearRegression. The dataset was first standardized or was bought to common grounds during data scaling using StandardScaler. After that we did train_test_split from in the ratio of 80:20. After training the data using LinearRegression(), we made our prediction in with the testing data.


![image](https://github.com/user-attachments/assets/e6e65fa8-c141-42e6-bd1d-c51884b5873c)

This shows the performance of the Linear Regression model by using all the features. A scatter plot of Actual vs Predicted values have been shown in the above figure to get a better understanding of the model and better visualization. The red line above is known as the best-fit line and it is at a 45 degree angle as we can see the figure. The points which are closer to the line shows that the predicted values are close to the actual values and vice-versa.

### 2.	Random-Forest Regressor(RFE):
In RFE, for example we have 5000 datapoints and 8 features, it will split into nodes. Each node will perform on a sample of 5000 datapoints but it won’t select all the features it will select few of them randomly and on further extension of the same nodes again it selects the features randomly in the extended node. Here, I have taken n_estimators = 100 in my RFE model, so it will make 100 decision trees which will work with different permutation and combination of the output and in the end once every decision tree gives the output. It will give me average of those outputs in my prediction.


![image](https://github.com/user-attachments/assets/b4b75693-7372-4467-83ef-ae45e1b3b04c)

Here we can see the datapoints are more close to the best-fit line compared to the Linear Regression model. It is a good sign for the model and predictability of our output.

## Evaluating my models using Regression metrices:
For evaluating the performance of our models we are using three main regression metices : R-square score, MAE, MSE. These indicate how well my model is giving predicted output when compared with the actual output. In this case our output is Zone 1 electricity consumption.
 

![image](https://github.com/user-attachments/assets/d76a9240-7286-427f-9ccb-16c1025badfd)


![image](https://github.com/user-attachments/assets/76509d23-e338-4769-8b31-02a2c8782ebd)


As we can see the performance of both of our models. Evidently, RFE is outperforming our Linear Regression model by a significant margin. RFE is able to analyse non-linear relationships as well and also taking care of overfitting due to ensemble learning. Comparing and evaluating both my models for forecasting electricity consumption in Zone 1, RFE is quite preferrable.


## Conclusion:

•	Looking at the seasonal/monthly trends, the demand for electricity in zone 1 is highest from the month of April to the month of October.

•	The features that mostly affected the power consumption in Zone 1 are Temperature, Zone 2 & Zone 3.

•	With the KDE-plot, all the zones almost represent normal distribution of datapoints. Zone 1 and Zone 2 has more sharp edges while Zone 3 is comparatively flatter.

•	The RFE reflected more accuracy than the Linear regression model when compared with evaluation metrices i.e. R-score, MAE,MSE.

## Recommendation

•	Government or policy makers can ask households in the Zone 1 to reduce the power consumption during peak hours or during peak seasonality.

•	Our models can be used to predict seasonal demands which allows the electricity suppliers to manage the supply. For example, during less peak hours when people are hardly using electricity they can cut the power supply for energy saving and less wastage.

•	Also, we can do Zone wise consumption optimization. As, electricity consumption in zone 1 is highly correlated with zone 2, zone 3. For better utilization of energy, we can cut the energy in other zones during the peak in zone 1.

•	The electricity which is to be overproduced can be prevented using this forecasting.

•	We can set threshold for energy usage as well. For example, fixed number units used during this duration.

































