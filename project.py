import numpy as np;
import pandas as pd;
import matplotlib.pyplot as pltt;
import seaborn as sns;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score;
from sklearn.ensemble import RandomForestRegressor


# Note: Sources used for graphs - Stack overflow, geeksforgeeks, python libraries

# Reading the dataset

dff = pd.read_csv('Tetuan City power consumption.csv');
#print(df);

# ['DateTime', 'Temperature', 'Humidity', 'Wind Speed',
#        'general diffuse flows', 'diffuse flows', 'Zone 1 Power Consumption',
#        'Zone 2  Power Consumption', 'Zone 3  Power Consumption']



print(dff.info());      # checkiing datatype of columns

print(dff.describe());     # statistics of the data

print(dff.isnull().sum());    # checking the null values



# SPLITTING DATETIME COLUMN TO DAYS,MONTHS,HOUR,MINUTE, DAY OF THE WEEK, QUARTERS

dff['DateTime'] = pd.to_datetime(dff['DateTime']);
#print(df['DateTime']);

dff['Day'] = dff['DateTime'].dt.day;
#print(df['Day']);
dff['Hour'] = dff['DateTime'].dt.hour + 1;

dff['Month'] = dff['DateTime'].dt.month;

dff['Day of the week'] = dff['DateTime'].dt.day_of_week + 1;

dff['Minute'] = dff['DateTime'].dt.minute;

#print(df['Day of the week'].value_counts());

# Defininf quarters with the help of months

def quarters(data):
    if data['Month'] <= 3:
        return 1
    elif data['Month'] > 3 and data['Month'] <=6:
        return 2
    elif data['Month'] > 6 and data['Month'] <=9:
        return 3
    elif data['Month'] > 9 and data['Month'] <=12:
        return 4



dff['Quarters of the year'] = dff.apply(quarters,axis=1);

dff['Day of Year'] = dff['DateTime'].dt.strftime('%j').astype(int);

# Dropping the datetime column because we have split them into months, hours, quarters etc.

dff.drop('DateTime', axis=1,inplace=True)

#print(df['Hour'].isnull().sum())
 

# print(df.columns);


 
# Monthly distribution/trend Analysis
dff.groupby('Month')[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']].mean().plot(figsize=(15, 7))
pltt.grid(True)
pltt.title("Average Monthly Power Consumption")

pltt.ylabel("Average Consumption of Power")
pltt.xlabel("Monthly")
pltt.show()


# Correlation Heatmap Analysis

correl = dff.corr();
pltt.figure(figsize=(18, 10))
pltt.title("Heatmap")
sns.heatmap(correl, annot=True, cmap='coolwarm', fmt='.2f', square=True)

pltt.show()


# # Create a new DataFrame for boxplot
# consumption_df = dff[['Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption']]
# consumption_df.columns = ['Zone 1', 'Zone 2', 'Zone 3']  # Rename for cleaner axis labels


# KDE-Plot ANalysis

pltt.figure(figsize=(16, 9))

sns.kdeplot(dff['Zone 1 Power Consumption'], fill=True, label='Zone 1')
sns.kdeplot(dff['Zone 2  Power Consumption'], fill=True, label='Zone 2')
sns.kdeplot(dff['Zone 3  Power Consumption'], fill=True, label='Zone 3')

pltt.title('Density Plot on the basis of zone')

pltt.ylabel('Density')
pltt.xlabel('Consumption of Electricity')
pltt.legend()
pltt.show()




# Analysing the pair-plot

sns.pairplot(dff[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows', 'Zone 1 Power Consumption']], palette='winter')
pltt.show()


# Standardizing/ Scaling the data 

scl = StandardScaler();    

# features on which I want to do scaling not taking the time-features like month, year,minutes, hour
inps_for_scl = ['Temperature', 'Humidity', 'Wind Speed','general diffuse flows', 'diffuse flows','Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']

print(inps_for_scl);

arr_scl = scl.fit_transform(dff[inps_for_scl]);     #scaled array

df_scl = pd.DataFrame(arr_scl,columns= inps_for_scl);     # converting the scaled array to dataframe because array was scaled in numpy array
print(df_scl.columns);


# Visualising before and after scaling  -- took some help from stack overflow

len_features = len(inps_for_scl);    # Defining the length of iterations


pltt.figure(figsize=(16, len_features * 1.0))

for i, feature in enumerate(inps_for_scl):
    pltt.subplot(len_features, 2, 2*i + 1)
    sns.histplot(dff[feature], kde=True, bins=30, color='steelblue')
    pltt.title(f'{feature} (Before-scaling)')
    
    pltt.subplot(len_features, 2, 2*i + 2)
    sns.histplot(df_scl[feature], kde=True, bins=30, color='darkorange')
    pltt.title(f'{feature} (After-scaling)')
    



pltt.tight_layout();
pltt.show();


# Connecting time features like month, hour, week, minutes etc to the dataframe because while scaling we dropped all those features

time_inps = dff[['Day','Month','Hour','Minute','Day of the week','Quarters of the year','Day of Year']].reset_index(drop=True);
df_scaled_full = pd.concat([df_scl,time_inps],axis=1);
print(df_scaled_full.columns);


# Linear Regression 

x = df_scaled_full.drop(columns= 'Zone 1 Power Consumption');  # input variables
y = df_scaled_full['Zone 1 Power Consumption'];               # target variable
#print(x);

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state= 42);  # 80% of training

lr = LinearRegression();
lr.fit(x_train,y_train);

y_pred = lr.predict(x_test);
#print(y_pred)


# Evaluate

print("r2-score lr:",r2_score(y_test,y_pred));

print("MAE-lr:",mean_absolute_error(y_test,y_pred));

print("MSE-lr:",mean_squared_error(y_test,y_pred));


# Visualising Best-fit line and checking diff. between actual and predicted values

pltt.figure(figsize=(16,10));
sns.scatterplot(x=y_test, y=y_pred, color='blue');

# creating a diagonal from bottom left- to right
pltt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color='red'); 

pltt.ylabel('Predicted output-Zone 1 power-consumption');
pltt.xlabel('Actual output-Zone 1 power-consumption');

pltt.title('Actual vs predicted values (Linear Regression)');
pltt.grid(True);
pltt.show();





#Random Forest 

x = df_scaled_full.drop(columns= 'Zone 1 Power Consumption');
y = df_scaled_full['Zone 1 Power Consumption'];
#print(x);

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state= 42);

rf = RandomForestRegressor(n_estimators=100, random_state=42);
rf.fit(x_train, y_train);

y_pred = rf.predict(x_test);


print("MAE rfe:",mean_absolute_error(y_test, y_pred));

print("MSE rfe:",mean_squared_error(y_test, y_pred));
print("r-score Score rfe:",r2_score(y_test, y_pred));


# R2 Score lr: 0.8673926924734852          # R² Score rf: 0.9924350565989764
# MAE lr: 0.280810174320352                # MAE rf: 0.05606611126222478
# MSE lr: 0.13169559609594034              # MSE rf: 0.007512932350509082


# Visualising RFE output

pltt.figure(figsize=(18, 16));
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='green');


pltt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')  # 45-degree line

pltt.xlabel('Actual');
pltt.ylabel('Predicted');

pltt.title('Actual vs predicted zone1 Power-Consumption');


pltt.grid(True);
pltt.show();



#****************************************************************************************************************































































