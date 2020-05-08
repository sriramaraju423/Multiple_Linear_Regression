# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cars = pd.read_csv("C:/Datasets_BA/Linear Regression/cars.csv")
cars.describe()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
                             
# Correlation matrix 
cars.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit() # regression model

# Summary
ml1.summary()

# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model

# final model
final_ml= smf.ols('MPG~VOL+SP+HP',data = cars).fit()
final_ml.summary()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train,cars_test  = train_test_split(cars,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols("MPG~HP+SP+VOL",data=cars_train).fit()

# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.MPG

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 4.04 

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid  = test_pred - cars_test.MPG

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 3.16
