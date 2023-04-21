# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
# STEP 1
Read the given Data

# STEP 2
Clean the Data Set using Data Cleaning Process

# STEP 3
Apply Feature Transformation techniques to all the features of the data set

# STEP 4
Save the data to the file

# CODE:
```
Name : VAISHNAVI S
Register Number : 212222230165
```
# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```
# OUPUT:
# Feature Transformation - Data_to_Transform.csv
![Screenshot 2023-04-18 114930](https://user-images.githubusercontent.com/118541897/233544009-205936cb-5e8f-40f8-b7e4-fc3c84969957.png)
![Screenshot 2023-04-18 114943](https://user-images.githubusercontent.com/118541897/233544055-5dd9a663-3bb0-4a5b-a60e-a997d0b44a76.png)
![Screenshot 2023-04-18 114955](https://user-images.githubusercontent.com/118541897/233544088-c00ed49e-739d-4f0e-816b-20cf99dd8bd6.png)
![Screenshot 2023-04-18 115010](https://user-images.githubusercontent.com/118541897/233544106-67a894d5-5487-46ec-ad45-d8a73b00611d.png)
![Screenshot 2023-04-18 115025](https://user-images.githubusercontent.com/118541897/233544123-123f980a-55d9-4c6c-989e-5aa2b0aa4776.png)
![Screenshot 2023-04-18 115031](https://user-images.githubusercontent.com/118541897/233544145-571969ca-5ba0-4b36-88c1-6a00b8bb1e36.png)
![Screenshot 2023-04-18 115037](https://user-images.githubusercontent.com/118541897/233544225-23f612ab-5ed2-404c-b0ef-56b9176cc23b.png)
![Screenshot 2023-04-18 115042](https://user-images.githubusercontent.com/118541897/233544236-cd68bc07-d6ca-460a-9718-17dfb21d8133.png)

# Log Transformation
![Screenshot 2023-04-18 115047](https://user-images.githubusercontent.com/118541897/233544270-0a28d18c-74da-4e19-92cd-8177c1a01e65.png)

# Reciprocal Transformation
![Screenshot 2023-04-18 115051](https://user-images.githubusercontent.com/118541897/233544283-68dc7e1f-937c-47c8-831f-aa13b027983c.png)

# SquareRoot Transformation
![Screenshot 2023-04-18 115056](https://user-images.githubusercontent.com/118541897/233544313-5f1264b1-a148-4409-8d7a-c173cc014d3d.png)

# Power Transformation
![Screenshot 2023-04-18 115101](https://user-images.githubusercontent.com/118541897/233544556-7e696943-5ac2-4b6d-87ff-f79dcd968083.png)
![Screenshot 2023-04-18 115107](https://user-images.githubusercontent.com/118541897/233544567-cb803d08-c1f3-4988-936e-9a3398f78882.png)

# Quantile Transformation
![Screenshot 2023-04-18 115112](https://user-images.githubusercontent.com/118541897/233544578-fa93715a-95e1-4c05-8343-a62de5e42d7a.png)

# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
