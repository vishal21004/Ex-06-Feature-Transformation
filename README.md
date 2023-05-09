# Ex-06-Feature-Transformation


## AIM :

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM :

### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM :

Name:M.A.Vishal

Reg No. 212222230177
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

## OUTPUT :

![image](https://user-images.githubusercontent.com/119560261/232929125-9394a43e-a332-4ec3-ac01-c893ce7c944c.png)

![image](https://user-images.githubusercontent.com/119560261/232929137-af00a749-28e0-460b-bd5e-91ac4d010114.png)

![image](https://user-images.githubusercontent.com/119560261/232929160-16983c30-b0a1-46cb-9813-5f87ddcacdd8.png)

![image](https://user-images.githubusercontent.com/119560261/232929183-48a03172-56e2-4da1-8781-8d1654aa1a56.png)

![image](https://user-images.githubusercontent.com/119560261/232929196-b16e4148-11b4-49db-a903-0cbd91398ef7.png)

![image](https://user-images.githubusercontent.com/119560261/232929235-7151aa56-5dcf-4edc-b418-240211620821.png)

![image](https://user-images.githubusercontent.com/119560261/232929261-24e13d83-8f1e-47ec-8caf-af5c7a20b2e6.png)

![image](https://user-images.githubusercontent.com/119560261/232929271-f560cb5b-1011-4bc8-a7b6-9238e5e849d6.png)

![image](https://user-images.githubusercontent.com/119560261/232929278-6be45382-eb0f-4888-96b8-eea5947cc6c7.png)

![image](https://user-images.githubusercontent.com/119560261/232929296-fa37c705-7842-42c1-8064-4cbe962fbf19.png)

## RESULT :

Thus feature transformation is done for the given dataset.
