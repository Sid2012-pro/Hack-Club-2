import pandas as pd
from sklearn.tree import DecisionTreeClassifier 


da = pd.read_csv('vgsales.csv')
da['RankCode'] = da['RankCode'].astype('category').catcodes
da['PlatformCodes'] = da['Platform'].astype('category').catcodes
da['YearCode'] = da['Year'].astype('category').catcodes
da['GenreCode'] = da['Genre'].astype('category').catcodes
da['PublisherCode'] = da['Publisher'].astype('category').catcodes
x = da['PlatformCodes','YearCode','PublisherCode','GenreCode']

y = da['Global_Sales', 'RankCode']
model = DecisionTreeClassifier()
model.fit(x,y)
print(x)
