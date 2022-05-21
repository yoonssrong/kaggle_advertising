import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./data/advertising_pre2.csv')

# 라벨인코더 선언 및 Fitting
le = LabelEncoder()
le.fit(data['Country'])

# 인코딩한 데이터로 변환
le_encoded = le.transform(data['Country'])

# 결과물을 확인하면 array 형태로 나옵니다.
# 아래는 라벨 인코딩으로 만들어낸 데이터를 데이터 프레임으로 만들어주는 코드입니다.
data['Country'] = pd.DataFrame(le_encoded, columns = ['Country'])

le.fit(data['region'])

# 인코딩한 데이터로 변환
le_encoded = le.transform(data['region'])

# 결과물을 확인하면 array 형태로 나옵니다.
# 아래는 라벨 인코딩으로 만들어낸 데이터를 데이터 프레임으로 만들어주는 코드입니다.
data['region'] = pd.DataFrame(le_encoded, columns = ['region'])

le.fit(data['region_incomeLevel'])

# 인코딩한 데이터로 변환
le_encoded = le.transform(data['region_incomeLevel'])

# 결과물을 확인하면 array 형태로 나옵니다.
# 아래는 라벨 인코딩으로 만들어낸 데이터를 데이터 프레임으로 만들어주는 코드입니다.
data['region_incomeLevel'] = pd.DataFrame(le_encoded, columns = ['region_incomeLevel'])

data.to_csv('./data/advertising_pre3.csv', index=False)
