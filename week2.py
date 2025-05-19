#필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#1 데이터 불러오기
df = pd.read_csv('/Users/koyujung/Desktop/mlightenment/titanic/train.csv')

#데이터 구조 확인
print(df.shape)
print(df.info())
print(df.describe())

#2 결측치 시각화
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Titanic Dataset")
plt.show()

#3 결측치 처리
df.drop(columns=['Cabin'])
#Cabin은 전체의 약 77%가 결측 -> 삭제
df['Age'] = df['Age'].fillna(df['Age'].median())
#이상치 영향 -> 중앙값으로 단순 대체
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
#결측치가 2개뿐 -> 최빈값으로 단순 대체

#4 Age 컬럼의 이상치 탐지 및 제거
#Q1, Q3 계산
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

#이상치 기준 설정
#✅수정✅ #정규분포라면 약 99.3%의 데이터,양쪽 극단에 있는 약 0.7% (좌우 0.35%씩)가 이상치로 간주
lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR

#조건으로 필터링
df_cleaned = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]

#이상치 제거 전/후 박스플롯 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x=df['Age'], ax=axes[0])
axes[0].set_title('Before Outlier Removal')

sns.boxplot(x=df_cleaned['Age'], ax=axes[1])
axes[1].set_title('After Outlier Removal')
plt.tight_layout()
plt.show()

#5 Age 칼럼의 스케일링 비교
 # 라이브러리 불러오기
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv('/Users/koyujung/Desktop/mlightenment/titanic/train.csv')

# 전처리
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
df.dropna(inplace=True)  # 결측치 제거

# 범주형 → 숫자로 변환
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 입력/출력 나누기
X = df.drop(columns=['Survived'])
y = df['Survived']

# 학습용/테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 스케일링 함수 정의 (수정하지 마세요)
def scale_age(X_train, X_test, scaler):
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled['Age'] = scaler.fit_transform(X_train[['Age']])
    X_test_scaled['Age'] = scaler.transform(X_test[['Age']])
    return X_train_scaled, X_test_scaled

# ✅ KNN 평가 함수 (수정하지 마세요)
def evaluate_knn(X_train, X_test, y_train, y_test, label):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{label:<20} Accuracy: {acc:.4f}')

#✅ 아래 코드를 실행하려면 빈칸을 채워보세요
# 1. 스케일링 없이 원본 데이터로 평가
evaluate_knn(X_train, X_test, y_train, y_test, "No Scaling")

# 2. StandardScaler 적용
X_train_std, X_test_std = scale_age(X_train, X_test, StandardScaler())
evaluate_knn(X_train_std, X_test_std, y_train, y_test, "StandardScaler")

# 3. MinMaxScaler 적용
X_train_minmax, X_test_minmax = scale_age(X_train, X_test, MinMaxScaler())
evaluate_knn(X_train_minmax, X_test_minmax, y_train, y_test, "MinMaxScaler")

# 4. RobustScaler 적용
X_train_robust, X_test_robust = scale_age(X_train, X_test, RobustScaler())
evaluate_knn(X_train_robust, X_test_robust, y_train, y_test, "RobustScaler")

#6 전처리 전후의 데이터 구조 비교
# 전처리 전 (원본 데이터 기준)
원본행수 = 891  # Titanic train.csv 기본 행 수
원본열수 = 12   # Cabin 포함 기준

# 전처리 후
후행수 = df_cleaned.shape[0]
후열수 = df_cleaned.shape[1]

print(f"전처리 전 → 행: {원본행수}, 열: {원본열수}")
print(f"전처리 후 → 행: {후행수}, 열: {후열수}")

print("\n전처리 후 데이터 요약 info:")
print(df_cleaned.info())
