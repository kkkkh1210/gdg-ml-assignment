#1 데이터 불러오기
import pandas as pd
df = pd.read_csv('/Users/koyujung/Desktop/train.csv')
numeric_df = df.select_dtypes(include='number') #숫자형만 추출

#2 데이터 구조 파악
print("데이터 정보:")
print(df.info())
print()
print("\n숫자형 변수 요약:")
print(df.describe())
print()
print("\n처음 5행:")
print(df.head())
print()
print("\n마지막 5행:")
print(df.tail())

#3 결측치 탐색
print(df.isnull().sum())
#Age: 177개 -> 꽤 많음, Cabin: 687개 -> 제외 고려

#4 단일 변수 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 📊 생존 분포
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution (0=No, 1=Yes)')
plt.show()

# 📊 나이 분포
sns.histplot(df['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.show()

# 📊 성별에 따른 생존 분포
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# 📊 객실 등급과 생존률
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# 📊 변수 간 상관관계
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#5 분석 잘문 정의 및 탐색

# 🔍 질문1: 성별에 따라 생존률 차이가 있는가?

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()
print(df.groupby('Sex')['Survived'].mean())

# ✅ 해석:
#여성과 남성의 생존률은 각각 약 74%, 19% 수준
#성별은 생존 여부에 강한 영향을 주는 변수임

# 🔍 질문2: 나이에 따라 생존률이 달라지는가?

def age_category(age):
    if age < 13:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Elderly'

df['AgeGroup'] = df['Age'].apply(age_category)

sns.countplot(x='AgeGroup', hue='Survived', data=df)
plt.title('Survival by Age Group')
plt.show()

print(df.groupby('AgeGroup')['Survived'].mean())

# ✅ 해석:
#생존률: 어린이>성인>노인
#어린이: 생존자 많음, 성인: 생존자와 사망자 모두 많음, 노인: 사망자 많음
#구조 우선순위, 체력, 사회적 요인 등이 반영되었을 가능성

# 🔍 질문3: 가족과 함께 탑승한 사람의 생존률은 더 높았을까?

df['FamilySize'] = df['SibSp'] + df['Parch']
df['WithFamily'] = df['FamilySize'].apply(lambda x: 'Alone' if x == 0 else 'WithFamily') #가족 유무 변수
sns.countplot(x='WithFamily', hue='Survived', data=df)
plt.title('Survival: Alone vs With Family')
plt.show()
print(df.groupby('WithFamily')['Survived'].mean())

# ✅ 해석:
#생존률: 가족 동반>혼자 탑승
#가족 동승 여부가 생존률에 영향을 미쳤을 수 있음

#6 변수 조합 분석
# AgeGroup과 Sex에 따른 평균 생존률
print(pd.crosstab(df['AgeGroup'], df['Sex'], values=df['Survived'], aggfunc='mean'))

sns.catplot(x='AgeGroup', hue='Sex', col='Survived',
            data=df, kind='count', height=4, aspect=1)
plt.suptitle('Survival Count by Age Group and Gender', y=1.05)
plt.show()
# ✅ 해석:
#성인 남성은 사망자 매우 많음, 노인 남성은 사망 비율 높음
#성인 여성은 생존자 가장 많음, 노인 여성은 노인 남성에 비해 상대적으로 생존률 높음
#어린이는 남녀 생존 비슷함
#즉 어린이와 여성이 구조 우선시 됐음