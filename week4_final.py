# Titanic: 최종 모델 선정 및 예측 제출 파이프라인 (RandomForest 최종 선택)

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 불러오기
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/titanic/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/titanic/test.csv")

# 제출용 ID 저장
test_ids = test['PassengerId']

# 2. 전처리 함수 정의
def preprocess(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.drop(columns=['Cabin', 'Name', 'Ticket'])
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(columns=['Survived', 'PassengerId'])
y = train_proc['Survived']
X_test_final = test_proc.drop(columns=['PassengerId'])

# 컬럼 순서 일치
X_test_final = X_test_final[X.columns]

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# 4. 모델 정의 및 성능 비교
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_scaled, y)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    results[name] = {
        'model': model,
        'mean_cv_accuracy': scores.mean()
    }

# 5. RandomForest를 최종 모델로 선택 (Kaggle 결과 기준)
best_model_name = 'RandomForest'
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['mean_cv_accuracy']

# 6. 테스트셋 예측 및 제출 파일 생성
preds = best_model.predict(X_test_scaled)
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': preds
})

# 저장
filename = f"titanic_best_{best_model_name}.csv"
submission.to_csv(filename, index=False)
print(f"Best model: {best_model_name}, Accuracy: {best_accuracy:.4f}, File saved: {filename}")