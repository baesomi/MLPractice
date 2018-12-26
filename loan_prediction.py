# dataset 불러오기
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# 다운로드 받은 데이터를 pandas의 dataframe형태로 읽어온다.
df = pd.read_csv("data/Loan/train_u6lujuX_CVtuZ9i.csv")

# 1. LoanAmount의 count를 보면 다른 column의 count보다 개수가 부족함. 즉, (614-692) 22 missing value
# 2. Loan_Amount_Term, Credit_History의 값도 LoanAmount와 동일하게 missing values 발생
# 3. Credit_History의 경우 값(0,1)을 갖고 있기 때문에, 평균(84%)는 credit_history를 갖고 있다. 라고 말할 수 있음
# 4. ApplicationIncome의 distribution은 CoapplicantIncome과 유사한 형태를 보여주고 있다.

# LoanAmount, ApplicantIncome은 extreme values를 갖고 있기 때문에, data munging이 필요하다.

'''
Categorical variable analysis
'''
# credit history 가 있을 경우 대출 할 확률이 높다 ( 없는 경우 대비 약 8배)

# Data Munging in Python: using Pandas
# 1. null value 처리
# 2. outlier의 영향력 무효화 (로그화)

# Check missing values in the dataset null, Nan
df.apply(lambda x: sum(x.isnull()),axis=0)

# numerical data는 simple하게 mean의 값으로 채워 넣는다.
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].fillna('No',inplace=True)
df['Credit_History'] = df['Credit_History'].fillna(1)

# 로그를 씌워준다
df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

'''
Model building
'''

# scikit-learn은 numerical data만 허용하기 때문에 categorical variables을 numeric하게 변경해야함
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].astype(str))


# Generic function for making a classification model and accessing performance:
def classification_model(model, model_name, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])
    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    print("====== " + model_name + " =====")
    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)

    error = []
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]

    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)

    # Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])




### Logistic Regression

# 여기서 predictor_var의 값이 NaN이면 안되기 때문에 위에서 0으로 채움
outcome_var = 'Loan_Status'
model = LogisticRegression()
model_name = 'Logistic Regression'
predictor_var = ['Credit_History']
classification_model(model, model_name, df,predictor_var,outcome_var)


### Decision Tree
model = DecisionTreeClassifier()
model_name = 'Decision Tree'
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model,model_name, df,predictor_var,outcome_var)

# 위의 모든 data가 categorical variables이기 때문에 Credit_History보다 impact가 없다... 그래서 numerical variables로 변경하면
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, model_name, df,predictor_var,outcome_var)
# 내용을 보면 accuracy는 올라갔지만, cross-validation score는 떨어진 것을 볼 수 있다. 즉 이 모델인 over-fitting됬다고 할 수 있다.



### Random Forest

# Random Forest의 장점은 featuer의 importances를 반환해준다는게 장점이다
model = RandomForestClassifier(n_estimators=100)
model_name = 'Random Forest'
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log', 'TotalIncome_log']
classification_model(model, model_name, df,predictor_var,outcome_var)
# Accuracy가 100%의 값이 나오는 것을 볼 수 있다. overffiting이 된건데, 해결하기 위한 방법은 아래와 같다.
# Reducing the number of predictors
# Tuning the model parameters

# 각 변수별 importance
feature_importance = pd.Series(model.feature_importances_, index = predictor_var).sort_values(ascending=False)
print("\n" + "Importance")
print(feature_importance)


model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log', 'LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, model_name, df,predictor_var,outcome_var)
# Accuracy가 낮아진 것을 볼 수 있지만, cross-validation의 값이 증가한것을 알 수 있다.
# random forest는 값을 돌릴때마다 randomize때문에 다소 다른 값을 나타낸다.