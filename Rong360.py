import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc

# delete duplicated data

order = pd.read_csv(r"order.csv")
product = pd.read_csv(r"product.csv")
user = pd.read_csv(r"user.csv")
quality = pd.read_csv(r"quality.csv")

# retain the most recent data
user = user.sort_values(['user_id', 'date'], ascending = False)
#[order(user$user_id, -user$date),]
user.drop_duplicates('user_id', inplace = True)
del user['date']
#user <- subset(user, select = -date)

# merge data
dataset = order.merge(user, on = "user_id", how = 'left')
dataset = dataset.merge(product, on = "product_id", how = 'left')
dataset = dataset.merge(quality, on = ["user_id","limit","term","standard_type","guarantee_type"], how = 'left')

del order, product, user, quality

# delete misssing value
delete = []
for col in dataset.columns:
    if sum(dataset[col].isnull()) / len(dataset[col]) >= 0.5:
        delete.append(col)
dataset.drop(delete, axis = 1,inplace = True)
#
#for th in np.arange(0, 1, 0.1):
#    delete = []
#    for col in dataset.columns:
#        if sum(dataset[col].isnull()) / len(dataset[col]) >= th:
#            delete.append(col)
#    print(th, len(delete), len(delete)/133)

#colname = []
#nullno = []
#for col in dataset.columns:
#    nullno.append(sum(dataset[col].isnull()))
#    colname.append(col)
#res = pd.DataFrame({'col':colname, 'null': nullno})


#删除列之后就没有了
#delete = []
#for row in dataset.index:
#    if sum(dataset.iloc[row, :].isnull()) / len(dataset.iloc[row, :]) >= 0.7:
#        delete.append(row)
#dataset.drop(delete, axis = 0, inplace = True)

#
#for th in np.arange(0, 1, 0.1):
#    delete = []
#    for row in dataset.index:
#        if sum(dataset.iloc[row, :].isnull()) / len(dataset.iloc[row, :]) >= th:
#            delete.append(row)
#    print(th, len(delete), len(delete)/len(dataset))
#

# delete useless variables
delete = ['bank_id_x', 'bank_id_y', 'city_id_x', 'city_id_y', 'user_id', 
          'product_id','product_type_y']
dataset.drop(delete, axis = 1, inplace = True)

dataset['limit'] = dataset['limit'].apply(lambda x : float(x))
dataset['date'] = dataset['date']%7

#dataset.info()
factor = ['application_type', 'bank', 'col_type', 'tax', 'guarantee_required','loan_term_type',
'business_license', 'car', 'guarantee_type', 'house', 'house_register', 'id', 'income',
'interest_rate_type', 'qid77','lifecost', 'married', 'mobile_verify', 'op_type',
'platform', 'product_type_x', 'quality', 'repayment_type', 'socialsecurity',
'user_loan_experience', 'is_paid','date', 'standard_type']

for f in factor:
    lbl = preprocessing.LabelEncoder()
    dataset[f] = lbl.fit_transform(list(dataset[f].values))

#colname = []
#std = []
#for col in dataset.columns:
#    colname.append(col)
#    std.append(np.nanstd(dataset[col]))
#res = pd.DataFrame({'col': colname, 'std': std})

# delete low variance
delete = []
for col in dataset.columns:
    if col in factor or col == 'result':
        continue
    if np.nanstd(dataset[col]) < 0.005:
        delete.append(col)
dataset.drop(delete, axis = 1, inplace = True)

sum(dataset['result'] == 0)/len(dataset)

# partition the data
x_train_df, x_test_df, y_train, y_test = train_test_split(dataset.drop(['result'], axis = 1), dataset['result'], test_size = 0.3, random_state = 13)

# ratio of positive sample and negative sample
sumneg = sum(y_train == 0)
sumpos = sum(y_train == 1)

# sum(y_train == 0)/len(y_train)
# sum(y_test == 0)/len(y_test)


from sklearn.preprocessing import Imputer

# change dataframe as matrix
x_train = Imputer().fit_transform(x_train_df)
x_test = Imputer().fit_transform(x_test_df)


#std = []
#std_nan = []
#for col in range(x_train.shape[1]):
#    std.append(np.nanstd(x_train[:, col]))
#    std_nan.append(np.nanstd(x_train_df.iloc[:, col]))
#res = pd.DataFrame({'std':std, 'std_nan':std_nan, 'index': x_train_df.columns})

# single model - XGBoost
#x_train = x_train.as_matrix()
#x_test = x_test.as_matrix()

#param <- list(max_depth = 25, eta = 0.1,
#              objective = "binary:logistic", 
#              eval_metric = "auc",
#              scale_pos_weight = sumneg / sumpos,
#              silent = 1,
#              nthrea= 16,
#              max_delta_step=4,
#              subsample=0.8,min_child_weigth=2)
#set.seed(13)
#XG <- xgboost(data = train, label = result, params =param, nrounds = 100)


from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


#clf1 = SVC(C = 0.99, kernel = 'linear', probability = True, verbose = 1) 
# SVC is slow
clf2 = RandomForestClassifier(random_state = 13, n_estimators = 200, verbose = 1)
clf3 = LogisticRegression(random_state = 13, verbose = 1)
#clf4 = MultinomialNB(alpha = 0.1)
clf4 = xgb.XGBClassifier(max_depth = 25, learning_rate = 0.1, objective = "binary:logistic",
                    scale_pos_weight = sumneg / sumpos, silent = 0, nthread = 16,
                    max_delta_step = 4, subsample = 0.8, min_child_weigth = 2, seed = 13,
                    n_estimators = 100, verbose = 1) 

#eclf_hard = VotingClassifier(estimators=[('SVC', clf1), ('rf', clf2), ('lr', clf3), ('xgb', clf4)], voting = 'hard')
#eclf_soft = VotingClassifier(estimators=[('SVC', clf1), ('rf', clf2), ('lr', clf3), ('xgb', clf4)], voting = 'soft')

eclf_hard = VotingClassifier(estimators=[('rf', clf2), ('lr', clf3), ('xgb', clf4)], voting = 'hard')
eclf_soft = VotingClassifier(estimators=[('rf', clf2), ('lr', clf3), ('xgb', clf4)], voting = 'soft')


result = pd.DataFrame()
#for clf, label in zip([clf1, clf2, clf3, clf4, eclf_hard, eclf_soft], ['SVC', 'Random Forest', 'Logistic Reg', 'XGBBoosting', 'Ensemble_hard', 'Ensemble_soft']):
for clf, label in zip([clf2, clf3, clf4, eclf_hard, eclf_soft], ['Random Forest', 'Logistic Regression', 'XGBBoost', 'Hard Voting', 'Soft Voting']):
    print(label)
    clf.fit(x_train, y_train)
    if label == 'Hard Voting':
        result[label] = clf.predict(x_test)
        continue
    result[label] = clf.predict_proba(x_test)[:, 1]
    
result['true'] = y_test.values

labels = [0, 1]
for l in ['Random Forest', 'Logistic Regression', 'XGBoost', 'Hard Voting', 'Soft Voting']:
#	 打印结果
#    pred = result[l] >= 0.5
#    print(l)
#    print(classification_report(y_test, pred, target_names = ['neg', 'pos']))
#    print(confusion_matrix(y_test, pred, labels = labels))
#    print(accuracy_score(y_test, pred))
    fpr, tpr, thresholds = roc_curve(y_test, result[l])  
    roc_auc = auc(fpr, tpr) 
    plt.plot(fpr, tpr, lw = 1, label='AUC of %s  = %0.2f)' % (l,roc_auc))      
#画对角线  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
plt.legend()
plt.savefig(r"result_auc.png")
plt.show()  

# profit analysis
cal = pd.DataFrame({"limit":x_test_df['limit'].values, "real": y_test.values, "predict": result['Soft Voting'].values})

alpha = 0.15
referfee = 50
penaty = 60

profit = []
for th in np.arange(0, 1, 0.01):
    TP = sum(cal[cal['predict'] >= th]['real'] == 1)
    FP = sum(cal[cal['predict'] >= th]['real'] == 0)
    FN = sum(cal[cal['predict'] < th]['real'] == 1)
    p = TP * referfee + FP*(referfee - penaty) - FN * referfee 
    + alpha * sum(cal[cal['predict'] >= th][cal['real'] == 1]['limit'])
    - alpha * sum(cal[cal['predict'] < th][cal['real'] == 1]['limit'])
    profit.append(p)

plt.plot(np.arange(0, 1, 0.01), profit, label = "Max Profit = %0.2f, Best Threshold = 0.23" % max(profit))
plt.title("x : Threshold; y : Profit")
plt.legend()
plt.savefig(r"profit.png")







