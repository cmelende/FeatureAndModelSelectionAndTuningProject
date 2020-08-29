from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class Metrics:

    def __init__(self,df,test_size, target_col):
        self.dataframe = df
        self.target_column = target_col
        self.X = self.dataframe.drop(target_col, axis=1)
        self.Y = self.dataframe[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=1)
    
    def GetGradientScore(self):
        gradient = GradientBoostingRegressor(n_estimators = 50,random_state=1)
        gradient = gradient.fit(self.X_train, self.y_train)
        y_predict = gradient.predict(self.X_test)

        model_score = gradient.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(gradient, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict

    # def GetGradientPrediction(self):
    #     gradient = GradientBoostingRegressor(n_estimators = 50,random_state=1)
    #     gradient = gradient.fit(self.X_train, self.y_train)
    #     y_predict = gradient.predict(self.X_test)

    #     return y_predict

    def GetBoostingScore(self):
        boosting = AdaBoostRegressor(n_estimators=10, random_state=1)
        boosting = boosting.fit(self.X_train, self.y_train)
        y_predict = boosting.predict(self.X_test)

        model_score = boosting.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(boosting, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict

    # def GetBoostingPrediction(self):
    #     boosting = AdaBoostRegressor(n_estimators=10, random_state=1)
    #     boosting = boosting.fit(self.X_train, self.y_train)
    #     y_predict = boosting.predict(self.X_test)

    #     return y_predict

    def GetBaggingScore(self): 
        dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
        dTree.fit(self.X_train, self.y_train)
        bgcl = BaggingRegressor(base_estimator=dTree, n_estimators=50,random_state=1)
        bgcl = bgcl.fit(self.X_train, self.y_train)
        y_predict = bgcl.predict(self.X_test)

        model_score = bgcl.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(bgcl, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict
    # def GetBaggingPrediction(self): 
    #     dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
    #     dTree.fit(self.X_train, self.y_train)
    #     bgcl = BaggingRegressor(base_estimator=dTree, n_estimators=50,random_state=1)
    #     bgcl = bgcl.fit(self.X_train, self.y_train)
    #     y_predict = bgcl.predict(self.X_test)

    #     return y_predict

    def GetBaggingRegressorModel(self):
        dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
        dTree.fit(self.X_train, self.y_train)
        bgcl = BaggingRegressor(base_estimator=dTree, n_estimators=50,random_state=1)
        bgcl = bgcl.fit(self.X_train, self.y_train)
        y_predict = bgcl.predict(self.X_test)

        model_score = bgcl.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(bgcl, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict

    # def GetBaggingRegressorPrediction(self):
    #     dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
    #     dTree.fit(self.X_train, self.y_train)
    #     bgcl = BaggingRegressor(base_estimator=dTree, n_estimators=50,random_state=1)
    #     bgcl = bgcl.fit(self.X_train, self.y_train)
    #     y_predict = bgcl.predict(self.X_test)

    #     return y_predict

    def GetDecisionTreeScore(self):
        dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
        dTree.fit(self.X_train, self.y_train)
        y_predict = dTree.predict(self.X_test)

        model_score = dTree.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(dTree, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict

    # def GetDecisionTreePrediction(self):
    #     dTree = DecisionTreeRegressor(criterion = 'mse', random_state=1)
    #     dTree.fit(self.X_train, self.y_train)
    #     y_predict = dTree.predict(self.X_test)

    #     return y_predict



    def GetLinearRegressionScore(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        y_predict = model.predict(self.X_test)

        model_score = model.score(self.X_test, self.y_test)

        # num_folds = 50
        # seed = 7
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        # kfoldResults = cross_val_score(model, self.X, self.Y, cv=kfold, scoring='roc_auc')

        return model_score, y_predict

    # def GetLinearRegressionPrediction(self):
    #     model = LinearRegression()
    #     model.fit(self.X_train, self.y_train)

    #     y_predict = model.predict(self.X_test)

    #     return y_predict

    # def GetLinearRegressionModel(self):
    #     model = LinearRegression()
    #     model.fit(self.X_train, self.y_train)

    #     return model


    
    def GetScoreDataframe(self):
        gradient_score, gradient_predict = self.GetGradientScore()
        boosting_score, boosting_predict = self.GetBoostingScore()
        bagging_score, bagging_predict = self.GetBaggingScore()
        dtree_score, dtree_predict = self.GetDecisionTreeScore()
        lin_score, lin_predict  = self.GetLinearRegressionScore()

        gradient_results = self.regression_results(self.y_test, gradient_predict)
        boosting_results = self.regression_results(self.y_test, boosting_predict)
        bagging_results = self.regression_results(self.y_test, bagging_predict)
        dtree_results = self.regression_results(self.y_test, dtree_predict)
        lin_results = self.regression_results(self.y_test, lin_predict)



        i = [
            'gradient', 
            'boosting', 
            'bagging', 
            'dtree', 
            'linear'
            ]
        data = {
            'score': [
                gradient_score,
                boosting_score,
                bagging_score,
                dtree_score,
                lin_score
                ],
            'explained variance': [
                gradient_results[0], 
                boosting_results[0],
                bagging_results[0],
                dtree_results[0],
                lin_results[0]
                ],
            'mean abs error': [
                gradient_results[1],
                boosting_results[1],
                bagging_results[1],
                dtree_results[1],
                lin_results[1]
                ],
            'mse': [
                gradient_results[2],
                boosting_results[2],
                bagging_results[2],
                dtree_results[2],
                lin_results[2]
                ],
            'mean squared log error': [
                gradient_results[3],
                boosting_results[3],
                bagging_results[3],
                dtree_results[3],
                lin_results[3]
                ],
            'median abs error': [
                gradient_results[4],
                boosting_results[4],
                bagging_results[4],
                dtree_results[4],
                lin_results[4]
                ],
            'r2': [
                gradient_results[5],
                boosting_results[5],
                bagging_results[5],
                dtree_results[5],
                lin_results[5]
                ],
            # 'kfold': [
            #     gradient_kfold,
            #     boosting_kfold,
            #     bagging_kfold,
            #     dtree_kfold,
            #     lin_kfold
            
        }
        return pd.DataFrame(data, columns = ['score','explained variance','mean abs error','mse','mean squared log error','median abs error','r2'], index=i)


    def regression_results(self, y_test, y_predict):
        # Regression metrics
        explained_variance=metrics.explained_variance_score(y_test, y_predict)
        mean_absolute_error=metrics.mean_absolute_error(y_test, y_predict) 
        mse=metrics.mean_squared_error(y_test, y_predict) 
        mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_predict)
        median_absolute_error=metrics.median_absolute_error(y_test, y_predict)
        r2=metrics.r2_score(y_test, y_predict)

        return explained_variance, mean_absolute_error, mse, mean_squared_log_error, median_absolute_error, r2

        # print('explained_variance: ', round(explained_variance,4))    
        # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
        # print('r2: ', round(r2,4))
        # print('MAE: ', round(mean_absolute_error,4))
        # print('MSE: ', round(mse,4))
        # print('RMSE: ', round(np.sqrt(mse),4))