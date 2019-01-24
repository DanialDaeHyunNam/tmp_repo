import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

class ClassicModelTrain:
    def __init__(self, preprocessed_data, is_run_logistic = True, is_run_multi_NB = True, 
                 is_run_random = True, is_run_extra_random = True, is_run_svc = True, is_run_xgboost = True):
        self.is_run_logistic = is_run_logistic
        self.is_run_multi_NB = is_run_multi_NB
        self.is_run_random = is_run_random
        self.is_run_extra_random = is_run_extra_random
        self.is_run_svc = is_run_svc 
        self.is_run_xgboost = is_run_xgboost
        self.preprocessed_data = preprocessed_data
        
    def randomly_selected_classification(self, model):
        x = self.X_data
        y = self.y_data
        idx = np.random.choice(x.shape[0], 1000, replace=False)
        x_test = x[idx]
        y_true = y[idx]
        if self.preprocessed_data.is_vectorize:
            x_test = self.preprocessed_data.get_vectorized_data_to_eval(x_test, self.preprocessed_data.is_tfidf_vect)
        y_pred = model.predict(x_test)
        print(classification_report(y_true, y_pred))
        
        def display_(str_):
            display(Markdown(str(str_)))
        
        display_("##### Ten-fold cross-validation accuracy score : ")
        ac_score = np.zeros(10)
        for i in list(range(10)):
            idx = np.random.choice(x.shape[0], 1000, replace=False)
            x_test = x[idx]
            y_true = y[idx]
            if self.preprocessed_data.is_vectorize:
                x_test = self.preprocessed_data.get_vectorized_data_to_eval(x_test, self.preprocessed_data.is_tfidf_vect)
            y_pred = model.predict(x_test)
            ac_score[i] = accuracy_score(y_true, y_pred)
        display_("##### : " + str(ac_score.mean()*100))
    
    def print_classification_report(self, model, x, y_true):
        y_pred = model.predict(x)
        print(classification_report(y_true, y_pred))
                
    def run_all(self, num_boost_round = 10000, lr = 0.01, max_delta_step = 4):
        self.X_data, self.y_data = self.preprocessed_data.get_original_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessed_data.get_preprocessed_data()
        
        def display_(str_):
            display(Markdown(str(str_)))
        
        display_("##### shape of X train and X_test : ")
        print(self.X_train.shape, self.X_test.shape)
        
        if self.is_run_logistic:
            display_("#### Run logistic")
            self.logis_model = LogisticRegression().fit(self.X_train, self.y_train)
            display_("##### Logistic Regression classification report : ")
            self.print_classification_report(self.logis_model, self.X_test, self.y_test)
            display_("##### Logistic Regression <span style='color: red'>randomly selected 1000 data</span> classification report : ")
            self.randomly_selected_classification(self.logis_model)
            
        if self.is_run_multi_NB:
            display_("#### Run multinomialNB")
            self.multi_model = MultinomialNB().fit(self.X_train, self.y_train)
            display_("##### MultinomialNB classification report : ")
            self.print_classification_report(self.multi_model, self.X_test, self.y_test)
            display_("##### MultinomialNB <span style='color: red'>randomly selected 1000 data</span> classification report : ")
            self.randomly_selected_classification(self.multi_model)
        
        if self.is_run_random:
            display_("#### Run random forest")
            parameters = {'n_estimators': np.arange(100, 200, 10),'max_depth': np.arange(1, 10, 2)}
            kfold = KFold(10)
            random_model = RandomForestClassifier(random_state=0)
            grid_model = GridSearchCV(random_model, parameters, scoring='accuracy', cv = kfold, n_jobs = -1)
            grid_model.fit(self.X_train, self.y_train)
            params_ls = grid_model.cv_results_['params']
            mean_test_score_ls = grid_model.cv_results_["mean_test_score"]
            plt.plot(mean_test_score_ls)
            plt.title("Random forest test score")
            plt.show()
            display_(grid_model.best_score_)
            display_(grid_model.best_params_)
        
        if self.is_run_extra_random:
            display_("#### Run extra random forest")
            parameters = {'n_estimators': np.arange(100, 200, 10),'max_depth': np.arange(1, 10, 2)}
            kfold = KFold(10)
            extra_model = ExtraTreesClassifier(random_state=0)
            grid_model = GridSearchCV(extra_model, parameters, scoring='accuracy', cv=kfold, n_jobs=-1)
            grid_model.fit(self.X_train, self.y_train)
            params_ls = grid_model.cv_results_['params']
            mean_test_score_ls = grid_model.cv_results_["mean_test_score"]
            plt.plot(mean_test_score_ls)
            plt.title("Extra random forest test score")
            plt.show()
            display_(grid_model.best_score_)
            display_(grid_model.best_params_)
        
        if self.is_run_svc:
            self.svc_model = SVC(kernel='rbf',random_state=0).fit(self.X_train, self.y_train)
            display_("##### Kernel Support Vector Machine classification report : ")
            self.print_classification_report(self.svc_model, self.X_test, self.y_test)
            display_("##### Kernel Support Vector Machine <span style='color: red'>randomly selected 1000 data</span> classification report : ")
            self.randomly_selected_classification(self.svc_model)
        
        if self.is_run_xgboost:
            display_("#### Run xgboost")
            label_enc = LabelEncoder().fit(self.y_data)

            def get_encoded_target(y, label_enc):
                return label_enc.transform(y)

            def xgb_preprocess(x, y, is_need_to_be_encoded=False):
                y_label = y
                if is_need_to_be_encoded:
                    y_label = get_encoded_target(y, label_enc) 
                
                return xgboost.DMatrix(x, label = y_label)

            def set_predicted_values(model, x, y):
                test_y = get_encoded_target(y, label_enc)
                test_X = xgb_preprocess(x, test_y)
                y_pred_proba = model.predict(test_X)
                y_pred = [np.argmax(line) for line in y_pred_proba]
                y_true = label_enc.inverse_transform(test_y)
                y_pred = label_enc.inverse_transform(y_pred)
                return y_true, y_pred
            
            def print_classification_report_xgb(model, x, y):
                y_true, y_pred = set_predicted_values(model, x, y)
                print(classification_report(y_true, y_pred))
                
            def print_classification_report_xgb_1000(model):
                x = self.X_data
                y = self.y_data
                idx = np.random.choice(x.shape[0], 1000, replace=False)
                
                x_test = x[idx]
                y_true = y[idx]
                if self.preprocessed_data.is_vectorize:
                    x_test = self.preprocessed_data.get_vectorized_data_to_eval(x_test, self.preprocessed_data.is_tfidf_vect)
                print_classification_report_xgb(model, x_test, y_true)
                
                display_("##### Ten-fold cross-validation accuracy score : ")
                ac_score = np.zeros(10)
                for i in list(range(10)):
                    idx = np.random.choice(x.shape[0], 1000, replace=False)
                    x_test = x[idx]
                    y_true = y[idx]
                    if self.preprocessed_data.is_vectorize:
                        x_test = self.preprocessed_data.get_vectorized_data_to_eval(x_test, self.preprocessed_data.is_tfidf_vect)
                    y_true, y_pred = set_predicted_values(model, x_test, y_true)
                    ac_score[i] = accuracy_score(y_true, y_pred)
                display_("##### : " + str(ac_score.mean()*100))
            
            def runXgboost(num_boost_round = num_boost_round, lr = lr, max_delta_step = max_delta_step):
                dtrain = xgb_preprocess(self.X_train, self.y_train, True)
                dtest = xgb_preprocess(self.X_test, self.y_test, True)

                params = {'objective': 'multi:softprob', 
                          'eval_metric': 'mlogloss',
                          'num_class': 3, 
                          'max_delta_step': max_delta_step, 
                          'eta': lr}

                evals = [(dtrain, 'train'), (dtest, 'eval')]

                xgb = xgboost.train(params=params,  
                                dtrain=dtrain, 
                                num_boost_round=num_boost_round, 
                                evals=evals,
                                verbose_eval=False,
                                early_stopping_rounds=10)
                return xgb
            
            self.xgb_model = runXgboost()
            display_("##### XGBoost classification report : ")
            print_classification_report_xgb(self.xgb_model, self.X_test, self.y_test)
            display_("##### XGBoost <span style='color: red'>randomly selected 1000 data</span> classification report : ")
            print_classification_report_xgb_1000(self.xgb_model)