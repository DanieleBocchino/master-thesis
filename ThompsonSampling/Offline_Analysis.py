from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
def get_report(X, y, scaler=False, fit_intercept=True):

        if(scaler):
            scaler = StandardScaler()
            scaler.fit(X)
            scaled_data = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Create a logistic regression model
        clf = LogisticRegression(fit_intercept=fit_intercept)
            
        # Train the model on the training data
        clf.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = clf.predict(X_test)
        coef = clf.coef_
        intercept = clf.intercept_
        accuracy = clf.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
            
        return y_pred, coef, intercept, accuracy, precision, recall, f1
          
def offline_analysis(offline_analysis_df, experiment_df, n_dim,  outputs_dir='outputs/train/'):
        
    X = np.concatenate([sublist for sublist in (offline_analysis_df['x'].tolist())])
    X = X.reshape(-1, n_dim)
    y = np.concatenate([sublist for sublist in (offline_analysis_df['y'].tolist())])

    #TS M 
    if n_dim==4:
        ts_m =pd.DataFrame({'m_1':experiment_df['m'].values[-1][0], 'm_id': experiment_df['m'].values[-1][1], 'm_dist': experiment_df['m'].values[-1][2], 'm_ang':experiment_df['m'].values[-1][3]}, index=['TS_m'])
    else:
        ts_m =pd.DataFrame({'m_id': experiment_df['m'].values[-1][0], 'm_dist': experiment_df['m'].values[-1][1], 'm_ang':experiment_df['m'].values[-1][2]}, index=['TS_m'])

    #No Scaler No Intercept
    y_pred, coef, intercept, accuracy, precision, recall, f1 = get_report(X, y, scaler=False, fit_intercept=False)

    if n_dim==4:
        ts_m =pd.concat([ts_m, pd.DataFrame({'m_1':coef[0][0], 'm_id': coef[0][1], 'm_dist': coef[0][2], 'm_ang':coef[0][3]}, index=['coef_No_Scal_No_Int'])])
    else:
        ts_m = pd.concat([ts_m, pd.DataFrame({'m_id': coef[0][0], 'm_dist': coef[0][1], 'm_ang':coef[0][1]}, index=['coef_No_Scal_No_Int'])])

    #No Scaler
    y_pred, coef, intercept, accuracy, precision, recall, f1 = get_report(X, y, scaler=False,  fit_intercept=True)
    if n_dim==4:
        ts_m =pd.concat([ts_m, pd.DataFrame({'m_1':coef[0][0], 'm_id': coef[0][1], 'm_dist': coef[0][2], 'm_ang':coef[0][3]}, index=['coef_No_Scal'])])
    else:
        ts_m = pd.concat([ts_m, pd.DataFrame({'m_id': coef[0][0], 'm_dist': coef[0][1], 'm_ang':coef[0][1]}, index=['coef_No_Scal'])])

    #Scaler
    y_pred, coef, intercept, accuracy, precision, recall, f1 = get_report(X, y, scaler=True,  fit_intercept=True)
    if n_dim==4:
        ts_m =pd.concat([ts_m, pd.DataFrame({'m_1':coef[0][0], 'm_id': coef[0][1], 'm_dist': coef[0][2], 'm_ang':coef[0][3]}, index=['coef'])])
    else:
        ts_m = pd.concat([ts_m, pd.DataFrame({'m_id': coef[0][0], 'm_dist': coef[0][1], 'm_ang':coef[0][1]}, index=['coef'])])

    #Report m
    ts_m.to_csv(f'{outputs_dir}csv/ts_m.csv')
    
    offline_out_df = pd.DataFrame({'X': [X], 'Y': [y], 'y_pred' : [y_pred]}) 
    offline_out_df.to_csv(f'{outputs_dir}csv/offline_out.csv', )  
            
    return ts_m