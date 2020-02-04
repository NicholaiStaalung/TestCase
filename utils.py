import pandas as pd
import numpy as np

import sklearn as skl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
#import bigquery as bq
from scipy.stats import wilcoxon, jarque_bera
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mlxtend.evaluate import mcnemar_table, mcnemar
import math
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.example import visualize
# from pyod.utils.data import generate_data, get_outliers_inliers
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
# import tensorflow as tf
from scipy.stats import skew, skewtest, linregress, kurtosistest

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.compat import lzip

from xgboost import XGBRegressor
from scipy import stats
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 123


def importDataBQ(sql_string=None):
    """Import data via bigquery

       Parameters
       ----------

       sql_string : A string with BQ SQL syntax - Required


       Returns
       -------

       df : Pandas DataFrame with columns and rows specified
            in the sql string

    """

    return bq.sql(sql_string).as_dataframe()

def importDataCSV(path, names=None):
    """Import data from CSV file.
       First row of the tabular csv file has to be the row names.
       Otherwise pass a list of column names to the 'names' argument and 'header=None' argument

       Parameters
       ----------

       path : Absolute path to CSV file - Required

       names : A list of names in the format ['name1', 'name2', ... 'nameN'] - Optional


       Returns
       -------

       df : Pandas DataFrame with columns and rows from the CSV file

    """
    if names != None:
        return pd.read_csv(filepath_or_buffer=path, names=names, header=None)
    else:
        return pd.read_csv(filepath_or_buffer=path)

def dropColumns(df, cols):
    """Dropping specified columns from the dataframe

       Parameters
       ----------

       df : A pandas dataframe with rows and columns

       cols: a column name as a string or multiple column names as a list of strings

       Returns
       -------

       df : A pandas dataframe wihout the dropped columns


    """
    return df.drop(cols, axis=1, inplace=True)

def mapMissingValues(df):
    """Maps the missing values from the dataframe into relative numbers

       Parameters
       ----------

       df : A pandas dataframe with rows and columns

       Returns
       -------

       df: A pandas dataframe with missing values as relative numbers

    """

    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum())/df.isnull().count().sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
    return missing_data

def plotRelations(df, loud=False, plot_size=20, text_size=10):
    """Plot the relations through a correlation matrix.
       Add argument "loud=True" for scatterplots and density plots as well

       Parameters
       ----------

       df : A pandas DataFrame with rows and columns - Required

       loud : [True] if to add density and scatter plots. Defaults to [False] - Optional

       plot_size : The size of the plot onm both axis. Defaults to [20] - Optional

       text_size : The size of the text in the plots. Defaults to [10] - Optional

       Returns
       -------

       A plot of correlations or correlations, densities and scatters for the continous values of the dataframe
    """
    if loud:
        df = df.select_dtypes(include =[np.number]) # keep only numerical columns
        ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plot_size, plot_size], diagonal='kde')
        corrs = df.corr().values
        for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
            ax[i, j].annotate('%.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=text_size)
        plt.suptitle('Scatter and Density Plot')
        sns.set()
    else:
        df = df.select_dtypes(include =[np.number]) # keep only numerical columns
        corrmatrix = df.corr()
        f, ax = plt.subplots(figsize=[15, 15])
        sns.heatmap(corrmatrix, annot=True, square=True, annot_kws={"size": 8})
        sns.set(font_scale=0.8)

def plotDiscreteRelations(df, figsize=[15, 10]):
    """Plot discrete relations through histograms to asses the distribution of categorical data

       Parameters
       ----------

       df : A Pandas dataframe consistion of rows and columns with at least one discrete feature

       Returns
       -------

       Plots of discrete features as histograms


    """
    df = df.select_dtypes(include=['category'])
    warns = []

    for col in df.columns.tolist():
        if len(df[col].value_counts()) > 20:
            warns.append('\n-------------------------------\nExcluded column -{}- as it has more than 20 categories. -{}- has {} categories\n-------------------------------\n'.format(col, col, len(df[col].value_counts())))
            dropColumns(df, col)

    f, ax = plt.subplots(math.ceil(df.shape[1]/2), 2 if df.shape[1] >= 2 else df.shape[1], figsize=figsize)
    p = 0
    for j in range(0, ax.shape[0]):
        for i in range(0, 2):
            try:
                col = df.columns.tolist()[p]
                sns.countplot(df[col], ax=ax[j, i])
                p += 1
            except Exception as e:
                print('error: {}'.format(e))


    for i in warns:
        print(i)

    sns.set()

def logTransformPlot(df):
    """Check for log transformations og continuous features.
    Will invert features consiting of only negative values to positive. TODO fix for values ranging from -inf to inf

       Parameters
       ----------

       df : A dataframe with rows and columns with at least on numerical feature


       Returns
       -------

       Plots containing the level, log and qq plots to asses normal distribution of the data



    """
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    fig, ax = plt.subplots(math.ceil(df.shape[1]), 3 if df.shape[1] >= 3 else df.shape[1], figsize=[15, 45])
    p = 0
    for j in range(0, df.shape[1]):
        col = df.columns[p]

        try:
            sns.distplot(np.log1p(df[col]), ax=ax[j, 1])
            sns.distplot(df[col], ax=ax[j, 0])
            sm.qqplot(np.log1p(df[col]), stats.norm, fit=True, line='45', ax=ax[j, 2]);

        except:
            sns.distplot(np.log1p(-df[col]), ax=ax[j, 1])
            sns.distplot(-df[col], ax=ax[j, 0])
            sm.qqplot(np.log1p(-df[col]), stats.norm, fit=True, line='45', ax=ax[j, 2]);


        ax[j, 1].set_xlabel('log(1+{})'.format(col))
        p += 1


def plotBoxPlots(df):
    """Plot continuous features as box plots to assess distribution and outliers

       Parameters
       ----------

       df : A dataframe with rows and columns with at least on numerical feature

       Returns
       -------

       Boxplots of all numerical features


    """
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    number_cols = 6
    f, ax = plt.subplots(math.ceil(df.shape[1]/number_cols), number_cols if df.shape[1] >= number_cols else df.shape[1], figsize=[15, 12])
    p = 0
    for j in range(0, ax.shape[0]):
        for i in range(0, number_cols):
            try:
                col = df.columns.tolist()[p]
                sns.boxplot(df[col], ax=ax[j, i])
                p += 1
            except Exception as e:
                # print('error: {}'.format(e))
                pass

    sns.set()

def detectOutliers(df, features, classifier='K Nearest Neighbors (KNN)', outliers_fraction=0.01, df_test=None):
    """Classify datapoints as outliers and return the DataFrame

       Parameters
       ----------

       df : A dataframe with rows and columns

       features : A list of columns names to ba analysed. Twor or more

       outlier_fraction : A floating point value determining the fraction of observations to be determined as outliers

       classifier : The outlier classification method

       Returns
       -------

       outliers : A binary numpy vector of outliers
    """

    classifiers = {
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    }

    try:
        clf = classifiers[classifier]
    except Exception as e:
        print(e)

    # scaler = MinMaxScaler(feature_range=(0, 1)) # Scaling to provide meaningfull visualizations
    # df[features] = scaler.fit_transform(df[features])
    # df_test[features] = scaler.transform(df_test[features])

    clf.fit(df[features])
    # predict raw anomaly
    scores_pred = (clf.decision_function(df[features]) * -1).T
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(df[features])
    if isinstance(df_test, pd.core.frame.DataFrame):
        y_pred_test = clf.predict(df_test[features])
    elif isinstance(df_test, type(None)):
        y_pred_test = None
    # print(y_pred_test.mean())

    return y_pred, y_pred_test, scores_pred






def detectOutliersCompare(df, features, use_categories=True, use_continuous=True, outliers_fraction=0.05):
    """Detect outliers from a number of features

       Parameters
       ----------

       df : A pandas dataframe with rows and columns

       features : A list of columns names to ba analysed. Twor or more

       outlier_fraction : A floating point value determining the fraction of observations to be determined as outliers

       Returns
       -------


    """
    # Define outlier detection tools to be compared
    classifiers = {
    'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=RANDOM_STATE),
    'Isolation Forest': IForest(contamination=outliers_fraction,random_state=RANDOM_STATE),
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',contamination=outliers_fraction)
    }

    if use_continuous:
        features = df.select_dtypes(include =[np.number]).columns.tolist()
    if use_categories:
        if len(df.select_dtypes(include = ['category']).columns.tolist()) > 0:
            print('\nReconstructed the following columns to dummies as they were not in the correct data format\n{}'.format(df.select_dtypes(include = ['category']).columns.tolist()))
            df = pd.get_dummies(df, columns=df.select_dtypes(include = ['category']).columns.tolist(), drop_first=True)
        try:
            features.extend(df.columns.tolist())
        except:
            features = df.columns.tolist() #If we didnt use continuos values


    # copy of dataframe

    if len(features) > 2:
        print('\nReducing dimensions because we have more than two features\nStats:')
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        x_pca = pca.fit_transform(df[features])
        explained_variance = pca.explained_variance_ratio_
        print('Reduced feature 1: {} %\nReduced feature 2: {} %'.format(round(explained_variance[0]*100, 2), round(explained_variance[1]*100, 2)))
        print('Sum of explained variance: {} %'.format(100 * round(np.sum(explained_variance), 2)))
        features = ['red_dim1', 'red_dim2']
        dfx = pd.DataFrame()
        dfx[features[0]] = x_pca[:, 0]
        dfx[features[1]] = x_pca[:, 1]
    else:
        dfx = df[features]


    scaler = MinMaxScaler(feature_range=(0, 1)) # Scaling to provide meaningfull visualizations
    dfx[features] = scaler.fit_transform(dfx[features])

    xx , yy = np.meshgrid(np.linspace(0,1 , 500), np.linspace(0, 1, 500))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(dfx[features])
        # predict raw anomaly
        scores_pred = (clf.decision_function(dfx[features]) * -1).T
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(dfx[features])
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)

        # copy of dataframe
        dfx = dfx[[features[0], features[1]]]
        dfx['outlier'] = y_pred.tolist()


        if n_outliers > 0:
            plt.figure(figsize=(5, 5))
            # IX1 - inlier feature 1,  IX2 - inlier feature 2
            IX1 =  np.array(dfx[features[0]][dfx['outlier'] == 0]).reshape(-1,1)
            IX2 =  np.array(dfx[features[1]][dfx['outlier'] == 0]).reshape(-1,1)
            # OX1 - outlier feature 1, OX2 - outlier feature 2
            OX1 =  dfx[features[0]][dfx['outlier'] == 1].values.reshape(-1,1)
            OX2 =  dfx[features[1]][dfx['outlier'] == 1].values.reshape(-1,1)

            # threshold value to consider a datapoint inlier or outlier
            threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
            # decision function calculates the raw anomaly score for every point
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
            Z = Z.reshape(xx.shape)
            # fill blue map colormap from minimum anomaly score to threshold value
            # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
            # draw red contour line where anomaly score is equal to thresold
            a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
#           # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
            plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
            b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
            c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
            plt.axis('tight')
            # loc=2 is used for the top left corner
            plt.legend(
                [a.collections[0], b,c],
                ['learned decision function', 'inliers = {} % ({})'.format(round(100.00 * n_inliers / (n_outliers + n_inliers), 2), n_inliers),'outliers = {} % ({})'.format(round(100.00 * n_outliers / (n_outliers + n_inliers), 2), n_outliers)],
                prop=matplotlib.font_manager.FontProperties(size=10),
                loc=2)
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.xlabel('{}_scaled'.format(features[0]))
            plt.ylabel('{}_scaled'.format(features[1]))
            plt.title(clf_name)
        else:
            print('\nNo outliers found from {}'.format(clf_name))
            print('---------------------------')




# squared_loss
def rmse_cv(model, n_folds, X_train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state = RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

def rmse_lv_cv(model, n_folds, X_train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state = RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
