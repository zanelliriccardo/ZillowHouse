import numpy as np
import sklearn
from scipy import stats
from scipy.stats import norm, skew 
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import date
from scipy.special import boxcox1p
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import math

def check_duplicates(df,features): 
    idsUnique = len(df[features].value_counts())
    idsTotal = df.shape[0]
    idsDupli = idsTotal - idsUnique
    print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
    
def drop_duplicates(df,features):
    print("Dropping all duplicates based on: " + str(features))
    return df.drop_duplicates(subset=features, keep='last', ignore_index=True)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def predict_in_chunks(model, preprocessor, df_test): 
    final_pred = []

    for group in chunker(df_test, 300000):
        group = preprocessor.transform(group)
        pred = model.predict(group)
        final_pred.append(pred)
        del group; gc.collect()
        del pred; gc.collect()
    return np.concatenate(final_pred).ravel()

def nan_values(df,percentage=75):
    min_count =  int(((100-percentage)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh=min_count)
    print(df.shape)
    return df

def drop_outliers(df,feature,std_factor=2.5):
    y=df[feature]
    highest_thres = y.mean() + std_factor*y.std()
    lowest_thres = y.mean() - std_factor*y.std()
    print("Highest allowed",highest_thres)
    print("Lowest allowed", lowest_thres)

    y = y[y > lowest_thres]
    y = y[y < highest_thres]

    df = df[df.logerror > lowest_thres]
    df = df[df.logerror < highest_thres]
    return y,df

def plot_variable(y):
    y.hist(bins=100, figsize=(8,5))
    sns.distplot(y , fit=norm);

    (mu, sigma) = norm.fit(y)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('LogError distribution')

    fig = plt.figure()
    res = stats.probplot(y, plot=plt)
    plt.show()
    y.describe()
    
def get_eval_metrics(models, X, y_true): 
    """
    Calculates MAE (Mean Absoulate Error) and RMSE (Root Mean Squared Error) on the data set for input models. 
    `models`: list of fit models 
    """
    for model in models: 
        y_pred= model.predict(X)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Model: {model}")
        print(f"MAE: {mae}, RMSE: {rmse}")
        return mae

def display_scores(model, scores):
    print("-"*50)
    print("Model:", model)
    print("\nScores:", scores)
    print("\nMean:", scores.mean())
    print("\nStandard deviation:", scores.std())
    
def get_cross_val_scores(models, X, y, cv=10, fit_params=None):
    """
    Performs k-fold cross validation and calculates MAE for each fold for all input models. 
    `models`: list of fit models 
    """    
    maes = []
    for model in models: 
        mae = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv, fit_params=fit_params)
        display_scores(model, mae) 
        maes.append(mae)
    return maes

class CreateDateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates simple date features by extracting the information from `transactiondate` 
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self   
    
    def transform(self, X): 
        dt = pd.to_datetime(X['transactiondate']).dt
        X['transaction_year'] = (dt.year).astype('category')
        X['transaction_month'] = ((dt.year - 2016)*12 + dt.month).astype('category')
        X['transaction_day'] = dt.day
        X['transaction_quarter'] = ((dt.year - 2016)*4 + dt.quarter).astype('category')
        X = X.drop(['transactiondate'], axis=1)
    
        return X

class CreateDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Creates new features by combining existing variables 
    """
    def __init__(self):
        return None # nothing else to do 
    
    def fit(self, X, y=None):
        return self  # nothing else to do 
    
    def transform(self, X): 
        # Average Size Features 
        X['N-avg_garage_size'] = X['garagetotalsqft'] / X['garagecarcnt']
        X['N-property_tax_per_sqft'] = X['taxamount'] / X['calculatedfinishedsquarefeet']
        
        # Average area in sqft per room
        mask = (X.roomcnt >= 1)  # avoid dividing by zero
        X.loc[mask, 'N-avg_area_per_room'] = X.loc[mask, 'calculatedfinishedsquarefeet'] / X.loc[mask, 'roomcnt']
        
        # Derived Room Count
        X['Nderived_room_cnt'] = X['bedroomcnt'] + X['bathroomcnt']
        
        # Use the derived room_cnt to calculate the avg area again
        mask = (X.Nderived_room_cnt >= 1)
        X.loc[mask,'N-derived_avg_area_per_room'] = X.loc[mask,'calculatedfinishedsquarefeet'] / X.loc[mask,'Nderived_room_cnt']
        
        # Rotated Coordinates
        X['N-location_1'] = X['latitude'] + X['longitude']
        X['N-location_2'] = X['latitude'] - X['longitude']
        X['N-location_3'] = X['latitude'] + 0.5 * X['longitude']
        X['N-location_4'] = X['latitude'] - 0.5 * X['longitude']
        
        X['x'] = X['latitude'].map(lambda x : math.cos(x)) * X['longitude'].map(lambda x : math.cos(x)) 
        X['y'] = X['latitude'].map(lambda x : math.cos(x)) * X['longitude'].map(lambda x : math.sin(x)) 
        X['z'] = X['latitude'].map(lambda x : math.sin(x))
        
        #error in calculation of living area
        mask = (X.finishedsquarefeet12 >= 1)  # avoid dividing by zero
        X.loc[mask,'N-living_area_error'] = X.loc[mask,'calculatedfinishedsquarefeet']/X.loc[mask,'finishedsquarefeet12']
        
        #proportion of living area
        mask = (X.lotsizesquarefeet >= 1)  # avoid dividing by zero
        X.loc[mask,'N-living_area_prop'] = X.loc[mask,'calculatedfinishedsquarefeet']/X.loc[mask,'lotsizesquarefeet']
        mask = (X.finishedsquarefeet15 >= 1)  # avoid dividing by zero
        X.loc[mask,'N-living_area_prop2'] = X.loc[mask,'finishedsquarefeet12']/X.loc[mask,'finishedsquarefeet15']
        
        X['N-extra_space'] = X['lotsizesquarefeet'] - X['calculatedfinishedsquarefeet']                                                            
        X['N-extra_space2'] = X['finishedsquarefeet15'] - X['finishedsquarefeet12']  
        
        mask = (X.landtaxvaluedollarcnt >= 1)  # avoid dividing by zero
        X.loc[mask,'N-value_prop'] = X.loc[mask,'structuretaxvaluedollarcnt']/X.loc[mask,'landtaxvaluedollarcnt']
        X['N-n_gar_pool_ac']= ((X['garagecarcnt']>0) & (X['pooltypeid10']>0) & (X['airconditioningtypeid']!=5))
        
        
        #Ratio of tax of property over parcel
        X['N-ValueRatio'] = X['taxvaluedollarcnt']/X['taxamount']

        #TotalTaxScore
        X['N-TaxScore'] = X['taxvaluedollarcnt']*X['taxamount']

        #polnomials of tax delinquency year
        X["N-taxdelinquencyyear-2"] = X["taxdelinquencyyear"] ** 2
        X["N-taxdelinquencyyear-3"] = X["taxdelinquencyyear"] ** 3

        #Length of time since unpaid taxes
        X['N-life'] = 2018 - X['taxdelinquencyyear']
        
        #Number of properties in the zip
        zip_count = X['regionidzip'].value_counts().to_dict()
        X['N-zip_count'] = X['regionidzip'].map(zip_count)

        #Number of properties in the city
        city_count = X['regionidcity'].value_counts().to_dict()
        X['N-city_count'] = X['regionidcity'].map(city_count)

        #Number of properties in the city
        region_count = X['regionidcounty'].value_counts().to_dict()
        X['N-county_count'] = X['regionidcounty'].map(city_count)
        
        #Indicator whether it has AC or not
        X['N-ACInd'] = (X['airconditioningtypeid']!=5)*1

        #Indicator whether it has Heating or not 
        X['N-HeatInd'] = (X['heatingorsystemtypeid']!=13)*1

        #There's 25 different property uses - let's compress them down to 4 categories
        X['N-PropType'] = X.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 
                                                             248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 
                                                             264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 
                                                             269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 
                                                             275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })
        
        #polnomials of the variable
        X["N-structuretaxvaluedollarcnt-2"] = X["structuretaxvaluedollarcnt"] ** 2
        X["N-structuretaxvaluedollarcnt-3"] = X["structuretaxvaluedollarcnt"] ** 3

        #Average structuretaxvaluedollarcnt by city
        group = X.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
        X['N-Avg-structuretaxvaluedollarcnt'] = X['regionidcity'].map(group)

        #Deviation away from average
        X['N-Dev-structuretaxvaluedollarcnt'] = abs((X['structuretaxvaluedollarcnt'] - X['N-Avg-structuretaxvaluedollarcnt']))/X['N-Avg-structuretaxvaluedollarcnt']

        # 'finished_area_sqft' and 'total_area' cover only a strict subset of 'finished_area_sqft_calc' in terms of 
        # non-missing values. Also, when both fields are not null, the values are always the same.
        # So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
        # If there're some patterns in when the values are missing, we can add two isMissing binary features
        X['N-missing_finished_area'] = X['finishedsquarefeet12'].isnull().astype(float)
        X['N-missing_total_area'] = X['finishedsquarefeet15'].isnull().astype(float)
        X = X.drop(['finishedsquarefeet12', 'finishedsquarefeet15'], axis=1)
        X['N-missing_bathroom_cnt_calc'] = X['calculatedbathnbr'].isnull().astype(float)
        X = X.drop(['calculatedbathnbr'], axis=1)
        
        return X
    
class ConvertToCategorical(BaseEstimator, TransformerMixin): 
    def __init__(self, cat_vars):
        self.cat_vars = cat_vars
        
    def fit(self, X, y=None): 
        return self  
    
    def transform(self, X): 
        for col in self.cat_vars: 
            X[col] = pd.Categorical(X[col])
        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    def fit(self, X, y=None):
        return self  
    def transform(self, X): 
        updated_X = X.drop(self.features_to_drop, axis=1)
        return updated_X
    
class SplitCensus(BaseEstimator, TransformerMixin):
    """
    Creates simple date features by extracting the information from `rawcensustractandblock` and `censustractandblock`
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self   
    
    def transform(self, X): 
        tract=[]
        block=[]
        for code in X.rawcensustractandblock.to_list():
            code = str(code)
            tract.append(code[4:11])
            block.append(code[11:])

        X = X.assign(tract_code=tract,
                       block=block)
        X.drop(['rawcensustractandblock'],axis=1,inplace=True)
        
        tract=[]
        block=[]
        for code in X.censustractandblock.to_list():
            code = str(code)
            tract.append(code[4:11])
            block.append(code[11:])

        X = X.assign(tract_code=tract,
                       block=block)
        X.drop(['censustractandblock'],axis=1,inplace=True)
        return X
    
class ColumnNamesAppender(BaseEstimator, TransformerMixin):
    def __init__(self, column_transformer, orig_columns, num_transformers):
        self.column_transformer = column_transformer
        self.orig_columns = orig_columns
        self.num_transformers = num_transformers
    def fit(self, X, y=None):
        return self  
    def transform(self, X): 
        X_column_names = self.get_columns_from_transformer(self.column_transformer, self.orig_columns, self.num_transformers)
        
        # Create dataframe from numpy array and column names 
        X = pd.DataFrame(X, columns=X_column_names)
        return X 
    
    @staticmethod
    def get_columns_from_transformer(column_transformer, input_colums, num_transformers):    
        col_name = []
        
        for transformer_in_columns in column_transformer.transformers_: #the last transformer is ColumnTransformer's 'remainder'
            raw_col_name = transformer_in_columns[2]
            if isinstance(transformer_in_columns[1],Pipeline): 
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]
            try:
                names = transformer.get_feature_names([raw_col_name])
            except AttributeError: # if no 'get_feature_names' function, use raw column name
                names = raw_col_name
            if isinstance(names,np.ndarray): 
                col_name += names.tolist()
            elif isinstance(names,list):
                col_name += names    
            elif isinstance(names,str):
                col_name.append(names)

        return col_name

class ConvertFeatureType(BaseEstimator, TransformerMixin): 
    def __init__(self, convert_to_int=[], convert_to_bool=[], convert_to_string=[], convert_to_float=[]):
        self.convert_to_int = convert_to_int
        self.convert_to_bool = convert_to_bool
        self.convert_to_string = convert_to_string
        self.convert_to_float = convert_to_float
        self.features = {"int": convert_to_int, "float": convert_to_float, "boolean": convert_to_bool, "str": convert_to_string}
    def fit(self, X, y=None): 
        return self  # Nothing else to do 
    
    def transform(self, X): 
        self.map_bool_features(X)        
        for data_type in self.features.keys(): 
            X = self.convert_feature_types(X, data_type)
        return X 
    
    def map_bool_features(self, X): 
        """Convert all non null values to True in bool features prior to changing type to Boolean."""
        for var in self.convert_to_bool:
            X[var][X[var].notnull()] = True
    
    def convert_feature_types(self, X, data_type): 
        for var in self.features[data_type]: 
            X[var] = X[var].astype(data_type) 
        return X

class FeatureEncoderAndScaler(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None, features_to_scale=None, numeric_types=["float"]):
        self.features_to_encode = features_to_encode
        self.features_to_scale = features_to_scale
        self.numeric_types = numeric_types
        self.feature_encoder_and_scaler = None
        
    def fit(self, X, y=None):
        if not self.features_to_encode:
            self.features_to_encode = X.select_dtypes(include = ["object"]).columns 
        if not self.features_to_scale:
            self.features_to_scale = X.select_dtypes(include = self.numeric_types).columns   
            feature_encoder_scaler = ColumnTransformer([
            ("ohe_cats", OneHotEncoder(handle_unknown='ignore', sparse=False), self.features_to_encode),
            ("num_scaler", RobustScaler(), self.features_to_scale),
        ],
            remainder='passthrough',
            )
                    
        self.feature_encoder_scaler = feature_encoder_scaler.fit(X)
        return self   
    
    def transform(self, X): 
        # OneHotEncoder returns numpy array which is converted to dataframe
        X_np = self.feature_encoder_scaler.transform(X)
        X = pd.DataFrame(
            X_np, 
            columns=self.feature_encoder_scaler.get_feature_names_out()
        )
        X = self.convert_feature_types(X)
        
        return X
    
    def convert_feature_types(self, X):
        """Convert feature types to object, float, bool based on the column name. 
        Columns with `ohe_cats` are object, `num_scaler` are float, `remainder` are bool"""
        for column in X:
            if 'ohe_cats' in column:
                X[column] = X[column].astype("object") 
            elif 'num_scaler' in column: 
                X[column] = X[column].astype("float") 
            elif 'remainder' in column: 
                X[column] = X[column].astype("boolean") 
        return X 
         
class ConvertToType(BaseEstimator, TransformerMixin): 
    def __init__(self, var_type, vars_to_convert=None):
        self.var_type = var_type
        self.vars_to_convert = vars_to_convert
        
    def fit(self, X, y=None): 
        return self 
    
    def transform(self, X): 
        if self.vars_to_convert: 
            for col in self.vars_to_convert: 
                X[col] = X[col].astype(self.var_type) 
        else: 
            for col in X.columns: 
                X[col] = X[col].astype(self.var_type)     
        return X

class CreateYearFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_features):
        self.date_features = date_features
        self.current_year = date.today().year
    
    def fit(self, X, y=None):
        return self  # nothing else to do 
    
    def transform(self, X): 
        for var in self.date_features.keys(): 
            new_var_name = self.date_features[var]
            X[new_var_name] = self.current_year - X[var]
            X[new_var_name] = X[new_var_name].astype('float') 
            
            # Drop old feature
            X.drop(var, axis=1, inplace=True)
        return X

class CreatePolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,most_corr_feat):
        self.most_corr_feat = most_corr_feat
    def fit(self, X, y=None):
        return self  # nothing else to do 
    
    def transform(self, X): 
        for var in self.most_corr_feat: 
            # New var names 
            s2_var_name = var + '-s2'
            s3_var_name = var + '-s3'
            sq_var_name = var + '-sqrt'
            
            # Create features 
            X[s2_var_name] = X[var] ** 2 
            X[s3_var_name] = X[var] ** 3 
            X[sq_var_name] = np.sqrt(X[var] + abs(min(X[var])))  # Translate feature to ensure min value is 0 before sqrt 
            
        return X

class BoxCoxSkewedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_thres=0.75):
        self.skewness_thres = skewness_thres
    def fit(self, X, y=None):
        return self  # nothing else to do 
    def transform(self, X): 
        numeric_feats = X.dtypes[X.dtypes == 'float'].index
        skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        
        skewed_feats = skewed_feats[abs(skewed_feats) > self.skewness_thres].index
        
        # Apply box-cox to each variable 
        lam = 0.18
        for feat in skewed_feats:
            X[feat] = X[feat] + abs(min(X[feat]))       # Translate feature to ensure minimum value is 0 
            X[feat] = boxcox1p(X[feat], lam)
        return X
    

def reduce_mem_usage(props):#TODO how we will manage null values?
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


