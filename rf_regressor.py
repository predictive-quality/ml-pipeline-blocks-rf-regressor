# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from sklearn.ensemble import RandomForestRegressor
from s3_smart_open import to_pckl, read_pckl, read_pd_fth, to_s3
from absl import logging
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow import keras
from sklearn.tree import export_graphviz
import os
import pydot

class Regressor(BaseEstimator, RegressorMixin):
    """Class built from sklearn Random Forest Regressor
    """
    def __init__(self,model_name, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples):
        """Initialize the Random Forest Regressor object

        Args:
            model_name (str): Name of the model to save/load.
            n_estimators (int): The number of trees in the forest.
            criterion (str): The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.
            max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            min_samples_split (int or float): The minimum number of samples required to split an internal node
            min_samples_leaf (int or float): The minimum number of samples required to be at a leaf node.
            min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            max_features (str,float or int): The number of features to consider when looking for the best split.
            max_leaf_nodes (int): Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            min_impurity_split (float): Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
            bootstrap (bool): Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            oob_score (bool): whether to use out-of-bag samples to estimate the R^2 on unseen data.
            n_jobs (int): [The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.
            random_state (int): Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
            verbose (int): Controls the verbosity when fitting and predicting.
            warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
            ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
            max_samples (int or float): If bootstrap is True, the number of samples to draw from X to train each base estimator.
        """    

        self.model_name = model_name
        assert n_estimators > 0
        self.n_estimators = n_estimators
        self.criterion = criterion
        if max_depth:   # Due to argo template  
            self.max_depth = int(max_depth)
        else:
            self.max_depth = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        if max_leaf_nodes:  # Due to Argo template
            self.max_leaf_nodes = int(max_leaf_nodes)
        else:
            self.max_leaf_nodes = None
        self.min_impurity_decrease = min_impurity_decrease
        if min_impurity_split:  # Due to Argo template
            self.min_impurity_split = float(min_impurity_split)
        else:
            self.min_impurity_split = None
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        assert n_jobs > 0 or n_jobs == -1
        self.n_jobs = n_jobs
        if random_state: # Due to Argo template 
            self.random_state = int(random_state)
        else:
            self.random_state = None
        self.verbose = verbose
        self.warm_start = warm_start
        assert ccp_alpha >= 0.0
        self.ccp_alpha = ccp_alpha
        if max_samples: # Due to Argo template 
            self.max_samples = max_samples
        else:
            self.max_samples = None


    def build_model(self,input_path):
        """Builds model from sklearn.ensemble.RandomForrestRegressor. 
        Args:
            input_path (str): Path where the already fitted model is stored.
        """ 
        '''
        The try - except structure is needed to handle multiple input types for one parameter.
        A "float-string" like x='5.0' will cause an error for int(x) and so float(x) will be used.
        A "interger-string" like x='5' will cause no error for int(x) and float(x) wont be used.
        Mostly when intergers are used they are considered as a fixed falue for the parameter
        While float values are considered as percentages and will be multiplied with some other values/parameters
        '''
        try:
            self.min_samples_split = int(self.min_samples_split)
            assert self.min_samples_split > 1
        except:
            self.min_samples_split = float(self.min_samples_split)
            assert 0.0 < self.min_samples_split <= 1.0

        try:
            self.min_samples_leaf = int(self.min_samples_leaf)
            assert self.min_samples_split > 0
        except:
            self.min_samples_leaf = float(self.min_samples_leaf)
            assert 0 < self.min_samples_split <= 0.5

        if self.max_features and self.max_features not in ['auto','sqrt','log2']:
            try:
                self.max_features = int(self.max_features)
            except:
                self.max_features = float(self.max_features)
        if self.max_samples:
            try:
                self.max_samples = int(self.max_samples)
                assert self.max_samples > 0
            except:
                self.max_samples = float(self.max_samples)
                assert 0.0 < self.max_samples < 1.0

        if self.warm_start == False:
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                                criterion=self.criterion, 
                                                max_depth=self.max_depth, 
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_features=self.max_features,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_impurity_decrease=self.min_impurity_decrease,
                                                min_impurity_split=self.min_impurity_split,
                                                bootstrap=self.bootstrap,
                                                oob_score=self.oob_score,
                                                n_jobs=self.n_jobs,
                                                random_state=self.random_state,
                                                verbose=self.verbose,
                                                warm_start=self.warm_start,
                                                ccp_alpha=self.ccp_alpha,
                                                max_samples=self.max_samples)
        else:
            self.model = self.load(input_path,self.model_name).model
            self.n_estimators = self.n_estimators + len(self.model.estimators_)
            logging.info('Expand estimators from {} to {}'.format(len(self.model.estimators_),self.n_estimators))
            old_parameter = self.model.get_params()
            self.model.set_params(n_estimators=self.n_estimators, 
                                    criterion=self.criterion, 
                                    max_depth=self.max_depth, 
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                    max_features=self.max_features,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    min_impurity_decrease=self.min_impurity_decrease,
                                    min_impurity_split=self.min_impurity_split,
                                    bootstrap=self.bootstrap,
                                    oob_score=self.oob_score,
                                    n_jobs=self.n_jobs,
                                    random_state=self.random_state,
                                    verbose=self.verbose,
                                    warm_start=self.warm_start,
                                    ccp_alpha=self.ccp_alpha,
                                    max_samples=self.max_samples)
            new_parameter = self.model.get_params()
            logging.info('Changed parameter for new estimators:')
            for key in old_parameter.keys():
                if old_parameter[key] != new_parameter[key]:
                    logging.info('- {} from {} to {}'.format(key,old_parameter[key],new_parameter[key]))


    def fit(self,X,y,input_path):
        """Build and fit the model to a given dataset X (features) and y (targets).
        Args:
            X (pd.DataFrame): features data
            y (pd.DataFrame): target data
        """ 
        X = X.values
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        self.build_model(input_path)
        logging.info('Begin fitting with '+str(X.shape)+' samples')
        self.model.fit(X,y)
        self.fitted_ = True


    def predict(self,X):
        """Predict the targets given a set of inputs with already fitted model.
        Args:
            X (pd.DataFrame): Features
        Returns:
            [pd.DataFrame]: Predicted targets
        """     
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'fitted_')
        
        return self.model.predict(X)


    def evaluate(self,X,y,metrics):
        """Evaluate a fitted model and return given keras metrics.
        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Targets
            metrics (list[str]): keras.metrics
        Returns:
            [dict]: Metrics in format {key,value}
        """ 
        X = X.values
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        check_is_fitted(self, 'fitted_')
        y_pred = self.model.predict(X)
        metrics = [keras.metrics.get(m) for m in metrics]
        res = {}
        for m in metrics:
            m.update_state(y, y_pred)
            res[m.name] = m.result().numpy().tolist()        
        return res


    def print_tree(self,output_path,X_col):
        if os.path.exists(output_path):
            dot_path = os.path.join(output_path,'tree.dot')
            svg_path = os.path.join(output_path,self.model_name+'_tree.svg')
        else:
            dot_path = os.path.join(os.getcwd(),'tree.dot')
            svg_path = os.path.join(os.getcwd(),self.model_name+'_tree.svg')

        path_list = [dot_path]
        export_graphviz(self.model.estimators_[0], out_file=dot_path ,feature_names=X_col, filled=True, rounded=True, special_characters=True)
        (graph,) = pydot.graph_from_dot_file(dot_path)
        graph.write_svg(svg_path)
        if output_path[:5] == 's3://':
            to_s3(output_path,self.model_name+'_tree.svg',svg_path)
            path_list.append(svg_path)

        for p in path_list:
            if os.path.exists(p):
                os.remove(p)
            else:
                logging.warning('{} does not exists!'.format(p))

    def save(self,output_path):
        """Saves the Random Forest Regressor object and the first Estimator/Tree to disk or to s3 bucket.
        Args:
            output_path (str): Path to storage location.
            X_col (list[str]): list of x/feature columnnames
        """        
        to_pckl(output_path,self.model_name+'.pckl',self)

    @staticmethod
    def load(input_path:str, model_name:str):
        """Loads the Random Forest object.
        Args:
            input_path (str): Path where the object is stored
            model_name (str): Name of the object to load
        Returns:
            [sv_regression.Regression object]: Random Forest Regressor to use for predict and evaluate.
        """    
        model = read_pckl(input_path, model_name+'.pckl')
        return model