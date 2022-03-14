# A random forest regressor
A [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

Read more in the [User Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest).

Note: sklearn.ensemble.RandomForestRegressor.fit() is implemented without fit parameter `sample_weight` at the moment.  
## Installation

Clone the repository and install all requirements using `pip install -r requirements.txt` .


## Usage



You can run the code in two ways.
1. Use command line flags as arguments `python main.py --input_path= --output_path=...`
2. Use a flagfile.txt which includes the arguments `python main.py --flagfile=example/flagfile.txt`

## Input Flags/Arguments

#### --model_name
Name of the model to save/load.

#### --input_path
Specify the a local or s3 object storage path where the input files are stored.
For a s3 object storage path a valid s3 configuration is required.

#### --output_path
Specify the path where the output files will be stored.
For a s3 object storage path a valid s3 configuration is required.

#### --filename_x
Filename of Dataframe with feautres.

#### --filename_y
Filename of Dataframe with the target.

#### --y_col_name
Name of the y column to train or to predict.
#### --stage
Wether to fit, predict or evaluate the model.

#### --metrics
Metrics for evaluation. See [here](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics) for candidate functions.
#### --n_estimators
The number of trees in the forest.
#### --criterion
The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.
#### --max_depth
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#### --min_samples_split
The minimum number of samples required to split an internal node:
 - If int, then consider min_samples_split as the minimum number.
 - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
#### --min_samples_leaf
The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
 - If int, then consider min_samples_leaf as the minimum number.
 - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
#### --min_weight_fraction_leaf
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
#### --max_features
The number of features to consider when looking for the best split:
 - If int, then consider max_features features at each split.
 - If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
 - If “auto”, then max_features=n_features.
 - If “sqrt”, then max_features=sqrt(n_features).
 - If “log2”, then max_features=log2(n_features).

Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
  #### --max_leaf_nodes
Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
#### --min_impurity_decrease
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

The weighted impurity decrease equation is the following:
`N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)`
where `N` is the total number of samples, `N_t` is the number of samples at the current node, `N_t_L` is the number of samples in the left child, and `N_t_R` is the number of samples in the right child.

`N`, `N_t`, `N_t_R` and `N_t_L` all refer to the weighted sum, if `sample_weight` is passed.

#### --min_impurity_split
Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
```Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19. The default value of min_impurity_split has changed from 1e-7 to 0 in 0.23 and it will be removed in 1.0 (renaming of 0.25). Use min_impurity_decrease instead.```
#### --bootstrap
Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
#### --oob_score
whether to use out-of-bag samples to estimate the R^2 on unseen data.
#### --n_jobs
The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-n_jobs) for more details.
#### --random_state
Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state) for details.

RandomState instance is not implemented at the moment!
#### --verbose
Controls the verbosity when fitting and predicting.
#### --warm_start
When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the [Glossary](https://scikit-learn.org/stable/glossary.html#term-warm_start).
#### --ccp_alpha
Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See [(Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) for details.
#### --max_samples
If bootstrap is True, the number of samples to draw from X to train each base estimator.
- If None (default), then draw X.shape[0] samples.
- If int, then draw max_samples samples.
- If float, then draw `max_samples * X.shape[0]` samples. Thus, max_samples should be in the interval (0, 1).


## Example

First move to the repository directory. \
Now you can fit the random forest with `python main.py  --flagfile=example/ff_fit.txt`. \
After fitting the Regressor you can continue with fit, predict or evaluate the fitted model. \
When fit an already trained model it will fit additional new estimators/trees. For this, n_estimators must be larger than the already existing estimators/trees. \
It is possible to change further parameter for the new trees aswell. Run a warm start with `python main.py --flagfile=ff_fit_warm_start.txt`. \
After a model was fitted use `python main.py --flagfile=example/ff_evaluate.txt` to evaluate the fitted model \
or use `python main.py --flagfile=example/ff_predict.txt` to predict some targets.

## Data Set

The data set was recorded with the help of the Festo Polymer GmbH. The features (`x.csv`) are either parameters explicitly set on the injection molding machine or recorded sensor values. The target value (`y.csv`) is a crucial length measured on the parts. We measured with a high precision coordinate-measuring machine at the Laboratory for Machine Tools (WZL) at RWTH Aachen University.