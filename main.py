# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from absl import app, flags, logging
from rf_regressor import Regressor
from s3_smart_open import to_txt, read_pd_fth, to_json, to_pd_fth
import pandas as pd

flags.DEFINE_string('model_name',None,'Name of the model to save/load')
flags.DEFINE_string('input_path',None,'Specify the a local or s3 object storage path where the input files are stored')
flags.DEFINE_string('output_path',None,'Specify the path where the output files will be stored')
flags.DEFINE_string('filename_x',None,'Filename of Dataframe with features')
flags.DEFINE_string('filename_y',None,'Filename of Dataframe with the target')
flags.DEFINE_string('y_col_name',None,'Name of the y column to train or to predict')
flags.DEFINE_enum('stage',None,['fit','predict','evaluate'],'Wether to fit, predict or evaluate the regressor')
flags.DEFINE_list('metrics',None,'Metrics for evaluation. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/metrics')
flags.DEFINE_integer('n_estimators',100,'The number of trees in the forest.')
flags.DEFINE_enum('criterion','mse',['mse','mae'],'The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.')
flags.DEFINE_string('max_depth',None,'The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.')
flags.DEFINE_string('min_samples_split','2','The minimum number of samples required to split an internal node.')
flags.DEFINE_string('min_samples_leaf','1','The minimum number of samples required to be at a leaf node.')
flags.DEFINE_float('min_weight_fraction_leaf',0.0,'The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.')
flags.DEFINE_string('max_features','auto','The number of features to consider when looking for the best split.')
flags.DEFINE_string('max_leaf_nodes',None,'Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.')
flags.DEFINE_float('min_impurity_decrease',0.0,'A node will be split if this split induces a decrease of the impurity greater than or equal to this value.')
flags.DEFINE_string('min_impurity_split',None,'Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.')
flags.DEFINE_boolean('bootstrap',True,'Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.')
flags.DEFINE_boolean('oob_score',False,'whether to use out-of-bag samples to estimate the R^2 on unseen data.')
flags.DEFINE_integer('n_jobs',1,'The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.')
flags.DEFINE_string('random_state',None,'Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).')
flags.DEFINE_integer('verbose',0,'Controls the verbosity when fitting and predicting.')
flags.DEFINE_boolean('warm_start',False,'When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.')
flags.DEFINE_float('ccp_alpha',0.0,'Complexity parameter used for Minimal Cost-Complexity Pruning. ')
flags.DEFINE_string('max_samples',None,'If bootstrap is True, the number of samples to draw from X to train each base estimator.')
flags.DEFINE_boolean('print_tree',True,'Wether to save a svg of the first decision tree after fit, predict or evaluate. Big Trees may need a lot of coputional time!')


FLAGS = flags.FLAGS

flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('input_path')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('stage')
flags.mark_flag_as_required('filename_x')


def main(argv):
    """Fit, predict or evaluate datasets with a sklearn Random Forest Regressor.
    """    
    del argv

    if FLAGS.n_jobs == -1:
        logging.info('Using all available processors!')

    to_txt(FLAGS.output_path,'flags_'+FLAGS.stage+'.txt',FLAGS.flags_into_string())
    X = read_pd_fth(FLAGS.input_path, FLAGS.filename_x)

    if FLAGS.stage == 'fit':

        if FLAGS.warm_start == True:
            logging.warning('This execution will overwrite the model when the input path and the output path are the same!')

        regr = Regressor(model_name=FLAGS.model_name,
                            n_estimators=FLAGS.n_estimators, 
                            criterion=FLAGS.criterion, 
                            max_depth=FLAGS.max_depth, 
                            min_samples_split=FLAGS.min_samples_split,
                            min_samples_leaf=FLAGS.min_samples_leaf,
                            min_weight_fraction_leaf=FLAGS.min_weight_fraction_leaf,
                            max_features=FLAGS.max_features,
                            max_leaf_nodes=FLAGS.max_leaf_nodes,
                            min_impurity_decrease=FLAGS.min_impurity_decrease,
                            min_impurity_split=FLAGS.min_impurity_split,
                            bootstrap=FLAGS.bootstrap,
                            oob_score=FLAGS.oob_score,
                            n_jobs=FLAGS.n_jobs,
                            random_state=FLAGS.random_state,
                            verbose=FLAGS.verbose,
                            warm_start=FLAGS.warm_start,
                            ccp_alpha=FLAGS.ccp_alpha,
                            max_samples=FLAGS.max_samples
                            )

        y = read_pd_fth(FLAGS.input_path, FLAGS.filename_y, FLAGS.y_col_name,col_limit=1)
        regr.fit(X,y,FLAGS.input_path)
        regr.save(FLAGS.output_path)
        if FLAGS.print_tree:
            regr.print_tree(FLAGS.output_path,X.columns)
        return

    regr = Regressor.load(FLAGS.input_path,FLAGS.model_name)

    if FLAGS.stage == 'predict':
        y_pred = regr.predict(X)
        df_pred = pd.DataFrame(y_pred,columns=['Prediction'])
        to_pd_fth(FLAGS.output_path,FLAGS.model_name+'_results.fth',df_pred)
        logging.info(df_pred.head(n=5))

    if FLAGS.stage == 'evaluate':
        y = read_pd_fth(FLAGS.input_path, FLAGS.filename_y, FLAGS.y_col_name, col_limit=1)
        res = regr.evaluate(X,y,FLAGS.metrics)
        to_json(FLAGS.output_path,FLAGS.model_name+'_metrics.json',res)
        logging.info('Metrics: {}'.format(res))

    if FLAGS.print_tree:
        regr.print_tree(FLAGS.output_path,X.columns)

if __name__ == '__main__':
    app.run(main)