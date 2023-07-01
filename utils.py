import hashlib
import os
import json
import numpy as np
import lightgbm as lgb
import itertools
import pandas as pd

def get_tree_predictions(model, X):
    leaf_hits = model.predict(X[model.feature_name()], pred_leaf=True)
    
    model_df = model.trees_to_dataframe()
    model_leaves = model_df[model_df['left_child'].isin([None])].copy()

    model_leaves['tree'] = model_leaves['tree_index']
    model_leaves['leaf'] = model_leaves['node_index'].str.split('-L').str[1].astype('int')

    leaf_outputs = model_leaves[['tree','leaf','value']].pivot(index='tree', columns='leaf', values='value').values
    tree_preds = np.take_along_axis(leaf_outputs.T, leaf_hits.reshape(leaf_hits.shape[0], -1), axis=0)
    
    assert np.all((model.predict(X[model.feature_name()])-tree_preds.sum(axis=1))<1e-8)
    
    return tree_preds
  
def get_feature_group_masks(model, feature_groups):
    model_df = model.trees_to_dataframe()
    tree_sample_split_feature = model_df[model_df['left_child']!=None].drop_duplicates(subset=['tree_index'])[['tree_index', 'split_feature']]
    tree_split_category = []
    for i in tree_sample_split_feature['split_feature'].values:
        for j, group in enumerate(feature_groups):
            if i in group:
                tree_split_category.append(j)
    tree_split_category = np.array(tree_split_category)
    
    return [tree_split_category == i for i in range(len(feature_groups))]

def get_group_cov(booster, test, interaction_constraints):
    features = pd.Series(booster.feature_name())
    groups = [list(features[g].values) for g in interaction_constraints]
    group_masks = get_feature_group_masks(booster, groups)
    tree_preds = get_tree_predictions(booster, test)
    group_pred_parts = np.array([tree_preds.dot(mask) for mask in group_masks]).T
    return np.cov(group_pred_parts.T)

class TrainingData:
    def __init__(self, dataReader, target, columns=None):
        self.dataReader = dataReader
        self.name = self.dataReader.name
        self.target_ix = self.dataReader.targets.index(target)
        self.train_y, self.valid_y, self.test_y = self.dataReader.getLabels(self.target_ix)
        self.columns = columns
        self.categorical = dataReader.getCategorical()
        self.train, self.valid = self.dataReader.getLgbData([
            self.dataReader.getTrain(),
            self.dataReader.getValid(),
        ], columns=self.columns)
        self.train.set_label(self.train_y)
        self.valid.set_label(self.valid_y)

class Trainer:
    def __init__(self, cache_dir, default_params, nocache_params={}):
        self.cache_dir = cache_dir
        self.default_params = default_params
        self.nocache_params = nocache_params

    def train_model(self, trainingData, override_params={}, force_retrain=False):
        new_params = dict(self.default_params, **override_params)
        model_hash = hashlib.md5(json.dumps({
            'dataset_name': trainingData.name,
            'target_ix': trainingData.target_ix,
            'columns': trainingData.columns,
            'parameters': new_params,
        }).encode()).hexdigest()
        model_path = os.path.join(self.cache_dir, "%s.txt"%model_hash)
        if os.path.isfile(model_path) and not force_retrain:
            return lgb.Booster(model_file=model_path)

        new_params = dict(new_params, **self.nocache_params)
        num_boost_round = new_params['num_boost_round']
        del new_params['num_boost_round']
        booster = lgb.train(
            new_params,
            trainingData.train,
            valid_sets=trainingData.valid,
            num_boost_round=num_boost_round,
            categorical_feature=trainingData.categorical,
            callbacks=[lgb.log_evaluation(0)],
        )
        booster.save_model(model_path)
        return booster

class Evaluator:
    def __init__(self, dataReader, target, eval_on='test'):
        self.target_ix = dataReader.targets.index(target)
        self.dataReader = dataReader
        self.train_y, self.valid_y, self.test_y = dataReader.getLabels(self.target_ix)
        self.train =  self.dataReader.getTrain()
        self.valid =  self.dataReader.getValid()
        self.test =  self.dataReader.getTest()

        if(eval_on == 'train'):
            self.train = self.train
            self.train_y = self.train_y
        elif(eval_on == 'valid'):
            self.valid = self.valid
            self.valid_y = self.valid_y
        elif(eval_on == 'test'):
            self.eval = self.test
            self.eval_y = self.test_y

        self.base_mse = self.get_mse(self.eval_y, self.train_y.mean())

    def get_mse(self, real, pred):
        if type(pred) == int or type(pred) == float:
            pred = np.ones_like(real)*pred
        return np.power(real-pred, 2).mean()

    def get_model_mse(self, booster):
        mse = self.get_mse(self.eval_y, booster.predict(self.eval[booster.feature_name()]))
        return min(mse, self.base_mse)

    def get_feature_matrix(self, key_list, results_dict):
        colcomb = list(itertools.combinations(key_list, 2))
        colcomb_ix = list(itertools.combinations(range(len(key_list)), 2))
        ncols = len(key_list)
        mat = np.zeros((ncols, ncols))
        for ix, (i,j) in enumerate(colcomb_ix):
            mat[i,j] = results_dict[colcomb[ix]]
            mat[j,i] = mat[i,j]
        return mat

    def get_feature_vector(self, key_list, results_dict):
        cols = key_list
        cols_ix = range(len(key_list))
        ncols = len(key_list)
        vec = np.zeros(ncols)
        for ix, i in enumerate(cols_ix):
            vec[i] = results_dict[cols[ix]]
        return vec