import numpy as np
import lightgbm as lgb

from .outliers_detector import OutliersFinder
from .misc import prepare_lightgbm_sets, remove_docs, rename_dict_key


class SOUR():
    def __init__(self, queries, labels, qs_len, eval_set=None):
        self.queries = queries
        self.labels = labels
        self.qs_len = qs_len

        self.eval_set = eval_set

    def train(self, params, outliers_type, start, end, p_sour=1, last_sour=False, cutoff=None, min_neg_rel=0, **kwargs):
        if cutoff is None:
            cutoff = params['eval_at']

        end_num_boost_round = end
        is_curr = False
        if type(end) is list:
            end = sorted(end)
            end_num_boost_round = end[-1]
            is_curr = True
        
        safe_params = rename_dict_key(params, "num_iterations", ["num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "nrounds", "num_boost_round", "n_estimators", "max_iter"])
        safe_params = rename_dict_key(params, "early_stopping_round", ["early_stopping_rounds", "early_stopping", "n_iter_no_change"])

        outliers_finder = OutliersFinder(len(self.labels), outliers_type, start, end, cutoff, min_neg_rel, last_sour, is_curr)
        train_set, valid_sets, valid_names = prepare_lightgbm_sets((self.queries, self.labels, self.qs_len), include_train=True)

        save_early_stop = None
        save_num_iter = 100

        flag_num_iters = False
        if "num_iterations" in kwargs:
            save_num_iter = kwargs.pop("num_iterations")
        if "num_iterations" in safe_params:
            save_num_iter = safe_params.pop("num_iterations")
            flag_num_iters = True
        if "early_stopping_rounds" in safe_params:
            save_early_stop = safe_params.pop("early_stopping_rounds")

        lgb.train(params=safe_params, train_set=train_set, num_boost_round=end_num_boost_round, valid_sets=valid_sets, valid_names=valid_names, feval=outliers_finder, **kwargs)

        safe_params["early_stopping_rounds"] = save_early_stop
        model = kwargs.pop("init_model") if "init_model" in kwargs else None

        pre_end = 0
        num_iterations = save_num_iter
        for i, idx_to_removed in enumerate(outliers_finder.get_outliers_ids(p_sour=p_sour, last_sour=last_sour, curr_sour=is_curr)):
            clean_queries, clean_labels, clean_qs_lens = remove_docs(self.queries, self.labels, self.qs_len, idx_to_removed)
            train_set, valid_sets, valid_names = prepare_lightgbm_sets((clean_queries, clean_labels, clean_qs_lens), self.eval_set)
            
            if is_curr:
                if i >= len(end):
                    num_iterations = save_num_iter - pre_end if save_num_iter > pre_end else save_num_iter
                else:
                    num_iterations = end[i] - pre_end
                    pre_end = end[i]

                if flag_num_iters:
                    safe_params["num_iterations"] = num_iterations
            
            model = lgb.train(params=safe_params, train_set=train_set, num_boost_round=num_iterations, valid_sets=valid_sets, valid_names=valid_names, init_model=model, **kwargs)

        return model