"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Marco Valentini'
__email__ = 'm.valentini7@phd.poliba.it'

import pickle
import time

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.ann.user_fair_ann.user_fair_ann_similarity import LSHSimilarity
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger

from tqdm import tqdm


class UserFairANN(RecMixin, BaseRecommenderModel):
    r"""
    Fair ANN recommendations: user-to-user collaborative filtering, with equal opportunities for all the users in the neighborhood

    For further details, please refer to the `paper https://dl.acm.org/doi/pdf/10.1145/3502867`_

    Args:
        neighbors: Number of user neighbors
        similarity: Similarity function ('euclidean', 'cosine', 'jaccard')
        sampling_strategy: Strategy to sample from the buckets ('opt', 'uniform', 'weighted_uniform', 'approx_degree', 'rank', 'no_sampling')
        n_hash: number of hash functions
        n_tables: number of tables (repetitions)
        similarity_threshold: the minimum similarity threshold to be considered as similar after validation
        w: it is a scaling factor used when we use E2LSH


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        UserFairANN:
          meta:
            verbose: True
            save_recs: True
          neighbors: 40
          similarity: cosine
          sampling_strategy: uniform
          validate: False
          n_hash: 5
          n_tables: 1
          similarity_threshold: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # Define Parameters as tuples: (variable_name, public_name, shortcut, default, reading_function, printing_function)

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_sampling_strategy", "sampling_strategy", "samp_strat", "uniform", None, None),
            ("_validate", "validate", "val", False, None, None),
            ("_n_hash", "n_hash", "n_h", 1, int, None),
            ("_n_tables", "n_tables", "n_t", 1, int, None),
            ("_similarity_threshold", "similarity_threshold", "sim_thres", 0.5, float, None),
            ("_w", "w", "w", 1, int, None)
        ]
        self.autoset_params()
        # Here we have a dictionary containing dictionaries UserID: {ItemID:Rating}
        self._ratings = self._data.train_dict
        # initialize the LSH Similarity model
        self._model = LSHSimilarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity,
                                    sampling_strategy=self._sampling_strategy, implicit=self._implicit,
                                    validate=self._validate, n_hash=self._n_hash, n_tables=self._n_tables,
                                    similarity_threshold=self._similarity_threshold, w=self._w, csv_path=self._csv_path)

    def get_single_recommendation(self, mask, k, *args):
        # return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}
        recs = {}
        for i in tqdm(range(0, len(self._ratings.keys()), 1024), desc="Processing batches", total=len(self._ratings.keys()) // 1024 + (1 if len(self._ratings.keys()) % 1024 != 0 else 0)):
            batch = list(self._ratings.keys())[i:i+1024]
            mat = self._model.get_user_recs_batch(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    @property
    def name(self):
        return f"UserFairANN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()
