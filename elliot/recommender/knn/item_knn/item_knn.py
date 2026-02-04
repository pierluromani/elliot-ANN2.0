"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import time

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.item_knn.item_knn_similarity import Similarity
from elliot.recommender.knn.item_knn.aiolli_ferrari import AiolliSimilarity
from elliot.recommender.base_recommender_model import init_charger

from tqdm import tqdm


class ItemKNN(RecMixin, BaseRecommenderModel):
    r"""
    Amazon.com recommendations: item-to-item collaborative filtering

    For further details, please refer to the `paper <http://ieeexplore.ieee.org/document/1167344/>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type ('aiolli', 'classical')

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ItemKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          implementation: aiolli
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implementation", "implementation", "imp", "standard", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_shrink", "shrink", "shrink", 0, None, None),
            ("_normalize", "normalize", "norm", True, None, None),
            ("_asymmetric_alpha", "asymmetric_alpha", "asymalpha", False, None, lambda x: x if x else ""),
            ("_tversky_alpha", "tversky_alpha", "tvalpha", False, None, lambda x: x if x else ""),
            ("_tversky_beta", "tversky_beta", "tvbeta", False, None, lambda x: x if x else ""),
            ("_row_weights", "row_weights", "rweights", None, None, lambda x: x if x else "")
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        if self._implementation == "aiolli":
            self._model = AiolliSimilarity(data=self._data,
                                           maxk=self._num_neighbors,
                                           shrink=self._shrink,
                                           similarity=self._similarity,
                                           implicit=self._implicit,
                                           normalize=self._normalize,
                                           asymmetric_alpha=self._asymmetric_alpha,
                                           tversky_alpha=self._tversky_alpha,
                                           tversky_beta=self._tversky_beta,
                                           row_weights=self._row_weights)
        else:
            if (not self._normalize) or (self._asymmetric_alpha) or (self._tversky_alpha) or (self._tversky_beta) or (self._row_weights) or (self._shrink):
                self.logger.info("Options normalize, asymmetric_alpha, tversky_alpha, tversky_beta, row_weights are ignored with standard implementation. Try with implementation: aiolli")
            # self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)
            self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity,
                                     implicit=self._implicit, alpha=self._asymmetric_alpha,
                                     tversky_alpha=self._tversky_alpha, tversky_beta=self._tversky_beta,
                                     csv_path=self._csv_path)
    def get_single_recommendation(self, mask, k, *args):
        # return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}
        recs = {}
        for i in tqdm(range(0, len(self._ratings.keys()), 1024), desc="Processing batches",
                      total=len(self._ratings.keys()) // 1024 + (1 if len(self._ratings.keys()) % 1024 != 0 else 0)):
            batch = list(self._ratings.keys())[i:i + 1024]
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
        return f"ItemKNN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()

