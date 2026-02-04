import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, \
    manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import similaripy as sim
import random
from copy import deepcopy
from operator import itemgetter
from elliot.utils.custom_logging import add_CR_instance


class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, num_neighbors, similarity, implicit, pre_post_processing, **kwargs):
        self._data = data
        self._ratings = data.train_dict
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit
        self._pre_post_processing = pre_post_processing
        self._csv_path = kwargs.get('csv_path', None)

        if self._implicit:
            self._URM = self._data.sp_i_train  # this stores the interactions
        else:
            self._URM = self._data.sp_i_train_ratings  # this stores the ratings

        self._users = self._data.users  # contains a list of userIDs
        self._items = self._data.items  # contains a list of itemIDs
        self._private_users = self._data.private_users  # contains the mapping from private ID to public ID
        self._public_users = self._data.public_users  # contains the mapping from public ID to private ID
        self._private_items = self._data.private_items  # contains the mapping from private ID to public ID
        self._public_items = self._data.public_items  # contains the mapping from public ID to private ID

    def initialize(self):
        """
        This function initialize the data model -> create the LSH index
        """

        self.supported_similarities = ["cosine", "dot"]
        self.supported_dissimilarities = ["euclidean", "manhattan", "haversine", "chi2", 'cityblock', 'l1', 'l2',
                                          'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
                                          'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto',
                                          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                          'yule']
        print(f"\nSupported Similarities: {self.supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        if self._pre_post_processing == None or self._pre_post_processing in ['value', 'parity']:
            # avoid pre processing if a post processing will be applied
            pass
        elif self._pre_post_processing == 'interactions-users':
            # convert the train and test dict into private mappings
            self._data.private_train_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.train_dict.items()}
            self._data.private_test_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.test_dict.items()}
            # read the information about the group and convert it into private IDs
            g1 = self._data.side_information.ItemPopularityUserTolerance.user_group_map['0']  # diverse and niche-focused group
            g2 = self._data.side_information.ItemPopularityUserTolerance.user_group_map['1']  # interested in most popular group
            # map the public IDs to the private IDs
            g1 = [self._data.public_users.get(i) for i in g1]
            g2 = [self._data.public_users.get(i) for i in g2]
            # count the interactions of the training set
            train_interactions_g1 = [[user, item] for user, items in self._data.private_train_dict.items() for item in
                                     items if user in g1]
            train_interactions_g2 = [[user, item] for user, items in self._data.private_train_dict.items() for item in
                                     items if user in g2]
            # sample the interactions to balance the 2 groups
            if len(train_interactions_g1) < len(train_interactions_g2):
                n_to_remove = len(train_interactions_g2) - len(train_interactions_g1)
                train_interactions_to_remove = random.sample(train_interactions_g2, n_to_remove)
            elif len(train_interactions_g2) < len(train_interactions_g1):
                n_to_remove = len(train_interactions_g1) - len(train_interactions_g2)
                train_interactions_to_remove = random.sample(train_interactions_g1, n_to_remove)
            else:
                train_interactions_to_remove = []
            # remove the interactions to balance them -> set them to 0
            rows_to_remove, cols_to_remove = zip(*train_interactions_to_remove)
            self._URM[rows_to_remove, cols_to_remove] = 0
        elif self._pre_post_processing == 'interactions-items':
            # convert the train and test dict into private mappings
            self._data.private_train_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.train_dict.items()}
            self._data.private_test_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.test_dict.items()}
            self._data.inverse_train_dict = {}
            for user, items in self._data.private_train_dict.items():
                for item in items:
                    if item not in self._data.inverse_train_dict.keys():
                        self._data.inverse_train_dict[item] = []
                    self._data.inverse_train_dict[item].append(user)
            # do the same for the test dict
            self._data.inverse_test_dict = {}
            for user, items in self._data.private_test_dict.items():
                for item in items:
                    if item not in self._data.inverse_test_dict.keys():
                        self._data.inverse_test_dict[item] = []
                    self._data.inverse_test_dict[item].append(user)
            # read the information about the group and convert it into private IDs
            g1 = self._data.side_information.ItemPopularityUserTolerance.item_group_map['0']  # long-tail group
            g2 = self._data.side_information.ItemPopularityUserTolerance.item_group_map['1']  # most popular group
            # map the public IDs to the private IDs
            g1 = [self._data.public_items.get(i) for i in g1]
            g2 = [self._data.public_items.get(i) for i in g2]
            # count the interactions in the group 1
            train_interactions_g1 = [[user, item] for item, users in self._data.inverse_train_dict.items() for user in
                                     users if item in g1]
            train_interactions_g2 = [[user, item] for item, users in self._data.inverse_train_dict.items() for user in
                                     users if item in g2]
            # sample the interactions to balance the 2 groups
            if len(train_interactions_g1) < len(train_interactions_g2):
                n_to_remove = len(train_interactions_g2) - len(train_interactions_g1)
                train_interactions_to_remove = random.sample(train_interactions_g2, n_to_remove)
            elif len(train_interactions_g2) < len(train_interactions_g1):
                n_to_remove = len(train_interactions_g1) - len(train_interactions_g2)
                train_interactions_to_remove = random.sample(train_interactions_g1, n_to_remove)
            else:
                train_interactions_to_remove = []
            # remove the interactions to balance them -> set them to 0
            rows_to_remove, cols_to_remove = zip(*train_interactions_to_remove)
            self._URM[rows_to_remove, cols_to_remove] = 0
        elif self._pre_post_processing == 'users-resampling':
            # read the information about the group and convert it into private IDs
            g1 = self._data.side_information.ItemPopularityUserTolerance.user_group_map[
                '0']  # diverse and niche-focused group
            g2 = self._data.side_information.ItemPopularityUserTolerance.user_group_map[
                '1']  # interested in most popular group
            # sample the users to balance the 2 groups -> sampling without replacement
            if len(g1) < len(g2):
                resampled_g2 = np.random.choice(g2, len(g1), replace=False)
                g2 = resampled_g2
            elif len(g2) < len(g1):
                resampled_g1 = np.random.choice(g1, len(g2), replace=False)
                g1 = resampled_g1
            # map the public IDs to the private IDs
            g1 = [self._data.public_users.get(i) for i in g1]
            g2 = [self._data.public_users.get(i) for i in g2]
            # reduce the items
            self._reduced = g1 + g2
            # update the URM
            self._old_URM = self._URM.copy()
            self._URM = self._URM[self._reduced, :]
        else:
            raise ValueError("Pre-Post Processing: value for parameter 'pre_post_processing' not recognized."
                             f"\nAllowed values are: None, 'value', 'parity', 'interactions-users', 'interactions-items', 'users-resampling'."
                             f"\nPassed value was {self._pre_post_processing}\n")

        # compute the similarity matrix with similaripy to speed up the process
        if self._similarity == "cosine":
            W_sparse = sim.cosine(self._URM.T, k=self._num_neighbors, format_output='csr')
        elif self._similarity == "asym":
            W_sparse = sim.asymmetric_cosine(self._URM.T, alpha=self._alpha, k=self._num_neighbors,
                                             format_output='csr')
        elif self._similarity == "dot":
            W_sparse = sim.dot_product(self._URM.T, k=self._num_neighbors, format_output='csr')
        elif self._similarity == "jaccard":
            W_sparse = sim.jaccard(self._URM.T, k=self._num_neighbors, binary=True, format_output='csr')
        elif self._similarity == "dice":
            W_sparse = sim.dice(self._URM.T, k=self._num_neighbors, binary=True, format_output='csr')
        elif self._similarity == "tversky":
            W_sparse = sim.tversky(self._URM.T, k=self._num_neighbors, alpha=self._tversky_alpha,
                                   beta=self._tversky_beta, binary=True, format_output='csr')
        elif self._similarity == "euclidean":
            self._similarity_matrix = np.empty((len(self._items), len(self._items)))
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM.T)))  # avoid function call
            data, rows_indices, cols_indptr = [], [], []

            column_row_index = np.arange(len(self._data.items), dtype=np.int32)

            for item_idx in range(len(self._data.items)):
                cols_indptr.append(len(data))
                column_data = self._similarity_matrix[:, item_idx]

                non_zero_data = column_data != 0

                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-self._num_neighbors:]

                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

            W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                         shape=(len(self._data.items), len(self._data.items)),
                                         dtype=np.float32).tocsr()


        self._preds = self._URM.dot(W_sparse)
        
        # Calculate and log CR
        # Calculate and log CR
        if self._csv_path:
            n_items = self._data.num_items
            n_candidates_pair = np.count_nonzero(self._similarity_matrix)
            CR = n_candidates_pair / (n_items * n_items)
            print(f"CR: {CR}")
            
            meta_data = {
                "model": "ItemKNNFairness",
                "neighbors": self._num_neighbors,
                "similarity": self._similarity,
                "implicit": self._implicit,
                "CR": CR
            }
            add_CR_instance(self._csv_path, meta_data)

        if hasattr(self, '_similarity_matrix'):
            del self._similarity_matrix


        # for the item-based algorithm we use the Item Popularity grouping
        if self._pre_post_processing == 'value':
            pass
        elif self._pre_post_processing == 'parity':
            # map the train dict from the public IDs to the private IDs
            self._data.private_train_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.train_dict.items()}
            # map the test dict from the public IDs to the private IDs
            self._data.private_test_dict = {
                self._data.public_users.get(user): [self._data.public_items.get(item) for item in items] for user, items
                in self._data.test_dict.items()}
            # given the train dict, which contains for each user the rated items, we want to compute the inverse of it,
            # which contains for each item the users that rated it
            self._data.inverse_train_dict = {}
            for user, items in self._data.private_train_dict.items():
                for item in items:
                    if item not in self._data.inverse_train_dict.keys():
                        self._data.inverse_train_dict[item] = []
                    self._data.inverse_train_dict[item].append(user)
            # do the same for the test dict
            self._data.inverse_test_dict = {}
            for user, items in self._data.private_test_dict.items():
                for item in items:
                    if item not in self._data.inverse_test_dict.keys():
                        self._data.inverse_test_dict[item] = []
                    self._data.inverse_test_dict[item].append(user)
            # compute the average predicted rating for group 1 on training set
            g1 = self._data.side_information.ItemPopularityUserTolerance.item_group_map['0']  # long-tail group
            # convert the public IDs to the private IDs
            g1 = [self._data.public_items.get(i) for i in g1]
            # take from the train dict only the items belonging to group 1
            self._data.inverse_train_dict_g1 = {ItemID: users for ItemID, users in self._data.inverse_train_dict.items()
                                                if ItemID in g1}
            # take from the test dict only the items belonging to group 1
            self._data.inverse_test_dict_g1 = {ItemID: users for ItemID, users in self._data.inverse_test_dict.items()
                                               if ItemID in g1}
            # given the inverse train dict, I need to create a list of cells to access the predictions matrix
            g1_train_idx = [[user, item] for item, users in self._data.inverse_train_dict_g1.items() for user in users]
            # given the inverse test dict, I need to create a list of cells to access the predictions matrix
            g1_test_idx = [[user, item] for item, users in self._data.inverse_test_dict_g1.items() for user in users]
            # compute the average predicted rating for group 2 on training set
            g2 = self._data.side_information.ItemPopularityUserTolerance.item_group_map['1']  # most popular group
            # convert the public IDs to the private IDs
            g2 = [self._data.public_items.get(i) for i in g2]
            # take from the train dict only the items belonging to group 1
            self._data.inverse_train_dict_g2 = {ItemID: users for ItemID, users in self._data.inverse_train_dict.items()
                                                if ItemID in g2}
            # take from the test dict only the items belonging to group 1
            self._data.inverse_test_dict_g2 = {ItemID: users for ItemID, users in self._data.inverse_test_dict.items()
                                               if ItemID in g2}
            # given the inverse train dict, I need to create a list of cells to access the predictions matrix
            g2_train_idx = [[user, item] for item, users in self._data.inverse_train_dict_g2.items() for user in users]
            rows_g1_train, cols_g1_train = zip(*g1_train_idx)
            rows_g2_train, cols_g2_train = zip(*g2_train_idx)
            avg_y_g1_train = self._preds[rows_g1_train, cols_g1_train].mean()
            avg_y_g2_train = self._preds[rows_g2_train, cols_g2_train].mean()
            # compute the delta between the two averages
            delta_train = avg_y_g2_train - avg_y_g1_train # difference between non-protected and protected group average predicted ratings
            # apply the delta to the test set
            rows_g1_test, cols_g1_test = zip(*g1_test_idx)
            # boost the predictions for the protected group
            self._preds[rows_g1_test, cols_g1_test] = self._preds[rows_g1_test, cols_g1_test] + delta_train
        elif self._pre_post_processing is None or self._pre_post_processing in ['interactions-users', 'interactions-items', 'users-resampling']:
            pass
        else:
            raise ValueError("Pre-Post Processing: value for parameter 'pre_post_processing' not recognized."
                             f"\nAllowed values are: None, 'value', 'parity', 'interactions-users', 'interactions-items', 'users-resampling'."
                             f"\nPassed value was {self._pre_post_processing}\n")
        # restore the original URM with all the users
        if self._pre_post_processing == 'users-resampling':
            self._URM = self._old_URM.copy()

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._URM.T)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM.T @ self._URM).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM.T)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM.T)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM.T)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM.T)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:

            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")


    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_recs_batch(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        users_recs = np.where(mask[u_index, :], self._preds[u_index, :].toarray(), -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]


    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        saving_dict['_pre_post_processing'] = self._pre_post_processing
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']
        self._pre_post_processing = saving_dict['_pre_post_processing']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
