import pickle

import numpy as np
from scipy import sparse

from annoy import AnnoyIndex
from operator import itemgetter
from tqdm import tqdm
from elliot.utils.custom_logging import add_CR_instance

class ANNOYSimilarity(object):
    """
    ANN class to compute the similarity in an approximated way by exploiting LSH
    """

    def __init__(self, data, num_neighbors, similarity, implicit, n_trees, search_k, csv_path=None):
        self._data = data
        self._ratings = data.train_dict  # TODO capire se serve oppure no, è un dizionario {UserId: {ItemId:Rating}}
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._n_trees = n_trees
        self._search_k = search_k
        self._implicit = implicit # tells whether to use the ratings as explicit or implicit feedbacks
        self._csv_path = csv_path

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

        self._users = self._data.users  # contains a list of userIDs
        self._items = self._data.items  # contains a list of itemIDs
        self._private_users = self._data.private_users  # contains the mapping from private ID to public ID
        self._public_users = self._data.public_users  # contains the mapping from public ID to private ID
        self._private_items = self._data.private_items  # contains the mapping from private ID to public ID
        self._public_items = self._data.public_items  # contains the mapping from public ID to private ID

        # instantiate an IndexLSH object
        # first input argument is d, second one is nbits
        # d is the dimensionality of the data we need to index -> the users are represented in terms of items
        # nbits controls the number of buckets we create
        self._index_annoy = AnnoyIndex(f=len(self._data.items), metric=self._similarity)


    def initialize(self):
        """
        This function initialize the data model
        """
        self.supported_dissimilarities = ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        print(f"\nSupported distances: {self.supported_dissimilarities}\n")

        # initialize the similarity matrix
        self._similarity_matrix = np.empty((len(self._users), len(self._users)))
        # process the similarity matrix by giving the similarity parameter
        self.process_similarity(self._similarity)  # the resulting matrix will be an ndarray
        
        # Calculate and log CR
        if self._csv_path:
            n_users = self._data.num_users
            if self._similarity == 'euclidean':
                n_candidates_pair = np.count_nonzero(self._similarity_matrix)
            else:
                n_candidates_pair = (self._URM @ self._URM.T).nnz
            CR = n_candidates_pair / (n_users * n_users)
            print(f"CR: {CR}")
            
            meta_data = {
                "model": "UserAnnoy",
                "neighbors": self._num_neighbors,
                "similarity": self._similarity,
                "implicit": self._implicit,
                "n_trees": self._n_trees,
                "search_k": self._search_k,
                "CR": CR
            }
            add_CR_instance(self._csv_path, meta_data)
        
        data, rows_indices, cols_indptr = [], [], []

        column_row_index = np.arange(len(self._users), dtype=np.int32)

        for user_idx in range(len(self._users)):
            cols_indptr.append(len(data))
            column_data = self._similarity_matrix[:, user_idx]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._num_neighbors:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._users), len(self._users)), dtype=np.float32).tocsr()
        self._preds = W_sparse.dot(self._URM)

        del self._similarity_matrix

    def process_similarity(self, similarity):
        if similarity not in ["angular", "euclidean","manhattan","hamming","dot"]:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\n")

        print("Building ANNOY index...")
        # insert the data into the index
        for i in tqdm(range(len(self._data.users))):
            self._index_annoy.add_item(i, self._URM[i].toarray()[0])

        # build the index
        self._index_annoy.build(n_trees=self._n_trees, n_jobs=-1)
        print("ANNOY index built.")

        print("Retrieving neighbors from index...")
        # retrieve the data
        for user in tqdm(range(len(self._data.users))):
            # find the k nearest neighbors and the relative distances
            neighbors, distances = self._index_annoy.get_nns_by_item(user, self._num_neighbors,
                                                                     search_k=self._search_k,
                                                                     include_distances=True)
            # populate the similarity matrix, converting the distances into similarities
            self._similarity_matrix[user, neighbors] = 1 / (1 + np.array(distances))
        print("Neighbors retrieved.")


    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
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
        saving_dict['_n_trees'] = self._n_trees
        saving_dict['_search_k'] = self._search_k
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']
        self._n_trees = saving_dict['_n_trees']
        self._search_k = saving_dict['_search_k']


    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)

