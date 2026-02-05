import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances

from elliot.recommender.ann.lsh import LSHBuilder
from operator import itemgetter
from elliot.utils.custom_logging import add_CR_instance


class ANNLSHSimilarity(object):
    """
    ANN class to compute the similarity in an approximated way by exploiting LSH
    """

    def __init__(self, data, num_neighbors, similarity, implicit, validate, n_hash, n_tables,
                 similarity_threshold, w=1, csv_path=None):
        self._data = data
        self._ratings = data.train_dict  # TODO capire se serve oppure no, è un dizionario {UserId: {ItemId:Rating}}
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit # tells whether to use the ratings as explicit or implicit feedbacks
        self._validate = validate # lsh parameter that tells to check the actual similarity for the candidates
        self._n_hash = n_hash
        self._n_tables = n_tables
        self._n_tables = n_tables
        self._similarity_threshold = similarity_threshold # similarity threshold used during the validation of candidates
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

        # instantiate the LSH object
        lsh_params = dict()
        if self._similarity == 'euclidean':
            lsh_params['type'] = 'e2lsh'
            lsh_params['w'] = w
        elif self._similarity == 'jaccard':
            lsh_params['type'] = 'onebitminhash'
            # representation of items in terms of users, sorted according to the private indexing
            self._item_profiles = [set(self._data.sp_i_train.T.getrow(i).toarray()[0].nonzero()[0]) for i in range(0, self._data.sp_i_train.shape[1])]
        elif self._similarity == 'cosine':
            lsh_params['type'] = 'random-projection'
        # d is the dimensionality of the data, it is required for the E2LSH and RandomProjection method
        # for the item-based methods, d is the number of users on the platform
        self._lsh_index = LSHBuilder.build(d=len(self._users), r=self._similarity_threshold, k=self._n_hash, L=self._n_tables, lsh_params=lsh_params, validate=validate)

    def initialize(self):
        """
        This function initialize the data model
        """
        self.supported_similarities = ['cosine']
        self.supported_dissimilarities = ['euclidean', 'jaccard']
        print(f"\nSupported Similarities: {self.supported_similarities}\n")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        # initialize the similarity matrix
        self._similarity_matrix = np.empty((len(self._items), len(self._items)))
        # process the similarity matrix by giving the similarity parameter
        self.process_similarity(self._similarity)  # the resulting matrix will be an ndarray

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
                                     shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()
        self._preds = self._URM.dot(W_sparse)
        
        # Calculate and log CR
        if self._csv_path:
            if self._similarity == 'euclidean':
                n_candidates_pair = np.count_nonzero(self._similarity_matrix)
            else:
                n_candidates_pair = (self._URM.T @ self._URM).nnz
            CR = n_candidates_pair / (n_items * n_items)
            print(f"CR: {CR}")
            
            meta_data = {
                "model": "ItemANNLSH",
                "neighbors": self._num_neighbors,
                "similarity": self._similarity,
                "n_hash": self._n_hash,
                "n_tables": self._n_tables,
                "similarity_threshold": self._similarity_threshold,
                "w": 1, # default w is 1, maybe should log actual w if used
                "CR": CR
            }
            add_CR_instance(self._csv_path, meta_data)

        del self._similarity_matrix

    def process_similarity(self, similarity):
        # here we exploit the LSH object to compute the similarity
        if similarity == "cosine":
            self._lsh_index.preprocess(self._URM.T.toarray())
        elif similarity == "euclidean":
            self._lsh_index.preprocess(self._URM.T.toarray())
        elif similarity in ['jaccard']:
            # self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T.toarray(), metric=similarity)))
            self._lsh_index.preprocess(self._item_profiles)
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\n")
        # TODO: il sampling effettuato è con replacement, quindi abbiamo meno di k vicini -> pensare ad una soluzione
        # candidates is a dictionary {ItemID:[ID of candidate items to be similar]}, we need to use it to compute the similarity matrix
        if similarity == "cosine":
            similarity_function = lambda a, b: (1 + pairwise_distances(a,b, metric="cosine", n_jobs=-1))
        elif similarity == "euclidean":
            similarity_function = lambda a, b: 1 / (1 + pairwise_distances(a,b, metric="euclidean", n_jobs=-1))
        elif similarity == "jaccard":
            similarity_function = lambda a, b: 1 / (1 + pairwise_distances(a,b, metric="jaccard", n_jobs=-1))
        _, _, candidates, _, _ = self._lsh_index.preprocess_query(self._URM.T.toarray())
        for item, neighbors in enumerate(candidates):
            # Get the representation vector for the current item
            item_vector = self._URM.T[item].toarray()

            # Compute similarities only with the neighbors and Populate the similarity matrix
            neighbor_vectors = self._URM.T[list(neighbors)].toarray()
            self._similarity_matrix[item, list(neighbors)] = similarity_function(neighbor_vectors, item_vector).reshape(-1)



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
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
