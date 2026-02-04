import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
from operator import itemgetter

import faiss # Facebook AI Similarity Search library that contains IndexLSH
from tqdm import tqdm

class LSHfaissSimilarity(object):
    """
    ANN class to compute the similarity in an approximated way by exploiting LSH
    """

    def __init__(self, data, num_neighbors, similarity, implicit, nbits):
        self._data = data
        self._ratings = data.train_dict  # TODO capire se serve oppure no, è un dizionario {UserId: {ItemId:Rating}}
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._nbits = nbits
        self._implicit = implicit # tells whether to use the ratings as explicit or implicit feedbacks

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
        # d is the dimensionality of the data we need to index -> the items are represented in terms of users
        # nbits controls the number of buckets we create
        self._index_faiss_lsh = faiss.IndexLSH(len(self._data.users), self._nbits)



    def initialize(self):
        """
        This function initialize the data model
        """
        self.supported_similarities = ['cosine']
        print(f"\nSupported Similarities: {self.supported_similarities}\n")

        # initialize the similarity matrix
        self._similarity_matrix = np.empty((len(self._items), len(self._items)))
        # process the similarity matrix by giving the similarity parameter
        self.process_similarity(self._similarity)  # the resulting matrix will be an ndarray

        # Calculate the CR
        n_candidates_pair= np.count_nonzero(self._similarity_matrix)
        CR= n_candidates_pair / (len(self._items) * len(self._items))
        print(f"CR: {CR}")


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

        del self._similarity_matrix

    def process_similarity(self, similarity):
        print("Building the index with FAISS LSH...")
        # here we exploit the FAISS IndexLSH object to compute the similarity
        self._index_faiss_lsh.add(self._URM.T.toarray())
        print("Index built successfully!")
        print("Retrieving the candidate neighbors")
        # retrieve the k-neighbors
        _, candidates = self._index_faiss_lsh.search(self._URM.T.toarray(), self._num_neighbors)

        # TODO: il sampling effettuato è con replacement, quindi abbiamo meno di k vicini -> pensare ad una soluzione
        # candidates is a 2-d Numpy array, the i-th row contains the neighbors of the i-th item
        # we need to use it to compute the similarity matrix
        if similarity == "cosine":
            similarity_function = lambda a, b: (1 + pairwise_distances(a, b, metric="cosine", n_jobs=-1))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\n")
        print("Computing the similarity matrix...")
        for item, neighbors in enumerate(tqdm(candidates)):
            self._similarity_matrix[item, neighbors] = similarity_function(self._URM.T[neighbors], self._URM.T[item]).reshape(-1)
        print("Similarity matrix computed successfully!")



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
        saving_dict['_nbits'] = self._nbits
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']
        self._nbits = saving_dict['_nbits']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
