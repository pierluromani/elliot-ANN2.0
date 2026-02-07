import numpy as np
import random
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# implementation taken from `https://github.com/alfahaf/fair-nn`


class LSHBuilder:
    methods = ["opt",
               "uniform",
               "weighted_uniform",
               "approx_degree",
               "rank"]

    @staticmethod
    def build(d, r, k, L, lsh_params, validate=False):
        if lsh_params['type'] == 'e2lsh':
            return E2LSH(k, L, lsh_params['w'], d, r, validate)
        if lsh_params['type'] == 'onebitminhash':
            return OneBitMinHash(k, L, r, validate)
        if lsh_params['type'] == 'random-projection':
            return RandomProjection(k, L, d, r, validate)

    @staticmethod
    def invoke(lsh, method, queries, runs):
        assert method in LSHBuilder.methods  # check that the method is among the allowed ones otherwise there will be an error
        if method == "opt":
            res = lsh.opt(queries, runs)
        if method == "uniform":
            res = lsh.uniform_query(queries, runs)
        if method == "weighted_uniform":
            res = lsh.weighted_uniform_query(queries, runs)
        if method == "approx_degree":
            res = lsh.approx_degree_query(queries, runs)
        if method == "rank":
            res = lsh.rank_query_simulate(queries, runs)
        return res


class LSH:

    def preprocess(self, X):
        self.X = X  # vectors we need to index
        n = len(X)  # number of items in the index
        hvs = self._hash(
            X)  # apply the hashing, for each input data point we will get a new representation with reduced size (equal to k)
        self.tables = [{} for _ in range(self.L)]  # initialize the tables (minhashing)
        for i in tqdm(range(n)):  # iterate over all the data points
            for j in range(self.L):  # iterate over all the tables
                h = self._get_hash_value(hvs[i], j)  # takes the j-th element from the array contained in hvs[i]
                self.tables[j].setdefault(h, set()).add(
                    i)  # in this way we know that the bucket identified by the key h contains the i-th element

    def preprocess_query(self, Y):
        """Collect buckets, bucket sizes, and prefix_sums
        to quickly answer queries."""
        query_buckets = [[] for _ in range(len(Y))]  # for each query we need a bucket
        query_size = [0 for _ in range(len(Y))]  # initialize the query size
        bucket_sizes = [0 for _ in range(len(Y))]  # initialize the bucket sizes
        prefix_sums = [[0 for _ in range(self.L)] for _ in
                       range(len(Y))]  # for each query initialize a list having the number of tables as size
        query_results = [set() for _ in range(len(Y))]  # for each query point we have a list of candidate items

        hvs = self._hash(Y)  # hash the query point (using the same hash functions using during preprocessing)
        for j, q in enumerate(tqdm(hvs)):  # iterate over the hash values computed for the query
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            query_buckets[
                j] = buckets  # store the retrieved buckets for the j-th query point, for each table, the point is stored in a bucket
            s = 0  # initialize s?
            elements = set()  # initialize a set of elements
            for i, (table, bucket) in enumerate(buckets):
                s += len(self.tables[table].get(bucket,
                                                []))  # returns a list of candidate elements to be similar, otherwise an empty one (add its size to s)
                elements |= self.tables[table].get(bucket, set())  # inplace union between sets
                prefix_sums[j][i] = s  # for each query we save the bucket and its related size
            elements = set(x for x in elements
                           if self.is_candidate_valid(Y[j], self.X[
                x]))  # check that it is in the neighborhood, Actually it does nothing
            bucket_sizes[j] = s  # update the bucket size for the j-th query point
            query_size[j] = len(elements)  # update the number of results for the j-th query point
            query_results[j] = elements  # update with the retrieved points for the j-th query point
            # elements is treated as a running list of IDs, at the end we will have a single list of results for each query point
        return (query_buckets, query_size, query_results,
                bucket_sizes,
                prefix_sums)  # if we want to apply standard LSH minhashing/random projection, we can use the buckets contained here
    # query_buckets contains the bucket in which the query point is contained for each table
    # query_size contains the number of candidate items to be similar for each query point
    # query_results the candidate items to be similar for each item


    def get_query_size(self, Y):
        _, query_size, _, _, _ = self.preprocess_query(Y)  # we preprocess the query
        return query_size

    def uniform_query(self, Y, neighbors, runs=100):
        query_bucket, sizes, query_results, _, _ = self.preprocess_query(Y)
        results = {i: [] for i in range(len(Y))}
        for j in range(len(Y)):
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            for _ in range(neighbors * runs):
                if len(query_results[j]) == 0:
                    results[j].append(-1)
                    continue
                while True:
                    table, bucket = query_bucket[j][random.randrange(0, self.L)] # sample randomly one of the buckets containing the j-th query point
                    elements = list(self.tables[table].get(bucket, [-1])) # retrieve a lists of elements containing the buckets
                    p = random.choice(elements)
                    if p != -1 and self.is_candidate_valid(Y[j], self.X[p]):
                        results[j].append(p)
                        break
        return results

    def weighted_uniform_query(self, Y, neighbors, runs=100):
        from bisect import bisect_right
        print("Preprocessing the query...")
        query_buckets, query_size, elements, bucket_sizes, prefix_sums = self.preprocess_query(Y)
        results = {i: [] for i in range(len(Y))}
        # iterate over the query points
        print("Retrieving the neighbors...")
        for j in tqdm(range(len(Y))):
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            for _ in range(neighbors * runs):
                if len(elements[j]) == 0:  # if no results have been found, return -1
                    results[j].append(-1)
                    continue
                while True:
                    i = random.randrange(bucket_sizes[j]) # bucket size is the overall sum of the buckets containing the j-th query point
                    pos = bisect_right(prefix_sums[j], i) # prefix sums contains the interval of each bucket containing the j-th query point
                    table, bucket = query_buckets[j][pos] # pos will have higher probability for the bigger buckets to contain the i-th element
                    # choose randomly a point until it is valid
                    p = random.choice(list(self.tables[table][bucket]))
                    if self.is_candidate_valid(Y[j], self.X[p]):
                        results[j].append(p)
                        break
        return results

    def opt(self, Y, neighbors, runs=100, runs_per_collision=True):
        _, query_size, query_results, _, _ = self.preprocess_query(Y) # query size contains the number of candidates found for that query pont
        results = {i: [] for i in range(len(Y))}

        for j in range(len(Y)):
            elements = list(query_results[j])
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            iterations = neighbors * runs
            if not runs_per_collision:
                iterations = runs
            for _ in range(iterations):
                if query_size[j] == 0:
                    results[j].append(-1)
                    continue
                results[j].append(random.choice(elements))
        return results

    def approx_degree_query(self, Y, neighbors, runs=100):
        from bisect import bisect_right # Rightmost index to insert, so list remains sorted is -> method to choose the bucket according to the size
        query_buckets, query_size, _, bucket_sizes, prefix_sums = self.preprocess_query(Y)
        results = {i: [] for i in range(len(Y))}

        for j in range(len(Y)):
            cache = {}
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            for _ in range(neighbors * runs):
                if bucket_sizes[j] == 0:
                    results[j].append(-1)
                    continue
                while True:
                    i = random.randrange(bucket_sizes[j])
                    pos = bisect_right(prefix_sums[j], i)
                    table, bucket = query_buckets[j][pos] # choose the bucket with probability proportional to its size
                    p = random.choice(list(self.tables[table][bucket])) # choose a point randomly from the bucket
                    # discard not within distance threshold
                    if not self.is_candidate_valid(Y[j], self.X[p]):
                        continue
                    # if p not in cache:
                    #    cache[p] = int(np.median([self.approx_degree(query_buckets[j], p) for _ in range(30)]))
                    D = self.approx_degree(query_buckets[j], p)  # compute the degree in an approximate way
                    if random.randint(1, D) == D:  # output with probability 1/D
                        results[j].append(p)
                        break
        return results

    def exact_degree_query(self, Y, neighbors, runs=100):
        from bisect import bisect_right # Rightmost index to insert, so list remains sorted is -> method to choose the bucket according to the size
        query_buckets, query_size, _, bucket_sizes, prefix_sums = self.preprocess_query(Y)
        results = {i: [] for i in range(len(Y))}

        for j in range(len(Y)):
            cache = {}
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            for _ in range(neighbors * runs):
                if bucket_sizes[j] == 0:
                    results[j].append(-1)
                    continue
                while True:
                    i = random.randrange(bucket_sizes[j])
                    pos = bisect_right(prefix_sums[j], i)
                    table, bucket = query_buckets[j][pos] # choose the bucket with probability proportional to its size
                    p = random.choice(list(self.tables[table][bucket])) # choose a point randomly from the bucket
                    # discard not within distance threshold
                    if not self.is_candidate_valid(Y[j], self.X[p]):
                        continue
                    # if p not in cache:
                    #    cache[p] = int(np.median([self.approx_degree(query_buckets[j], p) for _ in range(30)]))
                    D = self.exact_degree(query_buckets[j], p)  # compute the degree in an approximate way
                    if random.randint(1, D) == D:  # output with probability 1/D
                        results[j].append(p)
                        break
        return results

    def rank_query_simulate(self, Y, neighbors, runs=100):
        import heapq
        n = len(self.X) # n initialized to the number of data points in the space
        m = len(Y) # initialized to the number of query points
        # ranks[i] is point with rank i
        # point_rank[j] is the rank of point j
        ranks = list(range(n)) # initialize the possible ranks
        point_rank = [0 for _ in range(n)] # initialize all the ranks to 0
        random.shuffle(ranks)

        for rank, point in enumerate(ranks):
            point_rank[point] = rank # assigns a rank randomly to each data point

        results = {i: [] for i in range(m)}

        query_buckets, query_size, query_results, _, _ = self.preprocess_query(Y)
        # iterate over each query point
        for j in range(m):
            elements = list((point_rank[point], point) for point in query_results[j]) # extract only for the query results the related rank and the point
            heapq.heapify(elements) # put the elements in increasing order of rank
            # MODIFICATA QUESTA RIGA PER FAR RESTITUIRE SOLO K VICINI
            for _ in range(neighbors * runs):
                while True:
                    rank, point = heapq.heappop(elements) # extract the top element from the heap
                    while rank != point_rank[point]: # check if the rank is the same as the one assigned
                        rank, point = heapq.heappop(elements) # continue popping elements until the rank is the same as the one assigned
                    if self.is_candidate_valid(Y[j], self.X[point]): # we are taking as candidate the point that has the same rank as the one popped
                        break

                results[j].append(point)
                # add a perturbation to the rank -> swap ranks of q and the retrieved point
                new_rank = random.randrange(rank, n) # generate a new rank randomly
                q = ranks[new_rank] # retrieve the point at the new sampled rank
                ranks[rank] = q # assign the new point to the old rank (for which we have already returned the point)
                ranks[new_rank] = point
                point_rank[q] = rank
                point_rank[point] = new_rank

                heapq.heappush(elements, (new_rank, point)) # add the point with its new rank to the heap (from the tail)
                if q in query_results[j]: # if the point q is in the results, we insert it into the heap memory
                    heapq.heappush(elements, (rank, q))
        return results

    def approx_degree(self, buckets, q):
        num = 0
        L = len(buckets)
        while num < L:
            num += 1
            table, bucket = buckets[random.randrange(0, L)]
            if q in self.tables[table].get(bucket, set()):
                break
        return L // num

    def exact_degree(self, buckets, q):
        cnt = 0
        for table, bucket in buckets:
            if q in self.tables[table].get(bucket, set()):
                cnt += 1
        return cnt

    def is_candidate_valid(self, q, x):
        # because it is implemented in the specific method according to the specific distance
        pass


class MinHash():
    def __init__(self):
        # choose four random 8 bit tables
        self.t1 = [random.randint(0, 2 ** 32 - 1) for _ in range(2 ** 8)]
        self.t2 = [random.randint(0, 2 ** 32 - 1) for _ in range(2 ** 8)]
        self.t3 = [random.randint(0, 2 ** 32 - 1) for _ in range(2 ** 8)]
        self.t4 = [random.randint(0, 2 ** 32 - 1) for _ in range(2 ** 8)]

    def _intern_hash(self, x):
        # the ^ is the binary XOR operator, 0xff is 255
        # x >> 24 is a shift to right by 24 bits (remove digits)
        # x << 24 is a shift to left by 24 bits (add digits)
        # They apply a XOR of the hash functions
        # Tabulation hashing is an efficient method to build hash functions
        # Added Integer cast
        x = int(x)
        return self.t1[(x >> 24) & 0xff] ^ self.t2[(x >> 16) & 0xff] ^ \
            self.t3[(x >> 8) & 0xff] ^ self.t4[x & 0xff]

    def _hash(self, X):
        '''
        Computes the minimum value among the hashes for a given input
        set of IDs, i.e., its minhash value
        :param X: input set of IDs
        :return: minhash value
        '''
        return min([self._intern_hash(x) for x in X])

    def get_element(self, L):
        h = self.hash(L)
        for x in L:
            if self.intern_hash(x) == h:
                return x


class OneBitMinHash(LSH):
    def __init__(self, k, L, r, validate=True):
        self.k = k  # k is the number of hash functions
        self.L = L  # L is the number of tables/bands
        self.r = r  # radius of the neighborhood
        self.hash_fcts = [[MinHash() for _ in range(k)] for _ in range(L)]
        self.validate = validate

    def _hash(self, X):
        self.hvs = []
        # iterate over each set in the input data
        for x in X:
            self.hvs.append([])
            for hash_fct in self.hash_fcts:
                h = 0
                for hf in hash_fct:
                    h += hf._hash(x) % 2
                    h *= 2
                self.hvs[-1].append(h)
        return self.hvs

    def _get_hash_value(self, arr, idx):
        return arr[idx]

    def is_candidate_valid(self, q, x):
        # validate the candidates, keeping only the ones having a Jaccard similarity over certain threshold r
        return not self.validate or 1 / (1 + pairwise_distances(q,x, metric="jaccard")) >= self.r

    def __str__(self):
        return f"OneBitMinHash(k={self.k}, L={self.L})"

    def __repr__(self):
        return f"k_{self.k}_L_{self.L}"


class E2LSH(LSH):
    def __init__(self, k, L, w, d, r, validate):
        # Overall, I will apply k*L hyperplanes whose values are sampled from the normal distribution
        self.A = np.random.normal(0.0, 1.0, (d, k * L))
        # sample a bias term from the uniform distribution
        self.b = np.random.uniform(0.0, w, (1, k * L))
        self.w = w  # Scaling factor after projection and sum of bias
        self.L = L  # Number of tables
        self.k = k  # Number of hash functions
        self.r = r  # radius of the neighborhood parameter
        self.validate = validate

    def _hash(self, X):
        # X = np.transpose(X)
        # project the data onto the hyperplanes contained in A
        hvs = np.matmul(X, self.A)
        # add the bias term (we could also remove it)
        hvs += self.b
        # divide by the weight term
        hvs /= self.w
        # we are adding a bias term b and a denominator w compared to the implementation of random projection
        return np.floor(hvs).astype(np.int32)
        # TODO attenzione che in random projection noi prendevamo 0 o 1 sulla base del segno del dot product
        # qui abbiamo sempre valori interi ma tra -4 e 4

    def _get_hash_value(self, arr, idx):
        return tuple(arr[idx * self.k: (idx + 1) * self.k])

    def is_candidate_valid(self, q, x):
        # print(distance.l2(q, x))
        return not self.validate or 1 / (1 + euclidean_distances(q,x)) >= self.r

    def __str__(self):
        return f"E2LSH(k={self.k}, L={self.L}, w={self.w})"

    def __repr__(self):
        return f"k_{self.k}_L_{self.L}_w_{self.w}"


class RandomProjection(LSH):
    def __init__(self, k, L, d, r, validate):
        # Overall, I will apply k*L hyperplanes whose values are sampled from the normal distribution
        self.A = np.random.normal(0.0, 1.0, (d, k * L))
        self.L = L  # Number of tables
        self.k = k  # Number of hash functions
        self.r = r  # radius of the neighborhood parameter
        self.validate = validate

    def _hash(self, X):
        # X = np.transpose(X)
        # project the data onto the hyperplanes contained in A
        hvs = np.matmul(X, self.A)
        # choose between 0 and 1 based on the sign of the result
        return (hvs > 0).astype(np.int32)
        # qui abbiamo sempre valori interi ma tra -4 e 4

    def _get_hash_value(self, arr, idx):
        return tuple(arr[idx * self.k: (idx + 1) * self.k])

    def is_candidate_valid(self, q, x):
        # print(distance.l2(q, x))
        return not self.validate or cosine_similarity(q, x) >= self.r

    def __str__(self):
        return f"RandomProjection(k={self.k}, L={self.L}, w={self.w})"

    def __repr__(self):
        return f"k_{self.k}_L_{self.L}_w_{self.w}"
