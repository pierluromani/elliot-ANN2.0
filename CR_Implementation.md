# CR (Candidate Ratio) Implementation in Elliot

This document provides a comprehensive overview of the **Candidate Ratio (CR)** metric implemented by the students Andrea Natile and Pierluigi Romani within the Elliot recommendation framework.

## 1. What is Candidate Ratio (CR)?

Candidate Ratio (CR) is an efficiency metric utilized during the similarity computation phase of Neighborhood-based and Approximate Nearest Neighbor (ANN) recommendation models. It represents the proportion of candidate pairs used for calculating the similarity scores out of all possible candidate pairs in the network. The more is approximate the search, the lower the number of candidate pairs where their similarity is computed, the lower the CR.

Mathematically, it is defined as:
```
CR = Number of Computed/Retained Candidate Pairs / Total Possible Pair Combinations
```

For **Item-based** algorithms:
`CR = n_candidates_pair / (n_items * n_items)`

For **User-based** algorithms:
`CR = n_candidates_pair / (n_users * n_users)`

* `n_candidates_pair`: The number of non-zero entries in the similarity matrix before selecting the top-k neighbors.
* `n_items`/`n_users`: Total number of items/users in the dataset.

## 2. Theoretical Background: Exact vs Approximate Methods

The primary rationale behind calculating CR is to evaluate the theoretical and practical computational savings provided by Approximate Nearest Neighbor (ANN) approaches (like LSH, FAISS, and ANNOY) compared to exact Nearest Neighbor approaches.

- **Exact Nearest Neighbor (e.g., UserKNN, ItemKNN):**
  These methods compute the exact distance between all possible pairs to find the true closest neighbors.
  - **CR = 1.0**: Indicates 100% of candidate pairs were evaluated.
 
- **Approximate Nearest Neighbor (e.g., ANNOY, FAISS, LSH):**
  These models limit the search space using indexing methods like random projection trees or hash buckets. Only pairs that fall into the same buckets or tree leaves are evaluated.
  - **CR < 1.0**: Reflects a reduced fraction of exact distance computations. Smaller values indicate higher computational efficiency and tighter indexing.

## 3. Implementation in Elliot

CR is calculated dynamically within the `initialize()` pipeline of similarity objects only if the `csv_path` parameter is provided in the config file. This parameter is used to specify the path of the csv file where the configuration of the model and the CR values are stored for each run.

### Generic Workflow

1. A zeroed similarity matrix is initialized: `self._similarity_matrix = np.zeros(...)`
2. The index/search approach is processed in `self.process_similarity()`, filling out non-zero similarities just for retrieved pairs.
3. The number of candidate pairs is counted via `np.count_nonzero(self._similarity_matrix)`.
4. The CR is then calculated and logged into a metadata CSV file using the custom logging utility `add_CR_instance`.

### Example Snippet (ItemAnnoy / UserAnnoy)

```python
# Calculate and log CR
if self._csv_path:
    # Get total count (users or items)
    n_total = self._data.num_items # or self._data.num_users
    
    # Extract evaluated candidate pairs
    n_candidates_pair = np.count_nonzero(self._similarity_matrix)
    
    # Component Return calculation
    CR = n_candidates_pair / (n_total * n_total)
    print(f"CR: {CR}")
    
    # Meta data definition logging framework
    meta_data = {
        "model": "ItemAnnoy",
        "neighbors": self._num_neighbors,
        "similarity": self._similarity,
        "n_trees": self._n_trees,
        "search_k": self._search_k,
        "CR": CR
    }
    add_CR_instance(self._csv_path, meta_data)
```

### Exact KNN Workarounds

For strictly non-approximate methods (e.g. standard `UserKNN` and `ItemKNN`), Elliot doesn't provide the similrity metrics, so it is not possible to compute the CR value analiticaly. In this case, from the theory the CR value is set to 1.0 to indicate that the model is using the exact method.

```python
if self._csv_path:
    CR = 1.0
    print(f"CR: {CR}")
    
    meta_data = {
        "model": "UserKNN",
        "neighbors": self._num_neighbors,
        "similarity": self._similarity,
        "implicit": self._implicit,
        "CR": CR
    }
    add_CR_instance(self._csv_path, meta_data)
```

## 4. Models supporting CR Tracking

The custom CR tracking mechanism has been horizontally pushed across multiple local components including (but not limited to):

**Exact KNN:**
- UserKNN
- ItemKNN
- AttributeUserKNN
- AttributeItemKNN

**Approximate ANN/LSH:**
- ItemAnnoy (`item_annoy_similarity.py`)
- UserAnnoy (`user_annoy_similarity.py`)
- ItemANNFaissLSH (`item_ann_faiss_lsh_similarity.py`)
- UserANNFaissLSH (`user_ann_lsh_similarity.py`)
- ItemFairANN (`item_fair_ann_similarity.py`)
- UserFairANN (`user_fair_ann_similarity.py`)
- LSH (Vanilla `user_ann_lsh`/`item_ann_lsh`)

## 5. Integration CR with the other metrics 
CR metrics tracked over test epochs are typically evaluated inside data integration scripts of the repository **Result Analyzer** that integrate the other metrics for each model with the CR value, creating a dataset that can be used to plot the metrics and evaluate the efficiency of the models.
