# we create 2 and 3 groups for both items and users, the items are divided according to their popularity, in the case
# of the 2 groups we have 20 most popular ones and 80 from the long tail, for the 3 groups we have 20 most popular
# and 60 from the mid-tail, finally 20 from the long tail the users are divided according to their tolerance to the
# popular items, in the case of the 2 groups we have 20% of the users interested in best-seller items, and the other
# ones interested in the long tail, for the 3 groups we have 20% of the users interested in best-seller items,
# 60% interested in the mid-tail and 20% in the long-tail

# import the required libraries
import pandas as pd
import numpy as np
import argparse
import json

# read the dataset
parser = argparse.ArgumentParser(description="Run split user/item groups depending on quartiles.")
parser.add_argument('--dataset', type=str, default='movielens_100k', choices=['movielens_100k', 'movielens_1m', 'amazon_books', 'lastfm_1k', 'yelp'])
args = parser.parse_args()
# take the dataset from the input arguments
dataset = args.dataset
# read the dataset from the corresponding csv file
df = pd.read_csv(f'./data/{dataset}/filtered_data/0/train.tsv', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
df = df.drop('timestamp', axis=1)

print("data successfully loaded")
# count the transactions in which any item occurs
item_count = df['item_id'].value_counts()

print("Sorting the values...")
item_frequencies = pd.DataFrame(data=np.concatenate([item_count.index.values.reshape(-1,1), item_count.values.reshape(-1,1)], axis=1)).sort_values(by=1, ascending=True)
item_frequencies.columns = ['ItemID', 'Frequency']

print("Computing the quantiles for the distributions...")
item_80 = np.quantile(item_frequencies['Frequency'], 0.80)
item_20 = np.quantile(item_frequencies['Frequency'], 0.20) # needed to split into 3 groups

# compute the bins and labels for the 2 groups
item_bins = [0, item_80, np.inf]
item_bins = list(set(item_bins))
item_bins.sort()
item_labels = [i for i in range(len(item_bins)-1)]

print("Assigning the labels to each bin")
item_frequencies['group'] = pd.cut(item_frequencies['Frequency'], bins=item_bins, labels=item_labels, right=True)

# save the tsv file with the grouping of ID-group
item_frequencies[['ItemID', 'group']].to_csv(f'data/{dataset}/items_popularity_2.tsv', sep='\t', header=None, index=None)

# create a reverse mapping groupID: [list of users/items belonging to that group]
group_to_items = item_frequencies.groupby('group')['ItemID'].apply(list).to_dict()

with open(f'data/{dataset}/group_popularity_2_to_items.json', 'w') as file:
    json.dump(group_to_items, file, indent=4)

# group labelled as 1 contains the most popularitems (20%), the group labelled as 0 contains items and
# items from the long-tail (80%)

# do the same but for 3 groups
# compute the bins and labels for the 2 groups
item_bins_3 = [0, item_20, item_80, np.inf]
item_bins_3 = list(set(item_bins_3))
item_bins_3.sort()
item_labels_3 = [i for i in range(len(item_bins_3)-1)]

print("Assigning the labels to each bin")
item_frequencies['group_3'] = pd.cut(item_frequencies['Frequency'], bins=item_bins_3, labels=item_labels_3, right=True)

# save the tsv file with the grouping of ID-group
item_frequencies[['ItemID', 'group_3']].to_csv(f'data/{dataset}/items_popularity_3.tsv', sep='\t', header=None, index=None)

# create a reverse mapping groupID: [list of users/items belonging to that group]
group_to_items_3 = item_frequencies.groupby('group_3')['ItemID'].apply(list).to_dict()

with open(f'data/{dataset}/group_popularity_3_to_items.json', 'w') as file:
    json.dump(group_to_items_3, file, indent=4)

# based on the user profiles, we will split the users into 2 groups, the first group will contain the users that
# interact with the most popular items, while the second group will contain the users that interact with the long-tail

# based on the user profiles we will split the users into 3 groups, the first group will contain the users that interact
# mostly with the most popular items (at least 80% of the user profile), the second group will contain the users that
# interact with the long-tail (at most 20% of the popular) and the third one
# the remaining users that interact with the mid-tail items

# starting by the division of items into 2 groups, we compute the ratio (pop. items/tot items in the profile)
# of each user profile in the dataset

# we create a dictionary ItemID: popularity group (0 for long-tail and 1 for popular items)
item_to_group = item_frequencies.set_index('ItemID')['group'].to_dict()

# now we need to compute the user profiles composed as a list of items, and then compute the ratio of popular items

# compute the user profiles
user_profiles = df.groupby('user_id')['item_id'].apply(list).to_dict()

# compute the ratio of popular items in the user profile
user_to_group_3 = {}
user_to_group_2 = {}

for user, profile in user_profiles.items():
    # compute the ratio of popular items
    pop_items = 0
    for item in profile:
        if item in item_to_group:
            pop_items += item_to_group[item]  #
    ratio = pop_items/len(profile)
    if ratio >= 0.8:
        user_to_group_3[user] = 2  # blockbuster-focused users [0.8-1]
        user_to_group_2[user] = 1
    elif ratio >= 0.2:
        user_to_group_3[user] = 1  # diverse users [0.2-0.8]
        user_to_group_2[user] = 0
    else:
        user_to_group_3[user] = 0  # niche-focused users [0-0.2]
        user_to_group_2[user] = 0

# in the case of 2 groups of users, we divide between blockbuster-focused and all the others
# save the results into a tsv file UserID, group2 and group3

user_tolerances_2 = pd.DataFrame(data=np.concatenate([np.array(list(user_to_group_2.keys())).reshape(-1,1), np.array(list(user_to_group_2.values())).reshape(-1,1)], axis=1))
user_tolerances_2.columns = ['UserID', 'group']

# save the grouping of ID-group
user_tolerances_2.to_csv(f'data/{dataset}/users_tolerance_2.tsv', sep='\t', header=None, index=None)

# do the same with the grouping of 3 groups
user_tolerances_3 = pd.DataFrame(data=np.concatenate([np.array(list(user_to_group_3.keys())).reshape(-1,1), np.array(list(user_to_group_3.values())).reshape(-1,1)], axis=1))
user_tolerances_3.columns = ['UserID', 'group']

# save the grouping of ID-group
user_tolerances_3.to_csv(f'data/{dataset}/users_tolerance_3.tsv', sep='\t', header=None, index=None)

# create a reverse mapping groupID: [list of users/items belonging to that group]
group_to_users_2 = user_tolerances_2.groupby('group')['UserID'].apply(list).to_dict()
group_to_users_3 = user_tolerances_3.groupby('group')['UserID'].apply(list).to_dict()

# save the reverse mapping groupID -> list of User/item IDs
with open(f'data/{dataset}/group_tolerance_2_to_users.json', 'w') as file:
    json.dump(group_to_users_2, file, indent=4)

with open(f'data/{dataset}/group_tolerance_3_to_users.json', 'w') as file:
    json.dump(group_to_users_3, file, indent=4)

# create a column for when the users/items are not split
user_tolerances_2['no_group'] = np.full(shape=user_tolerances_2.shape[0], fill_value=0)
item_frequencies['no_group'] = np.full(shape=item_frequencies.shape[0], fill_value=0)

# save the dataset without grouping
user_tolerances_2[["UserID", "no_group"]].to_csv(f"data/{dataset}/users_no_groups.tsv", sep="\t", header=None, index=None)
item_frequencies[["ItemID", "no_group"]].to_csv(f'data/{dataset}/items_no_groups.tsv', sep='\t', header=None, index=None)