from recommender_ubcf import get_recommendations
from recommender_ubcf import products_recommendations_mobelbased
import numpy as np

def precision(matrix,similarities,neighbours,masked_items,user,n):
    """Generate Precision@N for a specific User

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User Similarity Matrix
        neighbours {ndarray} -- List of N Neighbours for each User in User-Item Matrix
        masked_items {ndarray} -- List of Arrays with masked User-Items
        user {int} -- Index Number of User
        n {int} -- Number of Recommendations to be returned / measured by

    Returns:
        string -- Result of Precision@N Value for specific User
    """    
    # calculate recommendations
    rec = get_recommendations(matrix,similarities,neighbours,user,n)
    rec = list(rec['item'])

    # get the items which were masked for the user
    rating_index = np.where(masked_items[:,0] == user)
    # get list
    items_masked = list(masked_items[rating_index,:][0][:,1])

    # compare both list and return lenght of matches
    hits = len(set(rec).intersection(items_masked))

    return hits / n


def recall(matrix,similarities,neighbours,masked_items,user,n):
    """Generate Recall@N for a specific User

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User Similarity Matrix
        neighbours {ndarray} -- List of N Neighbours for each User in User-Item Matrix
        masked_items {ndarray} -- List of Arrays with masked User-Items
        user {int} -- Index Number of User
        n {int} -- Number of Recommendations to be returned / measured by

    Returns:
        string -- Result of Recall@N Value for specific User
    """    
    # calculate recommendations
    rec = get_recommendations(matrix,similarities,neighbours,user,n)
    rec = list(rec['item'])

    # get the items which were masked for the user
    rating_index = np.where(masked_items[:,0] == user)
    # get list
    items_masked = list(masked_items[rating_index,:][0][:,1])

    # compare both list and return lenght of matches
    hits = len(set(rec).intersection(items_masked[:n]))

    return hits / len(items_masked)

def repeat_test(test,matrix,similarities,neighbours,masked_items,numberofruns,users,n):

    if test == 'precision':
        res = 0
        print('precision@{0}:'.format(n))
        for i in range(numberofruns):
            res += precision(matrix,similarities,neighbours,masked_items,i,n)
        res = res / numberofruns
        print('Average over {0} Users: {1}'.format(numberofruns,res))

    if test == 'recall':
        res = 0
        print('recall@{0}:'.format(n))
        for i in range(numberofruns):
            res += recall(matrix,similarities,neighbours,masked_items,i,n)
        res = res / numberofruns
        print('Average over {0} Users: {1}'.format(numberofruns,res))

######################### MODEL BASED APPROACH #########################

def evaluation_model_based_user(user_index, predicted_rating, masked_items, train_set_count, n_of_recommendations):
 
    # calculate recommendations
    rec = products_recommendations_mobelbased(user_index, predicted_rating, train_set_count, n_of_recommendations)

    # get the items which were masked for the user
    rating_index = np.where(masked_items[:,0] == user_index)
    # get list
    items_masked = list(masked_items[rating_index,:][0][:,1])

    # turn list into array
    items_masked = np.array(items_masked)

    # compare both list and return lenght of matches

    hits = len(set(rec).intersection(items_masked))

    recall = hits / len(items_masked)
    precision = hits / n_of_recommendations

    return recall, precision
