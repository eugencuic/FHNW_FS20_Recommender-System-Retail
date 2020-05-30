from recommender import get_recommendations
from recommender import products_recommendations_mobelbased
import numpy as np
import random

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
    # get list of masked items
    items_masked = list(masked_items[rating_index,:][0][:,1])

    # compare both list and return lenght of matches
    hits = len(set(rec).intersection(items_masked))

    return hits / len(items_masked)

def repeat_test(test,matrix,similarities,neighbours,masked_items,numberofruns,users,n):
    """Function to repeat a certain evaluation n times to get a more accurate result

    Arguments:
        test {string} -- accepted values are 'precision' or 'recall'
        matrix {csr_matrix, ndarray} -- user-item matrix to use for testing
        similarities {ndarray} -- Matrix with user-user similarities
        neighbours {ndarray} -- list with nearest neighbours
        masked_items {ndarray} -- Masked items to check for
        numberofruns {int} -- number of testruns
        users {list} -- list of testusers to be used
        n {int} -- Number of item recommendations
    """    
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


def evaluate_ubcf(matrix,similarities,neighbours,masked_items,numberofruns,users,n):
    """Function to repeat both precision and recall test at the same time to get faster results

    Arguments:
        matrix {csr_matrix, ndarray} -- user-item matrix to use for testing
        similarities {ndarray} -- Matrix with user-user similarities
        neighbours {ndarray} -- list with nearest neighbours
        masked_items {ndarray} -- Masked items to check for
        numberofruns {int} -- number of testruns
        users {list} -- list of testusers to be used
        n {int} -- Number of item recommendations
    """    
    res_precision = 0
    res_recall = 0
    
    for i in range(numberofruns):

        rec = get_recommendations(matrix,similarities,neighbours,i,n)
        rec = list(rec['item'])

        # get the items which were masked for the user
        rating_index = np.where(masked_items[:,0] == i)
    
        # get list
        items_masked = list(masked_items[rating_index,:][0][:,1])

        # compare both list and return lenght of matches
        hits = len(set(rec).intersection(items_masked))

        # calculate precision
        temp_precision = hits / n
        #calculate recall
        temp_recall = hits / len(items_masked)

        # add to final result
        res_precision += temp_precision
        res_recall += temp_recall

    precision = res_precision / numberofruns
    recall = res_recall / numberofruns

    return precision, recall


################# NEU FÜR MEMORY BASED ####################

def evaluation_memory_based_user(matrix,similarities,neighbours,masked_items,user,n):
    """Generate Recall and Precision@N for a specific User

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
    # get list of masked items
    items_masked = list(masked_items[rating_index,:][0][:,1])

    # compare both list and return lenght of matches
    hits = len(set(rec).intersection(items_masked))

    recall = hits / len(items_masked)
    precision = hits / n

    return  precision, recall

def evaluation_memory_based_model(user_list, number_of_random_user, matrix,similarities,neighbours,masked_items, n):

    random_users = random.sample(user_list, number_of_random_user)
    precision = 0
    recall = 0

    for user in random_users:
        precision_value, recall_value = evaluation_memory_based_user(matrix,similarities,neighbours,masked_items,user,n)

        precision += precision_value
        recall += recall_value
    average_precision = precision / number_of_random_user
    average_recall = recall / number_of_random_user

    return average_precision, average_recall

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

    return  precision, recall

def evaluation_model_based_model(user_list, number_of_random_user, all_user_predicted_ratings, index_masked_count, train_set_count, number_of_recommendations):

    random_users = random.sample(user_list, number_of_random_user)
    precision = 0
    recall = 0

    for user in random_users:
        precision_value, recall_value = evaluation_model_based_user(user, all_user_predicted_ratings, index_masked_count, train_set_count, number_of_recommendations)
        precision += precision_value
        recall += recall_value
    average_precision = precision / number_of_random_user
    average_recall = recall / number_of_random_user

    return average_precision, average_recall