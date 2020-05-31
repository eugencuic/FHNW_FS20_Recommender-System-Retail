from recommender import get_recommendations
from recommender import products_recommendations_modelbased
import numpy as np
import random


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
    """Evaluating precision and recall for a given amount of users and a given amount of recommendations

    Arguments:
        user_list {array} -- Array of all Users
        number_of_random_user {int} -- Number of users to take from user list
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User Similarity Matrix
        neighbours {ndarray} -- List of N Neighbours for each User in User-Item Matrix
        masked_items {ndarray} -- List of Arrays with masked User-Items
        n {int} -- Number of Recommendations to be returned / measured by

    Returns:
        average_precision [float] -- Average Precision of the amount of users and recommendations
        average_recall [float] -- Average Recall of the amount of users and recommendations
    """
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

def evaluation_model_based_user(user_index, predicted_ratings, masked_items, matrix, n_of_recommendations):
    """Generate Recall and Precision@N for a specific User

    Arguments:
        user_index {array} -- Array of all Users
        predicted_ratings {ndarray} -- Array of all predicted rating for each user and item combination
        masked_items {[type]} -- [description]
        masked_items {ndarray} -- List of Arrays with masked User-Items
        n_of_recommendations {int} -- Number of Recommendations to be returned / measured by

    Returns:
        precision [float] -- Precision of the amount recommendations
        recall [float] -- Recall of the amount of recommendations
    """ 
    # calculate recommendations
    rec = products_recommendations_modelbased(user_index, predicted_ratings, matrix, n_of_recommendations)

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
    """Evaluating precision and recall for a given amount of users and a given amount of recommendations"

    Arguments:
        user_list {array} -- Array of all Users
        number_of_random_user {int} -- Number of users to take from user list
        all_user_predicted_ratings ndarray} -- Array of all predicted rating for each user and item combination
        index_masked_count {ndarray} -- List of Arrays with masked User-Items
        train_set_count {csr_matrix} -- User-Item Matrix
        number_of_recommendations {int} -- Number of Recommendations to be returned / measured by

    Returns:
        average_precision [float] -- Average Precision of the amount of users and recommendations
        average_recall [float] -- Average Recall of the amount of users and recommendations
    """
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