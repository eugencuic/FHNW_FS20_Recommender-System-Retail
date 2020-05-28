import random
from scipy.linalg import svd
import numpy as np
import sys
from scipy.sparse import csr_matrix

def create_user_item_matrix(data,type='unary'):
    """Function to create User-Item Matrix from raw (transaction) data. 

    Arguments:
        data {dataframe} -- Raw data, must include a user_id and product_id attribute
        type {string} -- Type of ratings to be generated, available options are unary or count
        max_count{integer} -- Set max count number of puchases per product

    Returns:
        csr_matrix -- User-Item Matrix in sparse format
    """     
    if type == 'unary':

        # for unary rating drop duplicates
        data = data.drop_duplicates()

        # create sparse matrix
        matrix = csr_matrix((data['rating'], (data['user_id'],data['product_id'])))

        # rows and cols with empty values will be dropped (doesnt make any difference in size for sparse matrix, but if later converted to dense, it saves space)
        # get all non empty rows and cols
        rows, cols = matrix.nonzero()
        unique_rows = np.unique(rows)
        unique_cols = np.unique(cols)

        # select only rows and cols with values
        matrix = matrix[unique_rows]
        matrix = matrix[:,unique_cols]

        return matrix

    if type == 'count':
    
        # group by user and product and count the occurence
        matrix_dense =  data.groupby(['user_id', 'product_id'])['rating'].sum().unstack(fill_value=0.0)
        users = matrix_dense.index.tolist()
        products = matrix_dense.columns.values.tolist()

        # Get max values of each row
        max_values = matrix_dense.max(axis=1)

        # transform to numpy array for matrix multiplication
        max_values = max_values.to_numpy()
        matrix_dense = matrix_dense.to_numpy()
        
        # Normalize values between 0 and 1 by original matrix multiplied by max value of each row
        matrix_dense = (matrix_dense.T / max_values).T

        # transform matrix to sparse matrix to safe space 
        matrix_sparse = csr_matrix(matrix_dense)

        return matrix_sparse, products, users



def calc_sparsity (data):
    """This function is used to test how much of the original data has empty values in the matrix.

    Arguments:
        data {csr_matrix} -- User-Item Matrix in sparse format
    
    Returns:
        Statement -- A Statment about the % of the sparsity in the original Matrix
    """    
    matrix_size = data.shape[0]*data.shape[1] # Number of possible interactions in the matrix
    num_purchases = len(data.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_purchases/matrix_size))
    print('{:.2f} % of the user interaction matrix is sparse'.format(sparsity,2))

def split_test_train(matrix, test_threshold):
    """Splitting Users in User-Item Matrix into Test and Train Set

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        test_threshold {float} -- Split for Test and Train set. Number between 0-1

    Returns:
        ndarray -- User-Item Matrix Train Set
        ndarray -- User-Item Matrix Test Set
    """    
    # Create list of random User Index Numbers for Users in Test Set
    user_idx_test = np.random.randint(matrix.shape[0], size=int(matrix.shape[0]*test_threshold))
    
    # Create list of all other Others for the Train Set
    user_idx_train = np.delete(np.arange(matrix.shape[0]),user_idx_test)

    # Filter origin Matrix for Test Users
    matrix_test = matrix[user_idx_test,:]

    # Filter origin Matrix for Train Users
    matrix_train = matrix[user_idx_train,:]

    return matrix_train, matrix_test


def mask_test_train(data, split):
    """Masking a percentage of all Ratings in a given Matrix
       with help from: https://jessesw.com/Rec-System/

    Arguments:
        data {ndarray} -- Full User-Item Matrix where Ratings will be masked
        split {float} -- Number between 0-1

    Returns:
        ndarray -- Matrix for Training Set (Ratings masked)
        ndarray -- Matrix for Test Set (All Ratings)
        list -- List of masked Users
        list -- List with User-Item Tuples of masked Ratings
    """    
    # create a copy of the full data for reduction
    training_set = data.copy()

    # find index of values which are not empty
    nonzero_inds = training_set.nonzero()

    # create list of index pairs
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))

    # set to zero for reproductability
    random.seed(0)

    # calculate the number of samples to be removed in training set
    num_samples = int(np.ceil(split*len(nonzero_pairs)))

    # get random samples
    samples = random.sample(nonzero_pairs, num_samples)

    # remove selected samples in training set
    user_inds = [index[0] for index in samples]
    item_inds = [index[1] for index in samples]
    training_set[user_inds, item_inds] = 0 

    return training_set, data, list(set(user_inds)), np.array(samples)

######################### DEPRECIATED #########################

# DEPRECIATED BECAUSE OF PERFORMANCE ISSUES
def unary_matrix(pivot_data_train, pivot_data_test):

    unary_func = lambda x: 1

    # Unary Matrix for Trainset
    matrix_train = pivot_data_train.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=unary_func,fill_value=0)
    matrix_train.reset_index(inplace=True)
    matrix_train = matrix_train.set_index('user_id')

    # Unary Matrix for Testset
    matrix_test = pivot_data_test.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=unary_func,fill_value=0)
    matrix_test.reset_index(inplace=True)
    matrix_test = matrix_test.set_index('user_id')

    return matrix_train, matrix_test

def count_matrix(pivot_data_train, pivot_data_test):

    # Count Matrix for Trainset
    matrix_train = pivot_data_train.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=len,fill_value=0)
    matrix_train.reset_index(inplace=True)
    matrix_train = matrix_train.set_index('user_id')

    # Count Matrix for Testset
    matrix_test = pivot_data_test.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=len,fill_value=0)
    matrix_test.reset_index(inplace=True)
    matrix_test = matrix_test.set_index('user_id')

    return matrix_train, matrix_test

def mask_test_train_count(data, rating_threshold):
    """Masking a percentage of all Ratings in a given Matrix
       with help from: https://jessesw.com/Rec-System/

    Arguments:
        data {ndarray} -- Full User-Item Matrix where Ratings will be masked
        split {float} -- Number between 0-1

    Returns:
        ndarray -- Matrix for Training Set (Ratings masked)
        ndarray -- Matrix for Test Set (All Ratings)
        list -- List of masked Users
        list -- List with User-Item Tuples of masked Ratings
    """    
    # create a copy of the full data for reduction
    training_set = data.copy()

    # create max split
    #max_split = int(split*(training_set.nnz))

    # find index of values which are not empty and over threshold
    rating_inds = np.nonzero(training_set > rating_threshold)
    
    # create list of index pairs
    rating_pairs = list(zip(rating_inds[0], rating_inds[1]))

    # set to zero for reproductability
    #random.seed(0)

    # Split ration, based on threshold
    #thres_max = len(nonzero_pairs)

    #if thres_max > max_split:
    #    split_ratio = max_split / thres_max
    #else:
    #    sys.exit('Your threshold for rating is too high, please recalculate and lower down the threshold')

    # calculate the number of samples to be removed in training set
    #num_samples = int(np.ceil(split_ratio*len(nonzero_pairs)))

    # get random samples
    #samples = random.sample(nonzero_pairs, num_samples)
    #samples = nonzero_pairs
    # remove selected samples in training set
    user_inds = [index[0] for index in rating_pairs]
    item_inds = [index[1] for index in rating_pairs]
    training_set[user_inds, item_inds] = 0 

    return training_set, data, list(set(user_inds)), np.array(rating_pairs)