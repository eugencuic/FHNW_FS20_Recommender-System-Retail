import random
from scipy.linalg import svd
import numpy as np
from scipy.sparse import csr_matrix

def unary_matrix(data): 
    """Calculate User-Item Matrix with unary Ratings. This function is very slow for large datasets

    Arguments:
        data {DataFrame} -- DataFrame with Transaction variables. Attributes user_id and product_id need to be present.

    Returns:
        csr_matrix -- User-Item Matrix sparse format
    """    
    # Remove unnecessary rows
    data = data[['user_id','product_id']]
    # Function for Pivot Values
    unary_func = lambda x: 1
    # Create Pivot Table
    matrix = data.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=unary_func,fill_value=0)
    # Save matrix as sparse Matrix for Memory reduction
    matrix = csr_matrix(matrix.values)

    return matrix

def count_matrix(data):
    """Calculate User-Item Matrix with sum of purchases as Ratings. This function is very slow for large datasets

    Arguments:
        data {DataFrame} -- DataFrame with Transaction variables. Attributes user_id and product_id need to be present.

    Returns:
        ndarray -- User-Item Matrix sparse format
    """   
    data = data[['user_id','product_id']]

    matrix = data.pivot_table(index='user_id',columns='product_id',values='product_id',aggfunc=len,fill_value=0)
    matrix.reset_index(inplace=True)
    matrix = matrix.set_index('user_id')

    matrix = matrix.to_numpy()

    return matrix

def create_user_item_matrix(data,type='unary'):
    """Function to create User-Item Matrix from raw (transaction) data. 

    Arguments:
        data {dataframe} -- Raw data, must include a user_id and product_id attribute
        type {string} -- Type of ratings to be generated, available options are unary or count

    Returns:
        csr_matrix -- User-Item Matrix in sparse format
    """    
    data = data[['user_id','product_id']]
    
    # add rating infos
    data['rating'] = 1

    if type == 'unary':
        # for unary rating drop duplicates
        data = data.drop_duplicates()

    #if type == 'count':
        # there is nothing else done at the moment, maybe add normalization or someting?

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

