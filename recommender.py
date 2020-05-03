import random
from scipy.linalg import svd

def split_test_train(data, test_threshold):
    # Create list of all User_IDs
    user_ids = data.user_id.unique()
    # Select random 10% of the Users
    random_users = random.choices(user_ids, k=round(len(user_ids)*test_threshold))

    #Train Set
    data_train = data[~(data.user_id.isin(random_users))]
    # Test Set
    data_test = data[(data.user_id.isin(random_users))]

    print('Train Set is saved as data_train, Test Set is saved as data_test')

    return data_train, data_test

def matrix_prep_pivot(train_data, test_data):

    pivot_data_train = train_data[['user_id','product_id']]
    pivot_data_test = test_data[['user_id','product_id']]
    
    return pivot_data_train, pivot_data_test

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


