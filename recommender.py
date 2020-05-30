import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse.linalg import svds
from math import sqrt

######################### MEMORY BASED APPROACH #########################


def user_sim(data):
    """Calculate User Similiarity for a given Matrix using sklearn function

    Arguments:
        data {ndarray} -- User-Item Matrix

    Returns:
        ndarray -- User-User Matrix with Cosine similarity
    """    
    similarities = cosine_similarity(data)
    return similarities


def user_sim_v1(data):
    """Very slow version of a cosine implementation
       This was a first try. Since every value in the resulting User-User Matrix is calculated separately this takes very long even on small matrices.
       The time to calculate this grows exponentially with the number of users.

    Arguments:
        data {ndarray} -- User-Item Matrix

    Returns:
        ndarray -- User-User Matrix with cosine similarity
    """    
    matrix = np.empty([0,data.shape[0]])
    #calculate norm of all vectors
    norm = []
    for n in range(data.shape[0]):
        v = np.linalg.norm(data[n].todense())
        norm.append(float(v))

    for j in range(data.shape[0]):
        row = []
        for i in range(data.shape[0]):
            sim = np.dot(data[j],data[i].T) / (norm[j] * norm[i])
            row.append(float(sim.todense()))
        matrix = np.vstack([matrix,row])
    
    similarities = matrix
    # Alternatively there is a function from scipy which does exactly that
    #similarities = cosine_similarity(data)

    return similarities


def user_sim_v2(data):
    """Improved version of user_sim_v1 but still very slow. Since only "new" User-User combinations are calculated this takes only half the time of v1.
       But this is still not a useful way even for relatively small matrices.
       source: https://stackoverflow.com/questions/42044770/efficiently-calculate-cosine-similarity-using-scikit-learn

    Arguments:
        data {ndarray} -- User-Item Matrix

    Returns:
        ndarray -- User-User Matrix with cosine similarity
    """    

    # getting length of user axis
    n = data.shape[0]

    # create a emtpy matrix
    matrix = np.array(n**2*[0],np.float).reshape(n,n)
    # function for cosine
    def cosine(a,b):
        dp = np.dot(data[a],data[b].T)
        norm_a = np.linalg.norm(data[a].todense())
        norm_b = np.linalg.norm(data[b].todense())
        return  float((dp / (norm_a * norm_b)).todense())

    for i in range(n):
        # the similarity to itself is always 1
        matrix[i,i] = 1
        
        for j in range(i+1,n):

            matrix[i,j] = cosine(i,j)
            matrix[j,i] = matrix[i,j]
        
    return matrix

def user_sim_v3(data):
    """Calculate Cosine Similarity for a Matrix
       This is a fast way for calculating cosine similarity on a matrix
       Calculating the square of the value is commented out, since in unary rating this does not change the result
       with help from: https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    
    Arguments:
        data {csr_matrix/ndarray} -- User-Item Matrix with Ratings

    Returns:
        ndarray -- User-User Similarity Matrix
    """    

    # Multiply is not needed with unary ratings
    #squared = out.multiply(out)

    # Calculate the sum of all User vectors
    row_sums = np.array(np.sqrt(data.sum(axis=1)))[:, 0]

    # Divide each value with its vector sum
    data = data / row_sums[:,None]

    # Multiply the matrix with itself
    data = np.dot(data.astype(np.float32),data.astype(np.float32).T)
    
    # Convert result to numpy array
    similarity_matrix = np.array(data)
    
    return similarity_matrix

def user_hood(similarities,k):
    """Calculate k most similar Users to all given Users
       This is probably a slow function due to the for loop and the argsort function, but for the lack of a better idea it is what it is

    Arguments:
        similarities {ndarray} -- User-User Matrix with similarity values for each User relation
        k {int} -- Number of similar Users to return

    Returns:
        ndarray -- Array with k most similar Users for each User in User-User Matrix
    """    
    neighbours = []
    # the user has the best rating with himself, therefor add 1 to get k real users
    k = k+1
    for x in similarities:
        newrow = x.argsort()[-k:]
        neighbours.append(newrow)
    neighbours = np.array(neighbours)
    return neighbours


def predict(matrix,similarities,neighbours,u,i):
    """Calculate predicted rating for given User and Item

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User-User Similarity Matrix
        neighbours {ndarray} -- K nearest Neighbours of all Users
        u {int} -- User Index number to calculate the rating for
        i {int} -- Item Index number to calculate the rating for

    Returns:
        float -- predicted rating
    """    
    numerator = float(0)
    denumerator = float(0)
    prediction = float(0)

    # This is the formula from Ekstrand, Riedl & Konstan (2010, 91) (2.6) but with removed averged Rating (since unary Ratings are used) 
    for nb in neighbours[u]:
        numerator = numerator + float(similarities[u,nb]) * int(matrix[nb,i])
        denumerator = denumerator + float(similarities[u,nb])
    prediction = float(numerator / denumerator)

    return prediction
    

def products_to_recommend(matrix,similarities,neighbours,u):
    """Calculate the Items which will be recommendend for a User

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User-User Similarity Matrix
        neighbours {ndarray} -- K nearest Neighbours of all Users
        u {int} -- User Index number to calculate the Items to recommend for

    Returns:
        list -- List of items to recommend the given user
    """    
    u_prod = None
    n_prod = None
    n_prod_sum = np.array([])
    n_prod_check = None

    # Calculate all Items the User already has
    u_prod = list(matrix[u,:].nonzero()[1])

    # Calculate all Items of the Users neighbours
    for nb in neighbours[u]:
        n_prod = (matrix[nb,:].nonzero())[1]
        n_prod_sum = np.append(n_prod_sum,n_prod)
    n_prod_sum = list(set(n_prod_sum))

    # Remove Items the User has already rated
    # map: convert values to int
    # set: remove duplicates
    n_prod_check = list(map(int, (set(n_prod_sum) - set(u_prod))))

    return n_prod_check


# Funktion für die Recommendations
def get_recommendations(matrix,similarities,neighbours,u,t):
    """Calculate Top N recommended Items and there rating

    Arguments:
        matrix {ndarray} -- User-Item Matrix
        similarities {ndarray} -- User-User Similarity Matrix
        neighbours {ndarray} -- K nearest Neighbours of all Users
        u {int} -- User Index number to calculate the Items to recommend for
        t {int} -- Number of recommendations to return

    Returns:
        Dataframe -- Recommended Items and calculated Ratings of these Items
    """    
    
    products = None
    r = pd.DataFrame(columns=['item','rating'])
    
    products = products_to_recommend(matrix,similarities,neighbours,u)
    
    # Calcualte all ratings for the items
    for item in products:
        new_row = {'item':item,'rating':predict(matrix,similarities,neighbours,u,item)}
        r = r.append(new_row, ignore_index=True)
    
    r = r.sort_values(by=['rating'], ascending=False).head(t)
    r.reset_index(inplace=True)
    
    return r

######################### MODEL BASED APPROACH #########################


def calc_optimal_fold(data, max_fold):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        max_fold {[type]} -- [description]
    """    
    columns =['k', 'singular_value_sigma']
    zero_data = np.zeros(shape=(max_fold-1,len(columns)))
    df = pd.DataFrame(zero_data, columns=columns)

    for i in range(1, max_fold, 1):
        U, s, VT = svds(data,i)

        df.at[i-1, 'k'] = i
        df.at[i-1, 'singular_value_sigma'] = s[0]


    plt.figure(figsize=(10,8))
    plt.scatter(df['k'], df['singular_value_sigma'], s=70, c='darkred', label = 'sigma' )
    plt.plot(df['k'], df['singular_value_sigma'], c='salmon', alpha=0.5)
    plt.title('Change of sigma (singular value) based on size of k (folds)')
    plt.ylabel('singular value - sigma')
    plt.xlabel('size of k (folds)')
    plt.legend(loc='upper right')
    plt.show()


def products_single_user_recommendations_mobelbased(user_id, predictions, original_purchases, n_of_recommendations):
    """[summary]

    Arguments:
        user_id {[type]} -- [description]
        predictions {[type]} -- [description]
        original_purchases {[type]} -- [description]
        n_of_recommendations {[type]} -- [description]

    Returns:
        [type] -- [description]
    """    
    original_purchases.loc[user_id].sort_values(ascending=False).head()
    products_user = original_purchases.loc[user_id]
    bought_products_user = products_user[products_user > 0].index
    products_recommended = predictions.loc[user_id].sort_values(ascending=False)
    products_recommended = products_recommended.index

    res = products_recommended.difference(bought_products_user, sort=False)

    products_high_recommend = res[:n_of_recommendations].index
    recommendations = list(set(products_high_recommend) - set(bought_products_user))

    return recommendations


def products_recommendations_mobelbased(user_index, predicted_rating, train_set_count, n_of_recommendations):
    
    # reconstruct dense matrix from sparse test matrix
    matrix_count_dense = train_set_count.todense()
    
    # Create DataFrame for ease of use
    df_matrix = pd.DataFrame(matrix_count_dense)

    # Save only bought products
    products_user = df_matrix.loc[user_index]
    bought_products_user = products_user[products_user > 0].index

    # Create Prediction DataFrame
    preds_df = pd.DataFrame(predicted_rating)

    # Extract recommendations values for user
    products_recommended = preds_df.loc[user_index]

    # drop bought products
    products_recommended = products_recommended.drop(bought_products_user)

    # sort predictions by rating
    products_recommended = products_recommended.sort_values(ascending=False)


    # get n highest recommendations
    products_high_recommend = products_recommended.iloc[:n_of_recommendations].index

    return products_high_recommend

