import pandas as pd
import numpy as np

def reduce_products(data, top_percent):
    # number of products
    n_of_products = data.product_name.nunique()

    # 20% is the regular percentage of reducing the products
    top_20 = int(n_of_products * top_percent)

    # select the top products
    n_of_products_bought = data.product_name.value_counts()
    prod_f = n_of_products_bought.nlargest(top_20)
    top_products = prod_f.index

    # filter the transactions only for the top products
    data = data[(data.product_name.isin(top_products))]

    # product most and least bought
    print('Product most bought: {0}'.format(prod_f.nlargest(1)))
    print('Product least bought (top 20%): {0}'.format(prod_f.nsmallest(1)))

    return data

def reduce_users_prod (data, number_of_products):
    # Number of purchases per User and Product
    n_of_purch_per_user = data.groupby(['user_id','product_id']).size()

    # Number of Product per User
    n_of_prod_per_user = n_of_purch_per_user.groupby('user_id').size()

    # Users with most and least products
    print('Users with most products: {0}'.format(n_of_prod_per_user.nlargest(5)))
    print('Users with least products: {0}'.format(n_of_prod_per_user.nsmallest(5)))

    # Filter for Users with more than n products
    n = number_of_products

    prod_per_user_f = n_of_prod_per_user[(n_of_prod_per_user >= n)]
    top_users = prod_per_user_f.index

    # Select Transactions from Users with morn than n products bought
    data = data[(data.user_id.isin(top_users))]

    return data

def reduce_user_purch(data, number_of_purchases):

    # Number of orders per user
    n_of_ord_per_user = data.groupby('user_id')['order_id'].nunique()

    # Find the treshhold for the lowest 75% of data points
    n_of_ord_per_user.describe()

    # Filter out users with n orders
    n = number_of_purchases
    n_of_ord_per_user_f = n_of_ord_per_user[n_of_ord_per_user > n]

    # Create list with users to filter
    top_users = n_of_ord_per_user_f.index.tolist()

    # Filter out all users that are not in the user list
    data = data[(data.user_id.isin(top_users))]

    return data