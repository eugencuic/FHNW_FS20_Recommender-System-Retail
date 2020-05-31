import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import squarify

# Plots to analyze to understand setting for recommender

def fig_prod_bought(data, safe):
    """Produces a graphic to show how many times a product was bought

    Arguments:
        data {pandas_frame} -- Frame that contains all transactions
        safe {bool} -- Boolean Value and option to save the plot

    Returns:
        Plot
    """    
    font = {'family' : 'arial',
            'weight' : 'light',
            'size'   : 14}

    plt.rc('font', **font)
    plt.figure(figsize=(14,6))


    # number of times a product is bought
    # show the products most bought
    n_of_products_bought = data.product_name.value_counts()
    n_of_products_bought_20 = n_of_products_bought.nlargest(20)

    sns.barplot(x=n_of_products_bought_20.index, y=n_of_products_bought_20.values)
    plt.xlabel('Products most ordered')
    plt.ylabel('Number of Orders')
    plt.title('Distribution of most bought products across all Transactions')
    plt.xticks(rotation=90)

    plt.show()

    if safe == 0:
        pass
    else:
        plt.savefig('fig_prod_bought.png', bbox_inches='tight')



def fig_prod_per_order(data, safe):
    """Produces a graphic to show how many products per order have been bought

    Arguments:
        data {pandas_frame} -- Frame that contains all transactions
        safe {bool} -- Boolean Value and option to save the plot

    Returns:
        Plot
    """
    font = {'family' : 'arial',
            'weight' : 'light',
            'size'   : 14}

    plt.rc('font', **font)
    plt.figure(figsize=(14,6))

    # Number of products bought per order
    # first row = number of products per order, second row = how many orders
    # This function does not consider if products are bought multiple times
    n_prod_order = data.groupby('order_id')['product_id'].count().value_counts().sort_index() 

    # limit to only plot the first n values
    n_prod_order_n = n_prod_order.head(50) 

    sns.barplot(x=n_prod_order_n.index, y=n_prod_order_n.values)
    plt.xlabel('Number of Products per Order')
    plt.ylabel('Orders count')

    # Adjust the lenght of the plot. Source: 
    # https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks-in-matplotlib

    N = 50
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])

    # inch margin
    m = 0.2 
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.show()

    if safe == 0:
        pass
    else:
        plt.savefig('prod_per_order.png', bbox_inches='tight')



def fig_ord_per_department(data, safe):
    """Produces a graphic to show how many products per department have been bought

    Arguments:
        data {pandas_frame} -- Frame that contains all transactions
        safe {bool} -- Boolean Value and option to save the plot

    Returns:
        Plot
    """
    font = {'family' : 'arial',
            'weight' : 'light',
            'size'   : 14}

    plt.rc('font', **font)
    # Number of Orders per Department
    p_c = data.groupby('department')['product_name'].count().sort_values()

    # plot as Treemap
    plt.figure(figsize=(16,9))
    squarify.plot(sizes=p_c.values, label=p_c.index, alpha=.8 )
    plt.axis('off')
    plt.title('Visual representation of number of Products bought in Departments, \nwhere size of square is the relative size of bought products')
    plt.show() 

    if safe == 0:
        pass
    else:
        plt.savefig('departments_treemap.png')

    # plot as a barplot
    plt.figure(figsize=(12,7))
    sns.barplot(x=p_c.values, y=p_c.index)
    plt.title('Visual representation as bar blot of number of Products bought in Departments')
    plt.xlabel('number of times bought (log)')
    plt.xscale('log')
    plt.show()

    if safe == 0:
        pass
    else:
        plt.savefig('departments_bar.png', bbox_inches='tight')


def fig_n_of_ord_per_user(data):
    """Produces a graphic to show how many orders per user have been done

    Arguments:
        data {pandas_frame} -- Frame that contains all transactions

    Returns:
        Plot
    """   
    font = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 14}

    plt.rc('font', **font)
    plt.figure(figsize=(16,8))

    # Create data to be plot
    n_of_ord_per_user = data.groupby('user_id')['order_id'].count().value_counts()

    # Sort by biggest first
    n_of_ord_per_user.sort_values(ascending=False)

    # reset index to have a linear order of numbers
    n_of_ord_per_user.reset_index(drop=True, inplace=True)

    # set x-axis limit based on intput shape
    xlen = len(n_of_ord_per_user)

    plt.bar(((n_of_ord_per_user.index)/xlen)*100, n_of_ord_per_user, label='Orders per User', color='steelblue')
    plt.legend(loc='best')
    plt.xlabel('% of User')
    plt.ylabel('Number of Orders per User')
    plt.title('Distribution of Orders per User')

    plt.show()

def cal_limit_orders_per_user(data, threshold):
    """Calcualtes the threshold where to cut off ertain % of users

    Arguments:
        data {pandas_frame} -- Frame that contains all transactions
        threshold {int} -- Number between 0-100 in % of users to cut off

    Returns:
        [int] -- Returns the number of orders where the cut off can happen
    """    
    # Create data to be plot
    n_of_ord_per_user = data.groupby('user_id')['order_id'].count().value_counts()

    # Sort by biggest first
    n_of_ord_per_user.sort_values(ascending=False)

    # reset index to have a linear order of numbers
    n_of_ord_per_user.reset_index(drop=True, inplace=True)

    # find the value of orders based on input % 
    cutoff = n_of_ord_per_user[int(len(n_of_ord_per_user)*(1-threshold/100))]

    # calculate cutoff users
    remainder = 100-threshold
    print("Bei {} Käufen pro Kunden können {} Prozent der Kunden eliminiert werden".format(cutoff, remainder))

    return cutoff

######################### DEPRECIATED #########################

# DEPRECIATED BECAUSE THE ANALYSES IS NOT NECESSARY
def num_of_prod_per_department(data):
    # Number of Products per Department
    n_p = data.groupby('department')['product_name'].value_counts()
    print(n_p.head(10))

    # Number of purchases per Category / Subcategory
    p_c_s = data.groupby('department')['aisle'].value_counts()
    print(p_c_s.head(10))

    del n_p, p_c_s