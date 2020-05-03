import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import squarify

# Plots to analyze to understand setting for recommender

def fig_prod_bought(data, safe):
    # number of times a product is bought
    # show the products most bought
    n_of_products_bought = data.product_name.value_counts()
    n_of_products_bought_20 = n_of_products_bought.nlargest(20)

    plt.figure(figsize=(16,7))
    sns.barplot(x=n_of_products_bought_20.index, y=n_of_products_bought_20.values)
    plt.xlabel('Products most ordered')
    plt.ylabel('Number of Orders')
    plt.xticks(rotation=90)

    plt.show()

    if safe == 0:
        pass
    else:
        plt.savefig('fig_prod_bought.png', bbox_inches='tight')



def fig_prod_per_order(data, safe):

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
    # Number of Orders per Department
    p_c = data.groupby('department')['product_name'].count().sort_values()

    # plot as Treemap
    plt.figure(figsize=(16,9))
    squarify.plot(sizes=p_c.values, label=p_c.index, alpha=.8 )
    plt.axis('off')
    plt.show() 

    if safe == 0:
        pass
    else:
        plt.savefig('departments_treemap.png')

    # plot as a barplot
    plt.figure(figsize=(12,7))
    sns.barplot(x=p_c.values, y=p_c.index)
    plt.xscale('log')
    plt.show()

    if safe == 0:
        pass
    else:
        plt.savefig('departments_bar.png', bbox_inches='tight')


def info_order_per_user(data):

    # Number of orders per user
    n_of_ord_per_user = data.groupby('user_id')['order_id'].nunique()

    # Find the treshhold for the lowest 75% of data points
    print(n_of_ord_per_user.describe())