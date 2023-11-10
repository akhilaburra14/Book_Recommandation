#!/usr/bin/env python
# coding: utf-8

# In[45]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install -U scikit-learn')


# In[3]:


import numpy as np
import pandas as pd


# In[6]:


books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')


# In[7]:


books.head()


# In[8]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[9]:


books.isnull().sum()
ratings.isnull().sum()
users.isnull().sum()


# In[10]:


users.isnull().sum()


# In[11]:


ratings.isnull().sum()


# In[12]:


books.isnull().sum()


# In[13]:


books.duplicated().sum()


# In[14]:


ratings.duplicated().sum()


# In[15]:


users.duplicated().sum()


# In[17]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[18]:


ratings_with_name


# ## Collaborative Filtering Based Recommender System

# In[23]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
read_rated_users = x[x].index
read_rated_users


# In[28]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(read_rated_users)]


# In[33]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[36]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[40]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[41]:


pt.fillna(0, inplace=True)


# In[42]:


pt


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity_scores = cosine_similarity(pt)


# In[48]:


similarity_scores.shape


# In[67]:


def recommend(book_name):
    #index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key= lambda x:x[1],reverse=True)[1:6]
    
    for i in similar_items:
        print(pt.index[i[0]])


# In[68]:


recommend('Message in a Bottle')


# In[ ]:




