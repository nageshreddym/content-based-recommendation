#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


credits.head(1)['cast']


# In[6]:


credits.head(1)['cast'].values


# In[7]:


movies.merge(credits,on='title')


# In[8]:


movies.merge(credits,on='title').shape


# In[9]:


movies=movies.merge(credits,on='title')


# In[10]:


movies.shape


# In[11]:


movies.info()


# In[12]:


movies[['id','genres','overview','title','keywords','cast','crew']]


# In[13]:


movies=movies[['id','genres','overview','title','keywords','cast','crew']]
movies


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.duplicated().sum()


# In[17]:


movies.iloc[0].genres


# In[18]:


def converter(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[20]:


movies['genres'].apply(converter)


# In[21]:


movies['genres']=movies['genres'].apply(converter)


# In[22]:


movies['genres']


# In[23]:


movies


# In[24]:


movies['keywords']=movies['keywords'].apply(converter)


# In[25]:


movies['keywords']


# In[26]:


movies


# In[27]:


movies['cast'][0]


# In[28]:


def converter3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[29]:


movies['cast'][0]


# In[30]:


movies['cast']=movies['cast'].apply(converter3)


# In[31]:


movies


# In[32]:


movies['crew'][0]


# In[33]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L


# In[34]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[35]:


movies


# In[36]:


movies['overview'][0]


# In[37]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[38]:


movies


# In[39]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[40]:


movies['cast']


# In[41]:


movies


# In[42]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['overview']=movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# movies.head()

# In[43]:


movies.head()


# In[44]:


movies['tags']=movies['overview']+movies['keywords']+movies['genres']+movies['cast']+movies['crew']


# In[45]:


movies['tags'][0]


# In[46]:


movies.head()


# In[47]:


new_df=movies[['id','title','tags']]


# In[48]:


new_df


# In[49]:


new_df['tags'][0]


# In[50]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[51]:


new_df['tags'][0]


# In[52]:


new_df['tags']


# In[53]:


new_df.head()


# In[54]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[55]:


new_df['tags'][0]


# In[56]:


new_df.head()


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[58]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[59]:


cv.fit_transform(new_df['tags']).toarray()


# In[60]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[61]:


vectors[0]


# In[62]:


cv.get_feature_names()


# In[63]:


import nltk
#verbal form


# In[64]:


from nltk.stem.porter import PorterStemmer


# In[65]:


ps=PorterStemmer()


# In[66]:


ps.stem('loved')


# In[67]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[68]:


new_df['tags']=new_df['tags'].apply(stem)


# In[69]:


new_df['tags']


# In[70]:


from sklearn.metrics.pairwise import cosine_similarity


# In[71]:


cosine_similarity(vectors)


# In[72]:


cosine_similarity(vectors).shape


# In[73]:


similarity=cosine_similarity(vectors)


# In[74]:


similarity


# In[75]:


similarity[0]


# In[76]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[77]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[86]:


recommend('Avatar')


# In[87]:


import pickle
pickle.dump(new_df,open('movies.pkl','wb'))


# In[80]:


import os

os.getcwd()


# In[81]:


import pickle
pickle.dump(new_df.to_dict(),open('Movies.pkl','wb'))


# In[82]:


import os

os.getcwd()


# In[83]:


new_df


# In[84]:


new_df['title'].values


# In[85]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




