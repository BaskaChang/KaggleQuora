
# coding: utf-8

# In[1]:

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim,os
import tokenize
import numpy as np
import pandas as pd


# In[2]:
os.chdir("your_path")
df_train = pd.read_csv('../../Data/train_clean.csv',encoding='utf-8')
df_test = pd.read_csv('../../Data/test_clean.csv',encoding='utf-8')


# In[3]:

qid_stack = df_train['qid1'].append(df_train['qid2'])
df_train_naExc = df_train[~df_train.isnull().any(axis=1)]
df_buf1 = df_train_naExc[['qid1','question1']]
df_buf2 = df_train_naExc[['qid2','question2']]
df_buf1.columns = ['qid','question']
df_buf2.columns = ['qid','question']
df_stack = df_buf1.append(df_buf2)
df_stack_dupExc = df_stack.drop_duplicates()
df_stack_dupExc['data_set'] = "train"


# In[4]:

print(len(df_stack_dupExc))
df_stack_dupExc.head()


# In[5]:

qid_stack = df_test['test_id']
df_train_naExc = df_test[~df_test.isnull().any(axis=1)]
df_buf1 = df_train_naExc[['test_id','question1']]
df_buf2 = df_train_naExc[['test_id','question2']]
df_buf1.columns = ['qid','question']
df_buf2.columns = ['qid','question']
df_stack = df_buf1.append(df_buf2)
df_stack_dupExc2 = df_stack.drop_duplicates()
df_stack_dupExc2['data_set'] = "test"


# In[6]:

print(len(df_stack_dupExc2))
df_stack_dupExc2.head()


# In[40]:

dataFrame_buf = df_stack_dupExc.append(df_stack_dupExc2)
test_buf = dataFrame_buf['question']#.head(1000)#.iloc[5000:7000]#head(1000)


# In[13]:

len(test_buf)


# In[41]:

# compile sample documents into a list
# doc_buf = df_stack_dupExc['question']
# doc_buf = doc_buf.tolist()
# doc_set = doc_buf


# In[36]:

#########
######### Tokenize the doc
#########
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
############
def tokenize_doc(new_corpus):
    new_corpus
    texts = []
    # loop through document list
    for i in new_corpus:    
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
#         stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        stemmed_tokens = []
        for i in stopped_tokens:
            try:
                stemmed_tokens = stemmed_tokens + [p_stemmer.stem(i)]
            except:
                print(i)
                stemmed_tokens = stemmed_tokens + [i]
        # add tokens to list            
        texts.append(stemmed_tokens)
    return texts

# ######
# ######  predict the topic
# ######
# def topic_predict(new_corpus, model, dictionary):
#     tokenize_buf = tokenize_doc([new_corpus])
#     doc_bow = [dictionary.doc2bow(text) for text in tokenize_buf]
#     que_vec = [item for itemList in doc_bow for item in itemList]
#     topic_vec = ldamodel[que_vec]

#     word_count_array = np.empty((len(topic_vec), 2), dtype = np.object)
#     for i in range(len(topic_vec)):
#         word_count_array[i, 0] = topic_vec[i][0]
#         word_count_array[i, 1] = topic_vec[i][1]

#     idx = np.argsort(word_count_array[:, 1])
#     idx = idx[::-1]
#     word_count_array = word_count_array[idx]

#     final = []
#     final = ldamodel.print_topic(word_count_array[0, 0], len(word_count_array))#1)

#     question_topic = final.split('*') ## as format is like "probability * topic"
#     return question_topic
#     #return question_topic[0]


# In[37]:

#texts = tokenize_doc(doc_set)
test_LDA_buf = test_buf.tolist()
doc_set = test_LDA_buf


# In[42]:

len(dataFrame_buf)


# In[44]:

num_piece = round(len(doc_set)/len(df_stack_dupExc))
sub_doc_set = []
for piece in range(num_piece):
#    print("piece:"+str(piece))
    if num_piece<=8:
        sub_piece = doc_set[((piece)*len(df_stack_dupExc)+1):(piece+1)*len(df_stack_dupExc)]
#         print("len:"+str(len(sub_piece)))
#         print("****from:"+str((piece)*len(df_stack_dupExc)))
#         print("to:"+str((piece+1)*len(df_stack_dupExc)))
        sub_doc_set = sub_doc_set + [sub_piece]
    else:
        sub_piece = doc_set[((piece)*len(df_stack_dupExc)+1):]
#         print("len:"+str(len(sub_piece)))
#         print("****from:"+str((piece)*len(df_stack_dupExc)))
#         print("to:"+str((piece+1)*len(df_stack_dupExc)))
        sub_doc_set = sub_doc_set + [sub_piece]


# In[45]:

print(len(sub_doc_set))
sub_doc_set[1][:10]


# In[20]:

import pickle

# selfref_list = [1, 2, 3]
# selfref_list.append(selfref_list)
for sub_take in range(len(sub_doc_set)):
    texts = tokenize_doc(sub_doc_set[sub_take])
    token_ind = sub_take
    pkl_path = '../../Data/token_piece'+str(token_ind)+'.pkl'
    output = open(pkl_path, 'wb')
    if num_piece<=8:
        range_df = [((token_ind)*len(df_stack_dupExc)+1),(token_ind+1)*len(df_stack_dupExc)]
    else:
        range_df = [((token_ind)*len(df_stack_dupExc)+1),len(df_stack_dupExc)] 
    save_dataFrame = dataFrame_buf.iloc[range_df[0]:range_df[1]]
    dict_buf = {"range_df_stack": range_df,
            "data_df": save_dataFrame,
            "tokenize_data": texts}
# Pickle dictionary using protocol 0.
    pickle.dump(dict_buf, output)

# Pickle the list using the highest protocol available.
#pickle.dump(selfref_list, output, -1)
    output.close()


# In[ ]:




# In[ ]:



