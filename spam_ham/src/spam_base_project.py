#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', None)
import numpy  as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Analysis and Visualization

# In[2]:


# load the data
spam_data = pd.read_csv("../Data/spambase.csv",  sep = ',', header= None )
print(spam_data.head())


# The colunmn names are integers so renaming the columns appropriately (column names are available at the UCI website here: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names)

# In[7]:



# renaming the columns
spam_data.columns  = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 
                      "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 
                      "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 
                      "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 
                      "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 
                      "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", 
                      "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", 
                      "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 
                      "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
                      "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", 
                      "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 
                      "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average", 
                      "capital_run_length_longest", "capital_run_length_total", "spam"]
print(spam_data.head())


# In[8]:


#To show the distribution of spam data
f,ax=plt.subplots(1,2,figsize=(18,8))
spam_data['spam'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Category')
ax[0].set_ylabel('')
sns.countplot('spam',data=spam_data,ax=ax[1])
ax[1].set_title('Category')
plt.show()


# In[9]:


sns.set_style("darkgrid")
sns.countplot(spam_data.spam)


# In[10]:


# look at dimensions of the df
print(spam_data.shape)


# In[11]:


# check missing values in the dataset 
spam_data.isnull().sum()


# ## Data Preparation
# Let's now conduct some prelimininary data preparation steps, i.e. rescaling the variables.Rescaling is required as some columns like e.g at the end (capital_run_length_longest, capital_run_length_total etc.) have much higher values (means = 52, 283 etc.) than most other columns which represent fraction of word occurrences (no. of times word appears in email/total no. of words in email)

# In[12]:


spam_data.describe()


# In[13]:


# splitting into X and y in order to seperate labels from features
X = spam_data.drop("spam", axis = 1)
y = spam_data.spam.values.astype(int)


# In[14]:


# scaling the features
# note that the scale function standardises each column, i.e.
# x = x-mean(x)/std(x)

from sklearn.preprocessing import scale
X = scale(X)


# In[15]:


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


# In[16]:


# confirm that splitting also has similar distribution of spam and ham 
# emails
print(y_train.mean())
print(y_test.mean())


# # Classification

# 
# # Part 1

# In[17]:


#Defining the model architecture
#A dropout layer is inserted  as a form of regularization 
#which will help reduce overfitting by randomly setting (here 30%) of the input unit values to zero.
model = keras.Sequential(
    [
        keras.layers.Dense(units=9, activation="relu", input_shape=(X_train.shape[-1],) ),
        # randomly delete 30% of the input units below
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=9, activation="relu"),
        # the output layer, with a single neuron
        keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)

# save the initial weights for later
initial_weights = model.get_weights()


# In[18]:


model.summary()


# In[19]:


import visualkeras


# In[20]:


from PIL import ImageFont


# In[21]:


visualkeras.layered_view(model, legend=True)  # font is optional!


# In[22]:


learning_rate = 0.001

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
              loss="binary_crossentropy", 
              metrics=keras.metrics.AUC()
             )


# In[23]:


history = model.fit(X_train, y_train, 
          epochs=500, 
          batch_size=1000, 
          validation_data=(X_test, y_test),
          verbose=1)


# # Part 2
# Finding the co relation between features and with y label 

# In[130]:


# Finding the co relation between each feature and label
# As the features are numerical and label is categorical so kendall is used to measure cor relation 
X = spam_data.drop("spam", axis = 1)
corr_action=X.corrwith(spam_data['spam'],method='kendall').abs()
print(corr_action)


# In[131]:


corr_action.sort_values(ascending=False).head(10)


# In[132]:


# Selecting those features whose co relation with dependent variable (y)
relevant_num_features = corr_action[corr_action>0.3]


# In[133]:


relevant_num_features_col=relevant_num_features.index


# In[134]:


selected_num_df = X[relevant_num_features_col]  # Only getting columns having corr> 0.3 wrt output


# In[135]:


selected_num_df.info()


# In[136]:


# To fing co relation between features
pearson_corr = X.corr(method='pearson').abs()
#Using Pearson Correlation
plt.figure(figsize=(20,20))
# cor = df.corr()
sns.heatmap(pearson_corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[137]:


cor_matrix = X.corr().abs()
print(cor_matrix)


# In[138]:


upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)


# In[139]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(); print(to_drop)


# # Part3
# Removing the significant features and building a fully connected classifier 

# In[141]:


["char_freq_!","char_freq_$","word_freq_remove","word_freq_free","word_freq_money","word_freq_your",
"capital_run_length_longest","word_freq_000","capital_run_length_average","word_freq_hp"]


# In[142]:


# Dropping at are highly significant i.e those features that have large impact on y 
#splitting into X and y
X_1 = spam_data.drop(["char_freq_!","char_freq_$","word_freq_remove","word_freq_free","word_freq_money","word_freq_your",
"capital_run_length_longest","word_freq_000","capital_run_length_average","word_freq_hp","spam"], axis = 1)
y_1 =spam_data.spam.values.astype(int)


# In[143]:


X_1 = scale(X_1)


# In[144]:


# split into train and test
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size = 0.3, random_state = 4)


# In[145]:


#Defining the model architecture
#A dropout layer is inserted  as a form of regularization 
#which will help reduce overfitting by randomly setting (here 30%) of the input unit values to zero.
model = keras.Sequential(
    [
        keras.layers.Dense(units=9, activation="relu", input_shape=(X_train_1.shape[-1],) ),
        # randomly delete 30% of the input units below
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=9, activation="relu"),
        # the output layer, with a single neuron
        keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)

# save the initial weights for later
initial_weights = model.get_weights()


# In[146]:


learning_rate = 0.001

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
              loss="binary_crossentropy", 
              metrics=keras.metrics.AUC())
             


# In[147]:


history = model.fit(X_train_1, y_train_1, 
          epochs=500, 
          batch_size=1000, 
          validation_data=(X_test_1, y_test_1),
          verbose=1)


# In[28]:


# load the test data
test_data = pd.read_csv("../test/test.csv",  sep = ',', header= None )
print(test_data.head())


# In[36]:


test_data.shape


# In[34]:


#Predict the class of data
pred=model.predict(test_data)


# In[35]:


print(pred)


# In[ ]:





# In[ ]:




