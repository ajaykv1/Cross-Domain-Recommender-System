#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:18:25 2022

@author: ajaykrishnavajjala

# Min-Yen Kan (Prof. Kan) from NUS (datasets)
# Atonios Professor GMU
# Ziyu Yao Professor GMU
# GMU Hopper cluster GPU

# ICML, NeurIps, RecSys, KDD, AAAI, SIGIR, IJCAI

"""
#%%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply, Average, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import os
import datetime
#%%
# opening the books dataset from amazon
booksDF = pd.read_csv("/Users/ajaykrishnavajjala/Documents/School/PHD/Recommender Systems/Summer 2022 Project/Datasets/Books.csv")
booksDF.columns = ["item_id", "user_id", "rating", "timestamp"]
# opening the movies dataset from amazon
moviesDF = pd.read_csv("/Users/ajaykrishnavajjala/Documents/School/PHD/Recommender Systems/Summer 2022 Project/Datasets/Movies_and_TV.csv")
moviesDF.columns = ["item_id", "user_id", "rating", "timestamp"]
#%%
print(booksDF.shape)
print(moviesDF.shape)
#%%
print(moviesDF.head())
#%%
booksDF = booksDF[:2000000]
moviesDF = moviesDF[:2000000]
#%%
print(booksDF.shape)
print(moviesDF.shape)
#%%
# only keeping user overlaps in both movies and bbooks domains
source = moviesDF.loc[moviesDF.user_id.isin(booksDF.user_id)].sort_values(by="user_id")
target = booksDF.loc[booksDF.user_id.isin(moviesDF.user_id)].sort_values(by="user_id")
#%%
# making sure item id's are from 0-n
source["item_id"] = pd.Categorical(source["item_id"])
source["itemId"] =source["item_id"].cat.codes
# making sure user id's are from 0-n
source["user_id"] = pd.Categorical(source["user_id"])
source["userId"] =source["user_id"].cat.codes
# making sure item id's are from 0-n
target["item_id"] = pd.Categorical(target["item_id"])
target["itemId"] = target["item_id"].cat.codes
# making sure user id's are from 0-n
target["user_id"] = pd.Categorical(target["user_id"])
target["userId"] = target["user_id"].cat.codes
#%%
#keeping 0-n values and dropping original
source = source.drop(["item_id", "user_id"], axis = 1)
target= target.drop(["item_id", "user_id"], axis = 1)
# making ratings binary, so if rating is >= 3, it is 1, else 0
ratings = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}
source["rating"] = source["rating"].map(ratings)
target["rating"] = target["rating"].map(ratings)
#%%
# gathering number of items and users in the source domain
source_num_users = len(set(source['userId'].values))
source_num_items = len(set(source['itemId'].values))

target_num_users = len(set(target['userId'].values))
target_num_items = len(set(target['itemId'].values))
#%%
print("source Info")
print(source_num_users)
print(source_num_items)
print("target Info")
print(target_num_users)
print(target_num_items)
#%%
def create_cdcars_model (num_users, num_items):
    
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    mf_user_embedding = Embedding(num_users, 150, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_user_emb') (user_input)
    mf_item_embedding = Embedding(num_items, 150, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_item_emb') (item_input)
    
    mlp_user_embedding = Embedding(num_users, 150, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_user_emb') (user_input)
    mlp_item_embedding = Embedding(num_items, 150, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_item_emb') (item_input)
    
    #GMF Layer
    mf_user_embedding = Flatten() (mf_user_embedding)
    mf_item_embedding = Flatten() (mf_item_embedding)
    gmf_layer = Multiply() ([mf_user_embedding, mf_item_embedding])
    
    #MLP Layer
    
    mlp_user_embedding = Flatten() (mlp_user_embedding)
    mlp_item_embedding = Flatten() (mlp_item_embedding)
    mlp_cat_layer = Concatenate() ([mlp_user_embedding, mlp_item_embedding])
    
    # Deep Network
    
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_cat_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    
    output_layer = Concatenate() ([gmf_layer, mlp_layer])
    output_layer = Dense(1, activation='sigmoid') (output_layer)
    
    cdcars_model = Model(inputs=[user_input,item_input], outputs=output_layer)
    
    return cdcars_model
#%%
source_model = create_cdcars_model(source_num_users,source_num_items)
#%%
source_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
source_model.summary()
#%%
source_users = source["userId"].values
source_items = source["itemId"].values
source_ratings = source["rating"].values
#%%
Ntrain = int(0.7 * len(source_ratings))

source_train_users = source_users[:Ntrain]
source_train_items = source_items[:Ntrain]
source_train_ratings = source_ratings[:Ntrain]

source_test_users = source_users[Ntrain:]
source_test_items = source_items[Ntrain:]
source_test_ratings = source_ratings[Ntrain:]
#%%
print(source_model.get_layer('mlp_item_emb').get_weights()[0])
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
source_results = source_model.fit(x = [source_train_users, source_train_items], 
        y = source_train_ratings, 
        epochs = 10, 
        validation_data = ([source_test_users, source_test_items], source_test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(source_results.history["loss"], label = 'loss')
plt.plot(source_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(source_results.history['accuracy'], label='train_acc')
plt.plot(source_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
source_preds = source_model.predict([source_test_users, source_test_items])
source_evaluation = source_model.evaluate([source_test_users, source_test_items], source_test_ratings)
#%%
print(source_evaluation)
#%%
target_model = create_cdcars_model(target_num_users,target_num_items)
#%%
target_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
target_model.summary()
#%%
target_users = target["userId"].values
target_items = target["itemId"].values
target_ratings = target["rating"].values
#%%
Ntrain = int(0.7 * len(target_ratings))

target_train_users = target_users[:Ntrain]
target_train_items = target_items[:Ntrain]
target_train_ratings = target_ratings[:Ntrain]

target_test_users = target_users[Ntrain:]
target_test_items = target_items[Ntrain:]
target_test_ratings = target_ratings[Ntrain:]
#%%
print(target_model.get_layer('mlp_item_emb').get_weights()[0])
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
target_results = target_model.fit(x = [target_train_users, target_train_items], 
        y = target_train_ratings, 
        epochs = 10, 
        validation_data = ([target_test_users, target_test_items], target_test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(target_results.history["loss"], label = 'loss')
plt.plot(target_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(target_results.history['accuracy'], label='train_acc')
plt.plot(target_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
target_preds = target_model.predict([target_test_users, target_test_items])
target_evaluation = target_model.evaluate([target_test_users, target_test_items], target_test_ratings)
#%%
print(target_evaluation)
#%%
source_user_mf_emb = source_model.get_layer('mf_user_emb').get_weights()[0]
source_user_mlp_emb = source_model.get_layer('mlp_user_emb').get_weights()[0]
source_user_embedding = Average() ([source_user_mf_emb, source_user_mlp_emb])

target_user_mf_emb = target_model.get_layer('mf_user_emb').get_weights()[0]
target_user_mlp_emb = target_model.get_layer('mlp_user_emb').get_weights()[0]
target_user_embedding = Average() ([target_user_mf_emb, target_user_mlp_emb])

target_item_mf_emb = target_model.get_layer('mf_item_emb').get_weights()[0]
target_item_mlp_emb = target_model.get_layer('mlp_item_emb').get_weights()[0]
target_item_embedding = Average() ([target_item_mf_emb, target_item_mlp_emb])
#%%
def final_cdcars_model(num_users, num_items, source_user_embedding, target_user_embedding, target_item_embedding):
    
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    user_emb = Average() ([source_user_embedding, target_user_embedding])
    item_emb = target_item_embedding
    
    user_embedding = Embedding(input_dim=num_users, output_dim=150, weights=[user_emb], trainable=False) (user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=150, weights=[item_emb], trainable=False) (item_input)
    
    user_embedding = Flatten() (user_embedding)
    item_embedding = Flatten() (item_embedding)
    cat_layer = Concatenate() ([user_embedding, item_embedding])

    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (cat_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    
    output_layer = Dense(1, activation='sigmoid') (mlp_layer)
    
    cdcars_model = Model(inputs=[user_input,item_input], outputs=output_layer)
    
    return cdcars_model
#%%
final_target_model = final_cdcars_model(target_num_users,target_num_items, source_user_embedding,target_user_embedding, target_item_embedding)
#%%
final_target_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
final_target_model.summary()
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
final_target_results = final_target_model.fit(x = [target_train_users, target_train_items], 
        y = target_train_ratings, 
        epochs = 10, 
        validation_data = ([target_test_users, target_test_items], target_test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(final_target_results.history["loss"], label = 'loss')
plt.plot(final_target_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(final_target_results.history['accuracy'], label='train_acc')
plt.plot(final_target_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
final_target_preds = final_target_model.predict([target_test_users, target_test_items])
final_target_evaluation = final_target_model.evaluate([target_test_users, target_test_items], target_test_ratings)
#%%
print(final_target_evaluation)
    
    
     
    
    
    






















