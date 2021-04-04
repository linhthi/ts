import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    Multiply,
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from typing import List

__author__ = 'linhthi'


class NeuMF:
    def __init__(self,
                 number_of_users: int,
                 number_of_items: int,
                 latent_dim_mf: int = 4,
                 latent_dim_mlp: int = 32,
                 reg_mf: float = 0,
                 reg_mlp: float = 0.01,
                 dense_layers: List[int] = [8, 4],
                 reg_layers: List[int] = [0.01, 0.01],
                 activation_dense: str = "relu"
                 ) -> keras.Model:
        """ Init class NeuMF """

        self.activation_dense = activation_dense
        self.reg_layers = reg_layers
        self.dense_layers = dense_layers
        self.reg_mlp = reg_mlp
        self.reg_mf = reg_mf
        self.latent_dim_mlp = latent_dim_mlp
        self.latent_dim_mf = latent_dim_mf
        self.number_of_items = number_of_items
        self.number_of_users = number_of_users

    def get_model(self):
        # input layer
        user = Input(shape=(1), dtype="int32", name="user_id")
        item = Input(shape=(1), dtype="int32", name="item_id")

        # embedding layer
        mf_user_embedding = Embedding(
            input_dim=self.number_of_items,
            output_dim=self.latent_dim_mf,
            name="mf_user_embedding",
            embeddings_initializer="RandomNormal",
            embeddings_regularizer=l2(self.reg_mf),
            input_length=1,
        )
        mf_item_embedding = Embedding(
            input_dim=self.number_of_items,
            output_dim=self.latent_dim_mf,
            name="mf_item_embedding",
            embeddings_initializer="RandomNormal",
            embeddings_regularizer=l2(self.reg_mf),
            input_length=1,
        )
        mlp_user_embedding = Embedding(
            input_dim=self.number_of_users,
            output_dim=self.latent_dim_mlp,
            name="mlp_user_embedding",
            embeddings_initializer="RandomNormal",
            embeddings_regularizer=l2(self.reg_mlp),
            input_length=1,
        )
        mlp_item_embedding = Embedding(
            input_dim=self.number_of_items,
            output_dim=self.latent_dim_mlp,
            name="mlp_item_embedding",
            embeddings_initializer="RandomNormal",
            embeddings_regularizer=l2(self.reg_mlp),
            input_length=1,
        )

        # MF vector
        mf_user_latent = Flatten()(mf_user_embedding(user))
        mf_item_latent = Flatten()(mf_item_embedding(item))
        mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

        # MLP vector
        mlp_user_latent = Flatten()(mlp_user_embedding(user))
        mlp_item_latent = Flatten()(mlp_item_embedding(item))
        mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

        mlp_vector = mlp_cat_latent

        # build dense layers for model
        for i in range(len(self.dense_layers)):
            layer = Dense(
                self.dense_layers[i],
                activity_regularizer=l2(self.reg_layers[i]),
                activation=self.activation_dense,
                name="layer%d" % i,
            )
            mlp_vector = layer(mlp_vector)

        predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
        result = Dense(1, name="predict_rating")
        output = result(predict_layer)

        model = Model(
            inputs=[user, item],
            outputs=output
        )

        return model

    def train(self, train, valid, n_epochs):
        N_EPOCHS = n_epochs
        neuMF_model = self.get_model()

        neuMF_model.compile(
            optimizer=SGD(lr=0.001),
            loss = tf.keras.losses.MSE,
            metrics=[
                tf.keras.metrics.mae,
                tf.keras.metrics.RootMeanSquaredError(),
            ]
        )

        neuMF_model._name = "neural_matrix_factorization"
        neuMF_model.summary()

        user_train, item_train, rating_train = [], [], []
        for data in train:
            user_train.append(data[0])
            item_train.append(data[1])
            rating_train.append(data[3])
        # print(user_train.reshape())
        # print(ds_train.shape)

        user_valid, item_valid, rating_valid = [], [], []
        for data in valid:
            user_valid.append(data[0])
            item_valid.append(data[1])
            rating_valid.append(data[3])

        # define logs and callbacks
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=6)

        train_hist = neuMF_model.fit(
            x=[np.array(user_train), np.array(item_train)],  # input
            y = np.array(rating_train).astype(float),  # label rating
            validation_data=([np.array(user_valid), np.array(item_valid)], np.array(rating_valid).astype(float)),
            epochs=N_EPOCHS,
            callbacks=[tensorboard_callback, early_stopping_callback],
            verbose=1,
        )
        neuMF_model.save("my_neuMF_h5_model.h5")

    def test(self, test):
        neuMF_model = keras.models.load_model("my_neuMF_h5_model.h5")
        user_test, item_test, rating_test = [], [], []
        for data in test:
            user_test.append(data[0])
            item_test.append(data[1])
            rating_test.append(data[3])
        predictions = neuMF_model.predict([np.array(user_test), np.array(item_test)])
        rmse_test = np.sqrt(keras.losses.MSE(np.array(rating_test).astype(float), predictions))
        mae_test = keras.metrics.mean_absolute_error(np.array(rating_test).astype(float), predictions)
        print("RMSE test: %0.4f, MAE test: %0.4f" % (rmse_test, mae_test))





