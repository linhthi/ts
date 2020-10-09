import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
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
        user = Input(shape=(), dtype="int32", name="user_id")
        item = Input(shape=(), dtype="int32", name="item_id")

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
        mf_cat_latent = Flatten()([mf_user_embedding, mf_item_embedding])

        # MLP vector
        mlp_user_latent = Flatten()(mlp_user_embedding(user))
        mlp_item_latent = Flatten()(mlp_item_embedding(item))
        mlp_cat_latent = Flatten()([mlp_user_embedding, mlp_item_embedding])

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
        result = Dense(1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction")
        output = result(predict_layer)

        model = Model(
            inputs=[user, item],
            outputs=[output]
        )

        return model
