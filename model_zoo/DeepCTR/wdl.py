# -*- coding:utf-8 -*-

import tensorflow as tf

from deepctr.feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import add_func, combined_dnn_input

from model_zoo.DeepCTR.mlora import Mlora
from model_zoo.lora_moe import MloraMoE


def WDL(linear_feature_columns, dnn_feature_columns, n_domain, lora_reduce, dnn_hidden_units=(256, 128, 64), l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary', num_experts=4):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())
    domain_input_layer = inputs_list[-1]

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    if num_experts <= 1:
        dnn_out = Mlora(n_domain, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed,lora_reduce=lora_reduce)([dnn_input,domain_input_layer])
    else:
        dnn_out = MloraMoE(n_domain, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed,lora_reduce=lora_reduce,num_experts=num_experts)([dnn_input,domain_input_layer])
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    final_logit = add_func([dnn_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


