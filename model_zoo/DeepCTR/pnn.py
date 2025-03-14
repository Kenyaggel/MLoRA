# -*- coding:utf-8 -*-


import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import InnerProductLayer, OutterProductLayer
from deepctr.layers.utils import concat_func, combined_dnn_input

from model_zoo.DeepCTR.mlora import Mlora
from model_zoo.lora_moe import MloraMoE


def PNN(dnn_feature_columns, n_domain, lora_reduce, dnn_hidden_units=(256, 128, 64), l2_reg_embedding=0.00001, l2_reg_dnn=0,
        seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False, kernel_type='mat',
        task='binary', num_experts=4):
    """Instantiates the Product-based Neural Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if kernel_type not in ['mat', 'vec', 'num']:
        raise ValueError("kernel_type must be mat,vec or num")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    domain_input_layer = inputs_list[-1]

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    inner_product = tf.keras.layers.Flatten()(
        InnerProductLayer()(sparse_embedding_list))
    outter_product = OutterProductLayer(kernel_type)(sparse_embedding_list)

    # ipnn deep input
    linear_signal = tf.keras.layers.Reshape(
        [sum(map(lambda x: int(x.shape[-1]), sparse_embedding_list))])(concat_func(sparse_embedding_list))

    if use_inner and use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, inner_product, outter_product])
    elif use_inner:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, inner_product])
    elif use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, outter_product])
    else:
        deep_input = linear_signal

    dnn_input = combined_dnn_input([deep_input], dense_value_list)
    # dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)
    if num_experts <= 1:
        dnn_out = Mlora(n_domain, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed,
                     lora_reduce=lora_reduce)([dnn_input, domain_input_layer])
    else:
        dnn_out = MloraMoE(n_domain, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed,
                        lora_reduce=lora_reduce, num_experts=num_experts)([dnn_input, domain_input_layer])

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    output = PredictionLayer(task)(dnn_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model
