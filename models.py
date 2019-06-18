import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import pickle
from keras.initializers import Initializer
import logging

"##########################  Basic Functions ##########################"


def simple_attention(target):
    attention = keras.layers.Dense(1, activation=keras.activations.tanh)(target)
    attention = keras.layers.Reshape((-1,))(attention)
    attention_weight = keras.layers.Activation(keras.activations.softmax)(attention)
    return keras.layers.Dot((1, 1))([target, attention_weight])


def ca(input_size, filter_shape, embedding_layer, dropout, output_dim=None):
    filter_count, filter_size = filter_shape
    i = keras.Input((input_size,), dtype='int32')
    cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
    e = keras.layers.Dropout(dropout)(embedding_layer(i))
    c = cnn(e)
    a = simple_attention(keras.layers.Dropout(dropout)(c))
    if output_dim is not None:
        a = keras.layers.Dense(output_dim)(a)
    return keras.Model(i, a)


class LzComputeMasking(keras.layers.Layer):
    def __init__(self, mask_value=0., **kwargs):
        super(LzComputeMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]



class LzLogits:
    def __init__(self, mode="mlp"):
        assert mode in ["mlp", "dot"]
        self.mode = mode

    def __call__(self, inputs, *args, **kwargs):
        usr_vec, doc_vec = inputs[0], inputs[1]
        if self.mode == "mlp":
            cat = keras.layers.concatenate
            hidden = keras.layers.Dense(units=int(usr_vec.shape[-1]), activation="elu")(cat(inputs, axis=-1))
            logits = keras.layers.Dense(units=1, activation="sigmoid")(hidden)
        else:
            logging.info("Dot logit applied")
            assert usr_vec.shape[-1] == doc_vec.shape[-1]
            logits = keras.layers.Dot(axes=-1)(inputs)
            logits = keras.layers.Activation('sigmoid')(logits)
        return logits


class CustomInitializer(Initializer):
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, shape, dtype=None):
        self.weights.reshape(shape)
        return tf.convert_to_tensor(self.weights, dtype=dtype)

"##########################  model functions ##########################"


class LzQueryAttentionPooling:
    def __call__(self, value, query, *args, **kwargs):
        mapping = keras.layers.Dense(units=int(value.shape[-1]), activation="elu", use_bias=False)
        value = keras.layers.TimeDistributed(mapping)(value)
        result = LzExternalSingleHeadAttentionPooling()([value, query])
        return result


class LzMultiHeadAttentionWeight(keras.layers.Layer):
    def __init__(self, head_count, enable_pretrain_attention = False, **kwargs):
        super(LzMultiHeadAttentionWeight, self).__init__(**kwargs)
        self.init = keras.initializers.get('glorot_uniform')
        self.supports_masking = True
        self.head_count = head_count
        self.enable_pretrain_attention = enable_pretrain_attention

    def build(self, input_shape):
        if self.enable_pretrain_attention:
            logging.info("Pretrain Method Applied")
            with open('./models/AutoEncoder_' + str(self.head_count) + '.pkl', 'rb') as p:
                pre_weights_biases = pickle.load(p)
            pre_weights = pre_weights_biases[0]
            pre_weights = pre_weights.transpose()
            self.attention_heads = [self.add_weight(shape=(input_shape[2], 1),
                                                    initializer=CustomInitializer(pre_weights[i].reshape((input_shape[2],1))),
                                                    name="head-{}".format(i))
                                    for i in range(self.head_count)]
        else:
            self.attention_heads = [self.add_weight(shape=(input_shape[2], 1),
                                        initializer=self.init,
                                        name="head-{}".format(i))
                                    for i in range(self.head_count)]
        super(LzMultiHeadAttentionWeight, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, mask = inputs, LzComputeMasking(0)(inputs)
        vectors, weights = [], []
        for head in self.attention_heads:
            ait = K.squeeze(K.dot(value, head), axis=-1)
            a = K.exp(ait)
            if mask is not None:
                a *= K.cast(mask, K.floatx())
            a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
            vectors.append(K.expand_dims(keras.layers.dot([value, a], axes=1), axis=1))
            weights.append(K.expand_dims(a, axis=1))
        return [keras.layers.concatenate(vectors, axis=1),
                keras.layers.concatenate(weights, axis=1)]




    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.head_count, input_shape[2]),
                (input_shape[0], self.head_count, input_shape[1])]




class LzMultiHeadAttentionWeightOrth(LzMultiHeadAttentionWeight):
    
    def __init__(self, normalize=False, enable_pretrain_attention = False, **kwargs):
        self.normalize = normalize
        self.enable_pretrain_attention = enable_pretrain_attention
        super(LzMultiHeadAttentionWeightOrth, self).__init__(enable_pretrain_attention = self.enable_pretrain_attention, **kwargs)

    def call(self, inputs, **kwargs):
        value, mask = inputs, LzComputeMasking(0)(inputs)
        vectors, weights = [], []
        # "------- attention calculation -------"
        for head in self.attention_heads:
            ait = K.squeeze(K.dot(value, head), axis=-1)
            a = K.exp(ait)
            if mask is not None:
                a *= K.cast(mask, K.floatx())
            a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
            vectors.append(K.expand_dims(keras.layers.dot([value, a], axes=1), axis=1))
            weights.append(K.expand_dims(a, axis=1))
        # "----- orthogonal regularization -----"
        heads = K.concatenate(self.attention_heads, axis=1)
        # updated normalization  --Jan 27th
        if self.normalize:
            heads /= K.sqrt(K.sum(heads*heads, axis=-1, keepdims=True)) + K.epsilon()
        orth_reg = K.batch_dot(heads, K.transpose(heads))
        orth_reg = K.mean(orth_reg, axis=-1, keepdims=False)
        orth_reg = K.mean(orth_reg, axis=-1, keepdims=True)

        # if self.normalize:
        #     norm_item = K.sqrt(K.sum(orth_reg*orth_reg, axis=-1, keepdims=True)) + K.epsilon()
        #     orth_reg /= norm_item

        return [keras.layers.concatenate(vectors, axis=1),
                orth_reg]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.head_count, input_shape[2]),
                (1,)]





class LzCompressionPredictor:
    def __init__(self, channel_count, mode="Post", enable_pretrain_attention = False):
        self.channel_count = channel_count
        self.mode = mode
        self.enable_pretrain_attention = enable_pretrain_attention

    def __call__(self, docs, *args, **kwargs):
        hidden_dim = int(docs.shape[-1])
        mapping = keras.layers.Dense(units=hidden_dim, activation="elu", use_bias=False)
        docs = keras.layers.TimeDistributed(mapping)(docs)
        if self.mode == "Post":
            # vectors, weights = LzMultiHeadAttentionWeight(self.channel_count)(docs)
            # orthodox_reg = self._off_diag_norm(weights, normalization=True)
            vectors, weights = LzMultiHeadAttentionWeight(self.channel_count, self.enable_pretrain_attention)(docs)
            orthodox_reg = self._off_diag_norm(weights, normalization=True)
            return vectors, weights, orthodox_reg
        else:
            
            vectors, orthodox_reg = LzMultiHeadAttentionWeightOrth(head_count=self.channel_count,
                                                                enable_pretrain_attention=self.enable_pretrain_attention)(docs)
            return vectors, orthodox_reg
            


    def _off_diag_norm(self, weights, normalization=False):
        # updated normalization --Jan 27th
        if normalization:
            weights /= K.sqrt(K.sum(weights*weights, axis=-1, keepdims=True)) + K.epsilon()
        matrix = K.batch_dot(weights, K.permute_dimensions(weights, (0, 2, 1)))
        # if normalization:
        #     matrix /= K.sqrt(K.sum(matrix*matrix, axis=-1, keepdims=True)) + K.epsilon()
        mask = K.ones_like(matrix) - K.eye(int(matrix.shape[-1]))
        matrix = matrix * mask
        # result = K.sum(matrix, axis=-1, keepdims=False)
        # result = K.sum(result, axis=-1, keepdims=True)
        "updated results"
        result = K.mean(matrix, axis=-1, keepdims=False)
        result = K.mean(result, axis=-1, keepdims=True)
        return result



class _LzSelfAttention:
    def __init__(self, mapping=True):
        self.mapping = mapping

    def __call__(self, inputs):
        scalar = np.sqrt(int(inputs.shape[1])*1.0)
        self_attention = keras.layers.Lambda(lambda x: K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1))) / scalar)
        exponential = keras.layers.Lambda(lambda x: K.exp(x))
        zero_masking = keras.layers.Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=1))
        normalization = keras.layers.Lambda(lambda x: x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon()))

        if self.mapping:
            tran_mat = keras.layers.Dense(units=int(inputs.shape[-1]), activation="elu", use_bias=False)
            inputs = keras.layers.TimeDistributed(tran_mat)(inputs)

        mask = LzComputeMasking(0)(inputs)
        weights_matrix = self_attention(inputs)
        weights_matrix = exponential(weights_matrix)
        weights_matrix = zero_masking([weights_matrix, mask])
        weights_matrix = normalization(weights_matrix)
        outputs = keras.layers.Dot(axes=(-1, 1))([weights_matrix, inputs])
        return outputs



class LzExternalSingleHeadAttentionPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LzExternalSingleHeadAttentionPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LzExternalSingleHeadAttentionPooling, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, query = inputs
        mask = LzComputeMasking(0)(value)
        ait = keras.layers.dot([value, query], axes=-1)
        a = K.exp(ait)
        if mask is not None:
            a *= mask
        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        out = keras.layers.dot([value, a], axes=1)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0][:1] + input_shape[0][2:]


