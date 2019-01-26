import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import pickle
from keras.initializers import Initializer


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


"##########################  Attention Modules ##########################"


class LzSelfAttention(keras.layers.Layer):
    def __init__(self, mapping=False, **kwargs):
        super(LzSelfAttention, self).__init__(**kwargs)
        self.mapping = mapping
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        query, mask = inputs, LzComputeMasking(0)(inputs)
        if self.mapping:
            _map = keras.layers.Dense(units=int(query.shape[-1]), activation="elu", use_bias=False)
            query = keras.layers.TimeDistributed(_map)(query)
        ait = keras.layers.Dot(-1, -1)([query, query])
        a = K.exp(ait) / K.sqrt(K.cast(K.shape(query)[-1], dtype=K.floatx()))
        a *= K.expand_dims(K.cast(mask, K.floatx()), axis=-2)
        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())
        return keras.layers.dot([a, query], axes=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class LzMultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, head_count=1, mask=None, **kwargs):
        super(LzMultiHeadSelfAttention, self).__init__(**kwargs)
        self.head_count = head_count
        self.supports_masking = True
        self.mask = mask
        self.init = keras.initializers.get('glorot_uniform')

    def build(self, input_shape):
        middle_dim = int(input_shape[-1]/self.head_count)
        self.W1 = [self.add_weight((input_shape[-1], middle_dim),
                                   initializer=self.init,
                                   name='{}_W'.format(self.name))
                   for _ in range(self.head_count)]

        self.W2 = self.add_weight((middle_dim*self.head_count, input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_W'.format(self.name))

        super(LzMultiHeadSelfAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_attention(self, inputs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query, key, value = inputs, inputs, inputs
        ait = keras.layers.Dot(-1, -1)([query, key])
        a = K.exp(ait) / K.sqrt(K.cast(K.shape(query)[-1], dtype=K.floatx()))
        if self.mask is not None:
            a *= K.expand_dims(K.cast(self.mask, K.floatx()), axis=-2)
        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())
        return keras.layers.dot([a, value], axes=-2)

    def call(self, query, **kwargs):
        self.mask = LzComputeMasking(0)(query)
        projections = [K.dot(query, W) for W in self.W1]
        projections = [self.compute_attention(p) for p in projections]
        if len(projections) > 1:
            projections = keras.layers.concatenate(projections, axis=-1)
        else:
            projections = projections[0]
        return K.dot(projections, self.W2)

    def compute_output_shape(self, input_shape):
        return input_shape


class LzScaleDotAttention(keras.layers.Layer):
    def __init__(self, mapping=False, **kwargs):
        super(LzScaleDotAttention, self).__init__(**kwargs)
        self.mapping = mapping
        self.supports_masking = True

    def build(self, input_shape):
        super(LzScaleDotAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        key, value, query = inputs[0], inputs[1], inputs[2]
        mask = LzComputeMasking(0)(value)
        if self.mapping:
            _map = keras.layers.Dense(units=int(key.shape[-1]), activation="elu", use_bias=False)
            key = keras.layers.TimeDistributed(_map)(key)
            value = keras.layers.TimeDistributed(_map)(value)
            query = keras.layers.TimeDistributed(_map)(query)
        ait = keras.layers.Dot(-1, -1)([query, key])
        a = K.exp(ait) / K.sqrt(K.cast(K.shape(query)[-1], dtype=K.floatx()))
        if mask is not None:
            a *= K.expand_dims(K.cast(mask, K.floatx()), axis=-2)
        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())
        return keras.layers.dot([a, value], axes=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class LzGlobalAveragePooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LzGlobalAveragePooling, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LzGlobalAveragePooling, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, mask = inputs, LzComputeMasking(0)(inputs)
        return K.sum(value, axis=-2) / (K.sum(mask, axis=-1, keepdims=True) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class LzGlobalMaxPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LzGlobalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LzGlobalMaxPooling, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, _ = inputs
        # return keras.layers.maximum(value)
        return K.max(value, axis=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class CustomInitializer(Initializer):
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, shape, dtype=None):
        self.weights.reshape(shape)
        return tf.convert_to_tensor(self.weights, dtype=dtype)

class LzInnerSingleHeadAttentionPooling(keras.layers.Layer):


    def __init__(self, pre_weights = None, **kwargs):
        super(LzInnerSingleHeadAttentionPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.pre_weights = pre_weights

    def build(self, input_shape):
        if self.weights:
            initializer = CustomInitializer(self.weights)
        else:
            initializer = keras.initializers.uniform()
        self.att_vec = self.add_weight(shape=(input_shape[2],1),
                                       initializer=initializer,
                                       # regularizer=keras.regularizers.l2(0.01),
                                       name='{}_ATT'.format(self.name))
        super(LzInnerSingleHeadAttentionPooling, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, mask = inputs, LzComputeMasking(0)(inputs)
        ait = K.squeeze(K.dot(value, self.att_vec), axis=-1)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        return keras.layers.dot([value, a], axes=[1, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + input_shape[2:]


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


class LzExternalQueryAttentionPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LzExternalQueryAttentionPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LzExternalQueryAttentionPooling, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, query = inputs[0], inputs[1:]
        mask = LzComputeMasking(0)(value)
        weights = []
        for q in query:
            ait = keras.layers.dot([value, q], axes=-1)
            a = K.exp(ait)
            if mask is not None:
                a *= mask
            weights.append(a)
        if len(weights) > 1:
            for w in weights[1:]:
                weights[0] *= w
        weights[0] /= K.cast(K.sum(weights[0], axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        out = keras.layers.dot([value, weights[0]], axes=1)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0][:1] + input_shape[0][2:]


class LzExternalAttentionWeight(keras.layers.Layer):
    def __init__(self, reverse=False, **kwargs):
        super(LzExternalAttentionWeight, self).__init__(**kwargs)
        self.supports_masking = True
        self.reverse = reverse

    def build(self, input_shape):
        super(LzExternalAttentionWeight, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, **kwargs):
        value, query = inputs
        mask = LzComputeMasking(0)(value)
        ait = keras.layers.dot([value, query], axes=-1)
        if self.reverse:
            ait *= -1.0
        a = K.exp(ait)
        if mask is not None:
            a *= mask
        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2]


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
            assert usr_vec.shape[-1] == doc_vec.shape[-1]
            logits = keras.layers.Dot(axes=-1)(inputs)
            logits = keras.layers.Activation('sigmoid')(logits)
        return logits


"####################################   Mingxiao's Modules   ####################################"


class MxQueryAttentionMasked(keras.layers.Layer):
    def __init__(self, mask=None, hidden_size=100, **kwargs):
        super(MxQueryAttentionMasked, self).__init__(**kwargs)
        self.mask = mask
        self.hidden_size = hidden_size

    def call(self, inputs, **kwargs):
        self.mask = LzComputeMasking(0)(inputs[0])
        query_vec = keras.layers.Dense(self.hidden_size, use_bias=False)(inputs[1])
        query_vec = K.expand_dims(query_vec, axis=-2)
        attention = keras.activations.tanh(keras.layers.Dense(self.hidden_size)(inputs[0]) + query_vec)
        attention = keras.layers.Dense(1, use_bias=False)(attention)
        attention = K.squeeze(attention, axis=-1)
        attention = K.exp(attention) * self.mask
        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())
        return keras.layers.Dot((1, 1))([inputs[0], attention_weight])

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-2] + input_shape[0][-1:]


"########################################  User Encoder  ########################################"


class LzMultiHeadQueryAttentionPooling:

    def __init__(self, history_len, hidden_dim, head_count):
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.head_count = head_count

    def _build_model(self):
        docs = keras.layers.Input(shape=(self.history_len, self.hidden_dim))
        news = keras.layers.Input(shape=(self.hidden_dim,))
        transformed = []
        for _ in range(self.head_count):
            query = keras.layers.Dense(units=self.hidden_dim,
                                       # activation=keras.layers.PReLU(),
                                       # activation="relu",
                                       activation="elu",
                                       # kernel_regularizer=keras.regularizers.l2(0.01),
                                       use_bias=False)(news)
            transformed.append(LzExternalSingleHeadAttentionPooling()([docs, query]))
        user_vec = keras.layers.concatenate(transformed, axis=-1) if self.head_count > 1 \
            else transformed[0]
        user_vec = keras.layers.Dropout(0.2)(user_vec)
        user_vec = keras.layers.Dense(units=self.hidden_dim,
                                      activation="elu",
                                      # kernel_regularizer=keras.regularizers.l2(0.01),
                                      use_bias=False)(user_vec)
        return keras.Model([docs, news], user_vec)


class LzQueryAttentionPooling:
    def __call__(self, value, query, *args, **kwargs):
        mapping = keras.layers.Dense(units=int(value.shape[-1]), activation="elu", use_bias=False)
        value = keras.layers.TimeDistributed(mapping)(value)
        result = LzExternalSingleHeadAttentionPooling()([value, query])
        return result


class LzCompressUserEncoder:

    def __init__(self, history_len, hidden_dim, channel_count, enable_pretrain=False):
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.channel_count = channel_count
        if enable_pretrain:
            with open('./models/AutoEncoder_' + str(channel_count) + '.pkl', 'rb') as p:
                pre_weights_biases = pickle.load(p)
            pre_weights = pre_weights_biases[0]
            self.pool_heads = [LzInnerSingleHeadAttentionPooling(pre_weights=pre_weights[_]) for _ in range(channel_count)]
        else:
            self.pool_heads = [LzInnerSingleHeadAttentionPooling(pre_weights=None) for _ in range(channel_count)]

    def _build_model(self):
        docs = keras.layers.Input(shape=(self.history_len, self.hidden_dim))
        reshape = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))
        vectors = [reshape(head(docs)) for head in self.pool_heads]
        vectors = keras.layers.concatenate(vectors, axis=1) if self.channel_count > 1 else vectors[0]
        return keras.Model(docs, vectors)

    def set_weight(self):
        pass


class LzCompressQueryUserEncoder:

    def __init__(self, history_len, hidden_dim, head_count, channel_count, enable_pretrain):
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.channel_count = channel_count
        self.compressor = LzCompressUserEncoder(history_len, hidden_dim, channel_count, enable_pretrain)._build_model()
        self.summarizer = LzMultiHeadQueryAttentionPooling(history_len, hidden_dim, head_count)._build_model()

    def _build_model(self):
        docs = keras.layers.Input(shape=(self.history_len, self.hidden_dim))
        news = keras.layers.Input(shape=(self.hidden_dim,))
        usr_compress = self.compressor(docs)
        usr_vector = self.summarizer([usr_compress, news])
        return keras.Model([docs, news], usr_vector)


class LzQueryMapUserEncoder:

    def __init__(self, history_len, hidden_dim):
        self.history_len = history_len
        self.hidden_dim = hidden_dim

    def _build_model(self):
        docs = keras.layers.Input(shape=(self.history_len, self.hidden_dim))
        news = keras.layers.Input(shape=(self.hidden_dim,))
        query = keras.layers.Dense(units=self.hidden_dim, activation="elu", use_bias=False)(news)
        vector = LzExternalSingleHeadAttentionPooling()([docs, query])
        return keras.Model([docs, news], vector)


class LzRecentAttendPredictor:

    def __init__(self, history_len, window_len, hidden_dim, mode="non"):
        self.history_len = history_len
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.mode = mode
        assert self.mode in ["org", "pos", "neg", "both"]

    def _build_model(self):
        cutting = keras.layers.Lambda(lambda x: x[:, -self.window_len:, :])
        # K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        normalizing = keras.layers.Lambda(lambda x: x[0]*x[1]/
                      K.cast(K.sum(x[0]*x[1], axis=-1, keepdims=True) + K.epsilon(), K.floatx()))
        mapping = keras.layers.Dense(units=self.hidden_dim, activation="elu", use_bias=False)
        cat = keras.layers.concatenate

        docs, news = keras.layers.Input(shape=(self.history_len, self.hidden_dim)), \
                     keras.layers.Input(shape=(self.hidden_dim,))
        views = LzGlobalAveragePooling()(cutting(docs))
        q_news, q_views = mapping(news), mapping(views)

        w_org, w_pos, w_neg = LzExternalAttentionWeight(reverse=False)([docs, q_news]), \
                              LzExternalAttentionWeight(reverse=False)([docs, q_views]), \
                              LzExternalAttentionWeight(reverse=True)([docs, q_views])
        w_pos, w_neg = normalizing([w_org, w_pos]), normalizing([w_org, w_neg])
        usr_o, usr_p, usr_n = keras.layers.dot([docs, w_org], axes=(1, 1)), \
                              keras.layers.dot([docs, w_pos], axes=(1, 1)), \
                              keras.layers.dot([docs, w_neg], axes=(1, 1))

        # hidden_o = keras.layers.Dense(units=self.hidden_dim, activation="elu")(cat([usr_o, news], axis=-1))
        # hidden_p = keras.layers.Dense(units=self.hidden_dim, activation="elu")(cat([usr_p, news], axis=-1))
        # hidden_n = keras.layers.Dense(units=self.hidden_dim, activation="elu")(cat([usr_n, news], axis=-1))
        #
        # logit_o = keras.layers.Dense(units=1, activation="sigmoid")(hidden_o)
        # logit_p = keras.layers.Dense(units=1, activation="sigmoid")(hidden_p)
        # logit_n = keras.layers.Dense(units=1, activation="sigmoid")(hidden_n)

        logit_o = LzLogits(mode="dot")([usr_o, news])
        logit_p = LzLogits(mode="dot")([usr_p, news])
        logit_n = LzLogits(mode="dot")([usr_n, news])

        if self.mode == "pos":
            gates = keras.layers.Dense(units=2, activation="softmax")(news)
            logit = cat([logit_o, logit_p], axis=-1)
            logit = keras.layers.dot([logit, gates], axes=(-1, -1))
        elif self.mode == "neg":
            gates = keras.layers.Dense(units=2, activation="softmax")(news)
            logit = cat([logit_o, logit_n], axis=-1)
            logit = keras.layers.dot([logit, gates], axes=(-1, -1))
        elif self.mode == "both":
            gates = keras.layers.Dense(units=3, activation="softmax")(news)
            logit = cat([logit_o, logit_p, logit_n], axis=-1)
            logit = keras.layers.dot([logit, gates], axes=(-1, -1))
        else:
            logit = logit_o
        return keras.Model([docs, news], logit)


class LzMultiHeadAttentionWeight(keras.layers.Layer):
    def __init__(self, head_count, **kwargs):
        super(LzMultiHeadAttentionWeight, self).__init__(**kwargs)
        self.init = keras.initializers.get('glorot_uniform')
        self.supports_masking = True
        self.head_count = head_count

    def build(self, input_shape):
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
    
    def __init__(self, normalize=False, **kwargs):
        self.normalize = normalize
        super(LzMultiHeadAttentionWeightOrth, self).__init__(**kwargs)

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
        orth_reg = K.batch_dot(heads, K.transpose(heads))

        if self.normalize:
            norm_item = K.sqrt(K.sum(orth_reg*orth_reg, axis=-1, keepdims=True)) + K.epsilon()
            orth_reg /= norm_item

        orth_reg = K.mean(orth_reg, axis=-1, keepdims=False)
        orth_reg = K.mean(orth_reg, axis=-1, keepdims=True)

        return [keras.layers.concatenate(vectors, axis=1),
                orth_reg]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.head_count, input_shape[2]),
                (1,)]


class LzCompressionPredictor:
    def __init__(self, channel_count, mode="Post"):
        self.channel_count = channel_count
        self.mode = mode

    def __call__(self, docs, *args, **kwargs):
        hidden_dim = int(docs.shape[-1])
        mapping = keras.layers.Dense(units=hidden_dim, activation="elu", use_bias=False)
        docs = keras.layers.TimeDistributed(mapping)(docs)
        if self.mode == "Post":
            vectors, weights = LzMultiHeadAttentionWeight(self.channel_count)(docs)
            orthodox_reg = self._off_diag_norm(vectors, normalization=True)
            return vectors, weights, orthodox_reg
        else:
            vectors, orthodox_reg = LzMultiHeadAttentionWeightOrth(head_count=self.channel_count)(docs)
            return vectors, orthodox_reg

    def _off_diag_norm(self, weights, normalization=False):
        matrix = K.batch_dot(weights, K.permute_dimensions(weights, (0, 2, 1)))
        if normalization:
            matrix /= K.sqrt(K.sum(matrix*matrix, axis=-1, keepdims=True)) + K.epsilon()
        mask = K.ones_like(matrix) - K.eye(int(matrix.shape[-1]))
        matrix = matrix * mask
        # result = K.sum(matrix, axis=-1, keepdims=False)
        # result = K.sum(result, axis=-1, keepdims=True)
        "updated results"
        result = K.mean(matrix, axis=-1, keepdims=False)
        result = K.mean(result, axis=-1, keepdims=True)
        return result


if __name__ == "__main__":

    docs = keras.layers.Input(shape=(100, 200))
    news = keras.layers.Input(shape=(200,))

    model = LzCompressUserEncoder(100, 200, 4, False)._build_model()
    print(model.input_shape, model.output_shape)

    model = LzMultiHeadQueryAttentionPooling(100, 200, 4)._build_model()
    print(model.input_shape, model.output_shape)

    model = LzCompressQueryUserEncoder(100, 200, 1, 4, False)._build_model()
    print(model.input_shape, model.output_shape)
    output = model([docs, news])
    print(output.shape)

    model = LzQueryMapUserEncoder(100, 200)._build_model()
    print(model.input_shape, model.output_shape)
    output = model([docs, news])
    print(output.shape)

    output = LzExternalAttentionWeight(reverse=False)([docs, news])
    print(output.shape, docs.shape[:2])

    slicing = keras.layers.Lambda(lambda x: x[:, -3:, :])
    views = slicing(docs)
    views = LzGlobalAveragePooling()(views)
    out = LzExternalQueryAttentionPooling()([docs, views, news])
    print(out.shape, docs.shape)

    model = LzRecentAttendPredictor(100, 3, 200, "org")._build_model()
    print(model.input_shape, model.output_shape)
    print(model.summary())

    # output_tensor = model.output
    # listOfVariableTensors = model.trainable_weights
    # gradients = K.gradients(output_tensor, listOfVariableTensors)
    # from tensorflow import Session
    # sess = Session()
    # print("OK")

    print("\n-----------------------\n")

    model = LzCompressUserEncoder(100, 200, 1)._build_model()
    print(model.input_shape, model.output_shape)
    output = model(docs)
    print(output.shape)

    model = LzCompressUserEncoder(100, 200, 3)._build_model()
    print(model.input_shape, model.output_shape)
    output = model(docs)
    print(output.shape)

    print("\n--------- compress -----------\n")
    vec, wei, orth = LzCompressionPredictor(channel_count=3)(docs)
    # vec, wei = LzMultiHeadAttentionWeight(head_count=3)(docs)
    print(docs.shape[:1], 3, docs.shape[1])
    print(vec.shape, wei.shape, orth.shape, K.mean(orth))

    vec, orth = LzCompressionPredictor(channel_count=3, mode="Pre")(docs)
    print(vec.shape, orth.shape)

    print("\n--------- compress -----------\n")
    vec, orth = LzMultiHeadAttentionWeightOrth(head_count=3)(docs)
    print(vec.shape, orth.shape)
