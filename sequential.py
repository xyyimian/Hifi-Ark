import logging

import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('../')


import document
import models
import settings
import utils
import abc


class Seq2Vec:
    class Window:
        __slots__ = ['count', 'docs', 'click_history']

        def __init__(self, docs, window_size):
            self.count = 0
            self.docs = docs
            self.click_history = [0 for _ in range(window_size)]

        def get_title(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].title if i not in elimination else self.docs[0].title for i in self.click_history])
            else:
                return np.stack([self.docs[i].title for i in self.click_history])

        def push(self, doc):
            self.click_history.append(doc)
            self.click_history.pop(0)
            self.count += 1

        @property
        def window_size(self):
            return len(self.click_history)

    class Impression:
        __slots__ = ['pos', 'neg']

        def __init__(self, d):
            d = d.split('#TAB#')
            self.pos = [int(k) for k in d[0].split(' ')]
            self.neg = [int(k) for k in d[1].split(' ')]

        def negative_samples(self, n):
            return np.random.choice(self.neg, n)

    class News:
        __slots__ = ['title', 'body']

        def __init__(self, title, body):
            self.title = title
            self.body = body

    def _extract_impressions(self, x):
        ih = [self.Impression(d) for d in x.split('#N#')
              if not d.startswith('#TAB#') and not d.endswith('#TAB#')]
        return ih

    def __init__(self, config: settings.Config):
        self.config = config
        self._load_docs()
        self._load_users()
        self._load_data()

    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        title_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape)
        )
        body_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.body_shape)
        )
        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.strip('\n').split('\t') for line in file]

        self.docs = {
            int(line[1]): self.News(
                title_parser(line[4])[0],
                body_parser(line[5])[0],
            ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body))

        logging.info("[-] loaded docs metadata")

    def _load_users(self):
        pass

    def _load_data(self):
        self.training_step = self.config.training_step
        self.validation_step = self.config.validation_step

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def valid_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2] and line[3]:
                        ih1 = self._extract_impressions(line[2])
                        ih2 = self._extract_impressions(line[3])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih1:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih2:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.title, 0

        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = ch.get_title()
                        yield list(__gen__(clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    @property
    def train(self):
        pool = []
        size = self.config.batch_size * 100
        gen = self.train_gen()
        while True:
            pool.append(next(gen))
            if len(pool) >= size:
                np.random.shuffle(pool)
                batch = [np.stack(x) for x in zip(*pool[:self.config.batch_size])]
                yield batch[:-1], batch[-1]
                pool = pool[self.config.batch_size:]

    @property
    def valid(self):
        gen = self.valid_gen()
        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))]
            yield batch[:-1], batch[-1]

    @property
    def test(self):
        for b in self.test_gen():
            batch = [np.stack(x) for x in zip(*b)]
            yield [self.model.predict(batch[:-1]).reshape(-1), batch[-1]]

    def build_model(self, epoch):
        if epoch == 0:
            self._build_model()
        return self.model

    def loss(self, y_true, y_pred):
        return -0.5 * (1 + self.config.negative_samples) * tf.reduce_mean(
            y_true * tf.log(y_pred + 1e-8) * self.config.gain +
            (1 - y_true) * tf.log(1 - y_pred + 1e-8) / self.config.negative_samples)

    def aux_loss(self, reg_item):
        return tf.reduce_mean(reg_item) * K.abs(self.config.l2_norm_coefficient)

    def get_doc_encoder(self):
        if self.config.enable_pretrain_encoder:
            encoder = utils.load_model(self.config.encoder_input)
            if not self.config.pretrain_encoder_trainable:
                encoder.trainable = False
            return encoder
        else:
            if self.config.debug:
                title_embedding = np.load(self.config.title_embedding_input + '.npy')
            else:
                title_embedding = utils.load_textual_embedding(self.config.title_embedding_input,
                                                               self.config.textual_embedding_dim)
            title_embedding_layer = keras.layers.Embedding(
                *title_embedding.shape,
                input_length=self.config.title_shape,
                weights=[title_embedding],
                trainable=self.config.textual_embedding_trainable,
            )
            return models.ca(
                self.config.title_shape,
                self.config.title_filter_shape,
                title_embedding_layer,
                self.config.dropout,
                self.config.user_embedding_dim
            )

    @abc.abstractmethod
    def _build_model(self):
        doc_encoder = self.get_doc_encoder()
        user_encoder = keras.layers.TimeDistributed(doc_encoder)

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        clicked_vec = user_encoder(clicked)
        candidate_vec = doc_encoder(candidate)

        mask = models.LzComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, mask])

        user_model = self.config.arch
        if user_model == 'att':
            clicked_vec = models.SimpleAttentionMasked(mask)(clicked_vec)
        elif user_model == 'gru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(clicked_vec)
        elif user_model == 'qatt':
            clicked_vec = models.QueryAttentionMasked(mask)([clicked_vec, candidate_vec])
        elif user_model == 'nqatt':
            clicked_vec = models.QueryAttentionMasked(mask)(clicked_vec)
            clicked_vec = keras.layers.BatchNormalization()(clicked_vec)
        else:
            if user_model != 'avg':
                logging.warning('[!] arch not found, using average by default')
            clicked_vec = keras.layers.GlobalAveragePooling1D()(clicked_vec)

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([clicked, candidate], logits)
        if self.__class__ == Seq2VecForward:
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.config.learning_rate),
                loss=self.loss,
                metrics=[utils.auc_roc]
            )
        else:
            return self.model

        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

    def callback(self, epoch):
        # metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
        # keras.backend.get_session().run(tf.initializers.variables(metric_vars))

        # keras.backend.set_value(self.model.optimizer.lr, keras.backend.get_value(self.model.optimizer.lr) *
        #                         self.config.learning_rate_decay)

        if epoch or True:
            def __gen__():
                for y_pred, y_true in self.test:
                    auc = roc_auc_score(y_true, y_pred)
                    ndcgx = utils.ndcg_score(y_true, y_pred, 10)
                    ndcgv = utils.ndcg_score(y_true, y_pred, 5)
                    mrr = utils.mrr_score(y_true, y_pred)
                    pos = np.sum(y_true)
                    size = len(y_true)
                    yield auc, ndcgx, ndcgv, mrr, pos, size

            values = [np.mean(x) for x in zip(*__gen__())]
            utils.logging_evaluation(dict(auc=values[0], ndcgx=values[1], ndcgv=values[2], mrr=values[3]))
            utils.logging_evaluation(dict(pos=values[4], size=values[5]))

    def save_model(self):
        logging.info('[+] saving models')
        utils.save_model(self.config.model_output, self.model)
        logging.info('[-] saved models')


class Seq2VecForward(Seq2Vec):
    def _build_model(self):
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder = keras.layers.TimeDistributed(doc_encoder)

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        clicked_vec = user_encoder(clicked)
        candidate_vec = doc_encoder(candidate)

        mask = models.LzComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, mask])

        user_model = self.config.arch
        logging.info('[!] Selecting User Model: {}'.format(user_model))
        if user_model == 'att':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = models.Attention()(clicked_vec)
        elif user_model == 'gru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(clicked_vec)
        elif user_model == 'cnn':
            clicked_vec = keras.layers.Conv1D(*self.config.title_filter_shape, padding='same', activation='relu')(
                clicked_vec)
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)
        elif user_model == 'catt':
            clicked_vec = keras.layers.Conv1D(*self.config.title_filter_shape, padding='same', activation='relu')(
                clicked_vec)
            clicked_vec = models.SimpleAttentionMasked(mask)(clicked_vec)
        elif user_model == 'qatt':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = models.QueryAttention()([clicked_vec, candidate_vec])
        elif user_model == 'satt':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = models.SelfAttention()(clicked_vec)
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)
        elif user_model == 'lstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.config.user_embedding_dim)(clicked_vec)
        elif user_model == 'lz1':
            input_shape = clicked_vec.get_shape()[-1].value

            def auto_attend():
                docs = keras.layers.Input(shape=(self.config.window_size, input_shape))
                cross_product = keras.layers.dot([docs, docs], axes=2)
                cross_weights = keras.layers.Softmax(axis=2)(cross_product)
                attended_docs = keras.layers.dot([cross_weights, docs], axes=1)
                return keras.Model(docs, attended_docs)

            def docs_pool():
                docs = keras.layers.Input(shape=(self.config.window_size, self.config.hidden_dim))
                pool_vec = keras.layers.Dense(units=1)
                weights = keras.layers.TimeDistributed(pool_vec)(docs)
                squeeze = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=2))
                weights = squeeze(weights)
                output = keras.layers.dot([docs, weights], axes=1)
                return keras.Model(docs, output)

            def docs_rnn():
                docs = keras.layers.Input(shape=(self.config.window_size, self.config.hidden_dim))
                gru = keras.layers.GRU(units=self.config.hidden_dim, activation="relu")
                output = gru(docs)
                return keras.Model(docs, output)

            def user_encode():
                raw_docs = keras.layers.Input(shape=(self.config.window_size, input_shape))
                att_docs = auto_attend()(raw_docs)
                ful_docs = keras.layers.concatenate([raw_docs, att_docs])
                ful_docs = keras.layers.Dense(units=self.config.hidden_dim)(ful_docs)
                pooling = docs_pool()(ful_docs)
                temporal = docs_rnn()(ful_docs)
                output = keras.layers.concatenate([pooling, temporal])
                return keras.Model(raw_docs, output)

            clicked_vec = user_encode()(clicked_vec)
        else:
            logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            clicked_vec = models.LzGlobalAveragePooling(name="user_vec")([clicked_vec, mask])

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )
        return self.model


class Seq2VecLongEncoder(Seq2VecForward):
    def _build_model(self):
        super(Seq2VecLongEncoder, self)._build_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih:
                            for pos in impression.pos:
                                clicked = ch.get_title([pos])
                                yield clicked, self.docs[pos].title, 1
                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].title, 0


class Seq2VecLong(Seq2Vec):
    def _negative_sample(self, queue_size=100000):
        while True:
            yield from np.random.choice(self.doc_count, queue_size, p=self.doc_freq)

    def _load_users(self):
        super(Seq2VecLong, self)._load_users()
        logging.info('[+] constructing user mapping')
        self.pairs = []
        self.user_mapping = {}
        self.doc_freq = np.zeros(self.doc_count)
        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user_idx = len(self.user_mapping)
                self.user_mapping[line[0] + line[1]] = user_idx
                if line[2]:
                    ih = self._extract_impressions(line[2])
                    for impression in ih:
                        for pos in impression.pos:
                            self.doc_freq[pos] += 1
                            self.pairs.append([user_idx, pos] +
                                              list(impression.negative_samples(self.config.negative_samples -
                                                                               self.config.nonlocal_negative_samples)))
                if user_idx % 10000 == 9999:
                    logging.info('[!] loaded {} users'.format(user_idx + 1))
        self.pairs = np.array(self.pairs)
        self.doc_freq = self.doc_freq ** 0.75
        self.doc_freq = self.doc_freq / np.sum(self.doc_freq)
        logging.info('[-] constructed user mapping')

    @property
    def train(self):
        def __gen__():
            sampler = self._negative_sample()

            while True:
                np.random.shuffle(self.pairs)
                for pair in self.pairs:
                    u, d = pair[:2]
                    yield u, self.docs[d].title, 1
                    for neg in pair[2:]:
                        yield u, self.docs[neg].title, 0
                    for _ in range(self.config.nonlocal_negative_samples):
                        yield u, self.docs[next(sampler)].title, 0

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))]
            yield batch[:-1], batch[-1]

    @property
    def valid(self):
        def __gen__():
            while True:
                with open(self.config.training_data_input) as file:
                    for line in file:
                        line = line.strip('\n').split('\t')
                        user = line[0] + line[1]
                        user_idx = self.user_mapping[user]
                        if line[2] and line[3]:
                            ih2 = self._extract_impressions(line[3])
                            for impression in ih2:
                                for pos in impression.pos:
                                    yield user_idx, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_idx, self.docs[neg].title, 0

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))]
            yield batch[:-1], batch[-1]

    @property
    def test(self):
        def __gen__(_user, _user_idx, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user_idx, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user_idx, doc.title, 0

        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user = line[0] + line[1]
                user_idx = self.user_mapping[user]
                if line[3]:
                    ih2 = self._extract_impressions(line[3])
                    for impression in ih2:
                        batch = [np.stack(x) for x in zip(*__gen__(user, user_idx, impression))]
                        yield [self.model.predict([batch[0], batch[1]]).reshape(-1), batch[2]]

    def _build_model(self):
        doc_encoder = self.get_doc_encoder()
        user_embedding_layer = keras.layers.Embedding(
            len(self.user_mapping),
            self.config.user_embedding_dim)

        user_idx = keras.Input((1,))
        candidate = keras.Input((self.config.title_shape,))

        user_vec = keras.layers.Reshape((-1,))(user_embedding_layer(user_idx))
        candidate_vec = doc_encoder(candidate)
        candidate_vec = keras.layers.Dense(self.config.user_embedding_dim)(candidate_vec)

        dist = keras.layers.Dot(-1, -1)([user_vec, candidate_vec])
        logits = keras.layers.Activation(keras.activations.sigmoid)(dist)

        self.model = keras.Model([user_idx, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )
