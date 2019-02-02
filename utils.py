import json
import logging
import os
import pickle
import threading

import keras
import numpy as np
import tensorflow as tf
import models


open = tf.gfile.GFile


def logging_history(history: keras.callbacks.History):
    try:
        history = sorted(history.history.items(), key=lambda x: x[0])
        logs = ['{}: {:.4f}'.format(k, np.mean(v) if k.startswith('val') else v[-1]) for k, v in history]
        logging.info('[*] {}'.format('\t'.join(logs)))
    except:
        pass


def logging_evaluation(evaluations):
    try:
        logs = ['{}: {:.4f}'.format(k, v) for k, v in sorted(evaluations.items())]
        logging.info('[*] {}'.format('\t'.join(logs)))
    except:
        pass


def load_textual_embedding(path, dimension):
    logging.info('[+] loading embedding data from {}'.format(os.path.split(path)[-1]))

    data = {
        int(r[-2]): np.array([float(x) for x in r[-1].split(' ')], dtype=np.float32)
        for r in [s.strip().split('\t') for s in open(path)]
        if r[-1].count(' ') == dimension - 1
    }

    embedding_matrix = np.array(
        [
            data[i] if i in data else
            np.zeros(dimension, dtype=np.float32)
            for i in range(max(data.keys()) + 1)
        ]
    )

    logging.info('[-] found {} vectors from {}'.format(len(data), os.path.split(path)[-1]))
    return embedding_matrix


def load_model(paths) -> keras.Model:
    json_path, weight_path = paths
    with open(json_path, 'r') as file:
        model = keras.models.model_from_json(json.load(file), models.__dict__)
    with open(weight_path, 'rb') as file:
        model.set_weights(pickle.load(file))
    return model


def save_model(paths, model: keras.Model):
    json_path, weight_path = paths
    with open(json_path, 'w') as file:
        json.dump(model.to_json(), file)
    with open(weight_path, 'wb') as file:
        pickle.dump(model.get_weights(), file, protocol=pickle.HIGHEST_PROTOCOL)


def yuxing_save_model(paths, model: keras.Model):
    # model dump_to .pkl
    # weight dump_to .json
    # else call save to pickle, embedding->None, load update.
    pass


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    if np.sum(y_true) == 0 or np.sum(1 - y_true) == 0:
        logging.error("y_true is all zeros, can not compute auc_roc")
        import ipdb; ipdb.set_trace()
    else:
        value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        with self.lock:
            return next(self.it)


def interactive_console(local):
    import code
    code.interact(local=local)
