import keras
import tensorflow as tf
from settings import Config
import utils
from sequential import *
from keras.callbacks import TensorBoard
import settings
import sys
import numpy as np
from keras import backend as K
import logging
import pickle
from keras.constraints import Constraint



input_dim = 200
hidden_dim = 10

class ZeroConstraints(Constraint):
    def __call__(self, w):
        w *= 0
        return w

class XyAutoEncoder(Seq2Vec):
    def __init__(self, config, input_dim, hidden_dim):
        super().__init__(config)
        self.doc_encoder = self.get_doc_encoder()
        self.doc_encoder._make_predict_function()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config.enable_pretrain_encoder=True
        self.config.pretrain_encoder_trainable=False
        self.batch_size = config.batch_size



    def _build_model(self):

        docs = keras.Input((self.input_dim, ))
        hidden_vec = keras.layers.Dense(hidden_dim, activation = "sigmoid")(docs)
        # output_vec = keras.layers.Dense(input_dim, bias_constraint=ZeroConstraints())(hidden_vec)
        output_vec = keras.layers.Dense(input_dim, use_bias = False)(hidden_vec)
        self.model = keras.Model(docs, output_vec)
        optimizer = keras.optimizers.Adam(lr=0.01, decay=1e-3)
        self.model.compile(optimizer = optimizer,
                            loss = 'mean_squared_error',
                            metrics=[keras.metrics.mse])
        return self.model

    def __gen__(self):
        docs = self.docs
        for doc in docs.values():
            yield doc.title
    def gen(self):
        while True:
            g = self.__gen__()
            while True:
                try:
                    batch = np.asarray([next(g) for _ in range(self.batch_size)])
                    embedding = self.doc_encoder.predict(batch)
                    yield embedding, embedding
                except:
                    break


def args_parser(args):
    args = dict(a.split("=") for a in args)
    # assert "models" in args.keys()
    # args["models"] = args["models"].split(",")
    if "rounds" in args.keys():
        args["rounds"] = int(args["rounds"])
    if "epochs" in args.keys():
        args["epochs"] = int(args["epochs"])
    return args


def train(config):


    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(config.log_output),
        logging.StreamHandler()]
    )

    UM = XyAutoEncoder(config, input_dim, hidden_dim)
    doc_generator = UM.gen()
    tbCallBack = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_graph = True, write_images = True, update_freq = 10000)
    for i in range(config.rounds):
        print('start round: %d' %i)
        # for epoch in range(config.epochs):
        model = UM._build_model()
        print(model.summary())#to reset generator for each epoch
        # print('start epoch: %d' %epoch)
        history = model.fit_generator(
            generator = doc_generator,
            epochs = config.epochs,
            steps_per_epoch = UM.doc_count / config.batch_size,
            # epochs = epoch + 1,
            # initial_epoch = epoch,
            verbose = 1,
            callbacks = [tbCallBack]
        )
        weightsAndBiases = model.layers[2].get_weights()
        with open('./models/AutoEncoder_'+str(hidden_dim)+'.pkl', 'wb') as p:
            pickle.dump(weightsAndBiases, p)
        UM.save_model()
        # K.clear_session()
    #valid
    for i in range(3):
        embedding = next(doc_generator)[0]
        print(embedding)
        print(model.predict(embedding))
        print()


if __name__ == '__main__':

    args = args_parser(sys.argv[1:])
    print(args)
    config = settings.Config(rounds=args["rounds"], epochs=args["epochs"])
    train(config=config)







