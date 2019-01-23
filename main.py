import logging
import sys
import abc
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import document
import models
import settings
import utils
# from Lz_compress import LzUserModeling
from lz import LzUserModeling
from test import RunUserModel


def train(config):

    # "1. log config"
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(config.log_output),
                  logging.StreamHandler()]
        )

    # "2. training loop: data, model and process"
    UM = LzUserModeling(config)
    training_data = UM.train
    for i in range(config.rounds):
        logging.info("launching the {} round of {}".format(i, m))
        for epoch in range(config.epochs):
            logging.info('[+] start epoch {}'.format(epoch))
            model = UM.build_model(epoch)
            history = model.fit_generator(
                training_data,
                config.training_step,
                epochs=epoch + 1,
                initial_epoch=epoch,
                callbacks=None,
                verbose=1 if config.debug and not config.background else 2)
            utils.logging_history(history)
            if hasattr(UM, 'callback'):
                UM.callback(epoch)
            try:
                evaluations = model.evaluate_generator(UM.valid,
                                                       config.validation_step,
                                                       verbose=1 if config.debug and not config.background else 2)
                utils.logging_evaluation(dict(zip(model.metrics_names, evaluations)))
            except:
                pass
            if hasattr(UM, 'callback_valid'):
                UM.callback_valid(epoch)
            logging.info('[-] finish epoch {}'.format(epoch))

        # "3. save model to .json and .pkl"
        print("saving model to file ...")
        UM.save_model()
        K.clear_session()
    return 0


def users(config: settings.Config):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(config.log_output),
                  logging.StreamHandler()]
        )
    UM = RunUserModel(config)
    UM.save_result()
    return 0


@abc.abstractmethod
def score(config: settings.Config):
    pass


@abc.abstractmethod
def evaluate(config: settings.Config):
    pass


def args_parser(args):
    args = dict(a.split("=") for a in args)
    assert "models" in args.keys()
    args["models"] = args["models"].split(",")
    if "rounds" in args.keys():
        args["rounds"] = int(args["rounds"])
    if "epochs" in args.keys():
        args["epochs"] = int(args["epochs"])
    return args


if __name__ == "__main__":

    # logging.basicConfig(
    #     format='%(asctime)s : %(levelname)s : %(message)s',
    #     level=logging.INFO,
    #     handlers=[logging.FileHandler(settings.Config().log_output),
    #               logging.StreamHandler()]
    # )

    args = args_parser(sys.argv[1:])
    print(args)
    # logging.info(sys.argv[1:])

    for m in args["models"]:
        config = settings.Config(rounds=args["rounds"], epochs=args["epochs"], arch=m, name=m)
        train(config=config)
