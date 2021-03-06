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
from lz import UserModeling


def train(config):

    # "1. log config"
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(config.log_output),
                  logging.StreamHandler()]
        )

    # "2. training loop: data, model and process"
    
    UM = UserModeling(config)
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
            try:
                evaluations = model.evaluate_generator(UM.valid,
                                                       config.validation_step,
                                                       verbose=1 if config.debug and not config.background else 2)
                assert len(evaluations) == len(model.metrics_names)                                             
                utils.logging_evaluation(dict(zip(model.metrics_names, evaluations)))
            except:
                pass
            if hasattr(UM, 'callback_valid'):
                UM.callback_valid(epoch)
            logging.info('[-] finish epoch {}'.format(epoch))
        if hasattr(UM, 'callback'):
            UM.callback(epoch)
        # "3. save model to .json and .pkl"
        print("saving model to file ...")
        UM.save_model()
        # K.clear_session()
    return 0

def test(config):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(config.log_output),
                  logging.StreamHandler()]
        )

    print('start testing')
    model = utils.load_model(config.model_output)
    UM = UserModeling(config)
    UM.model = model
    UM.callback(1)



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


    import os
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./models'):
        os.mkdir('./models')

    args = args_parser(sys.argv[1:])

    for m in args["models"]:
        config = settings.Config(rounds=args["rounds"], epochs=args["epochs"], arch=m, name=m)
        train(config=config)
