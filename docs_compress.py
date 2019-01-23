import keras
import tensorflow as tf
import keras.backend as K

import logging
import numpy as np

import document
import settings
import utils


class DocCompressor:

    def __init__(self, config: settings.Config):
        self.config = config
        self._load_docs()
        self._load_data()

    class News:
        def __init__(self, title, body):
            self.title = title
            self.body = body

    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        print("[+] loading docs metadata")
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

    def _load_data(self):
        self.training_step = int(np.ceil(self.doc_count/self.config.batch_size))

    def get_doc_encoder(self):
        encoder = utils.load_model(self.config.encoder_input)
        encoder.trainable = False
        return encoder

    def train(self):
        def __gen__(docs):
            for doc in docs.values():
                yield doc.title, 0
        gen = __gen__(self.docs)
        while True:
            yield [next(gen) for _ in range(self.config.batch_size)]

    def _build_model(self):
        doc_encoder = self.get_doc_encoder()
        news = keras.layers.Input(shape=(self.config.title_shape), dtype="int32")
        vector = doc_encoder(news)
        hidden = keras.layers.Dense(units=self.config.compression_dim, activation="relu")(vector)
        output = keras.layers.Dense(units=self.config.title_shape, activation="relu")(hidden)
        output = K.mean(K.sqrt(output-vector), axis=-1)
        self.model = keras.Model(news, output)

    def construct(self):
        pass

    def save_model(self):
        # self.model.save("models/compressor.pkl")
        pass


if __name__ == "__main__":

    d_com = DocCompressor(settings.Config())
    generator = d_com.train()
    count = 0
    for g in generator:
        print(type(g), len(g), g[0][0].shape, g[0][1])
        count += 1
        if count >= 10:
            break






