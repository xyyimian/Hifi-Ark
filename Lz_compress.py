from sequential import *
from settings import Config
import models
import utils
import ipdb


class LzUserModeling(Seq2Vec):
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

        # ------------------------------------------- "task: compression" ------------------------------------------- #

        if "lz-compress-plus" in user_model:
            channel_count = int(user_model.split("-")[-1])
            clicked_vec, weights, orth_reg = models.LzCompressionPredictor(channel_count=channel_count)(clicked_vec)
            clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            if "lz-compress-plus" in user_model:
                self.model.add_loss(self.aux_loss(K.sum(orth_reg)))
            
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])                 
            print("this is where metric tensor is added,\n no sure whether it works...\n")
            self.model.metrics_tensors += [K.sum(orth_reg)]

            # self.model.metrics_tensors.append(K.mean(orth_reg))
            # self.model.metrics_tensors = [K.mean(orth_reg)]

        # elif "lz-compress-vanilla" in user_model:
        else:
            channel_count = int(user_model.split("-")[-1])
            usr_model = models.LzCompressQueryUserEncoder(history_len=self.config.window_size,
                                                          hidden_dim=self.config.hidden_dim,
                                                          channel_count=channel_count,
                                                          head_count=1)._build_model()
            clicked_vec = usr_model([clicked_vec, candidate_vec])
            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
        # else:
        #     if user_model != "lz-base":
        #         logging.warning('[!] arch {} not found, using average by default'.format(user_model))
        #     clicked_vec = models.LzGlobalAveragePooling()(clicked_vec)

        return self.model


if __name__ == "__main__":

    conf = Config()
    names = ["lz-compress-plus-5", "lz-vanilla-compress-5"]
    for n in names:
        conf.arch = n
        print("name: {}\n".format(n))
        model = LzUserModeling(conf)._build_model()
        if n == "lz-compress-plus":
            print(model.summary())
