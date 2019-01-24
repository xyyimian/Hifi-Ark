from sequential import *
from settings import Config
import models
import utils


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

        # ------------------------------------------- "task: dual-effect" ------------------------------------------- #

        if "eff" in user_model:

            mode = user_model.split("-")[-1]
            _model = models.LzRecentAttendPredictor(history_len=self.config.window_size,
                                                    window_len=2,
                                                    hidden_dim=self.config.hidden_dim,
                                                    mode=mode)._build_model()
            logits = _model([clicked_vec, candidate_vec])

            # if user_model == "lz-eff-org":
            #     _model = models.LzRecentAttendPredictor(history_len=self.config.window_size,
            #                                             window_len=2,
            #                                             hidden_dim=self.config.hidden_dim,
            #                                             mode="org")._build_model()
            #     logits = _model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-eff-pos":
            #     _model = models.LzRecentAttendPredictor(history_len=self.config.window_size,
            #                                             window_len=2,
            #                                             hidden_dim=self.config.hidden_dim,
            #                                             mode="pos")._build_model()
            #     logits = _model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-eff-neg":
            #     _model = models.LzRecentAttendPredictor(history_len=self.config.window_size,
            #                                             window_len=2,
            #                                             hidden_dim=self.config.hidden_dim,
            #                                             mode="neg")._build_model()
            #     logits = _model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-eff-both":
            #     _model = models.LzRecentAttendPredictor(history_len=self.config.window_size,
            #                                             window_len=2,
            #                                             hidden_dim=self.config.hidden_dim,
            #                                             mode="both")._build_model()
            #     logits = _model([clicked_vec, candidate_vec])

            # else:
            #     if user_model != "lz-base":
            #         logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            #     clicked_vec = models.LzGlobalAveragePooling()(clicked_vec)
            #     logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])

        # ------------------------------------------- "task: compression" ------------------------------------------- #

        else:

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
                # print("this is where metric tensor is added,\n no sure whether it works...\n")
                self.model.metrics_names += ['orth_reg']
                self.model.metrics_tensors += [K.sum(orth_reg)]

            else:
                if "-non" in user_model:
                    usr_model = models.LzQueryMapUserEncoder(history_len=self.config.window_size,
                                                             hidden_dim=self.config.hidden_dim)._build_model()
                    clicked_vec = usr_model([clicked_vec, candidate_vec])
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

            # if user_model == "lz-compress-1":
            #     usr_model = models.LzCompressQueryUserEncoder(history_len=self.config.window_size,
            #                                                   hidden_dim=self.config.hidden_dim,
            #                                                   channel_count=1,
            #                                                   head_count=1)._build_model()
            #     clicked_vec = usr_model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-compress-3":
            #     usr_model = models.LzCompressQueryUserEncoder(history_len=self.config.window_size,
            #                                                   hidden_dim=self.config.hidden_dim,
            #                                                   channel_count=3,
            #                                                   head_count=1)._build_model()
            #     clicked_vec = usr_model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-compress-10":
            #     usr_model = models.LzCompressQueryUserEncoder(history_len=self.config.window_size,
            #                                                   hidden_dim=self.config.hidden_dim,
            #                                                   channel_count=10,
            #                                                   head_count=1)._build_model()
            #     clicked_vec = usr_model([clicked_vec, candidate_vec])
            #
            # elif user_model == "lz-compress-non":
            #     usr_model = models.LzQueryMapUserEncoder(history_len=self.config.window_size,
            #                                              hidden_dim=self.config.hidden_dim)._build_model()
            #     clicked_vec = usr_model([clicked_vec, candidate_vec])
            # else:
            #     if user_model != "lz-base":
            #         logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            #     clicked_vec = models.LzGlobalAveragePooling()(clicked_vec)
            #
            # logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])

        # -------------------------------------------------------------------------------------------------------- #

        return self.model


if __name__ == "__main__":
    # model = LzUserModeling(Config())._build_model()
    # print(model.summary())
    # print(len(model.layers))
    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)
    #
    # model = Seq2VecForward(Config())._build_model()
    # print(model.summary())
    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)

    conf = Config()
    names = ["lz-eff-org", "lz-eff-pos", "lz-eff-neg", "lz-eff-both",
             "lz-compress-1", "lz-compress-3", "lz-compress-10",
             "lz-compress-non"]
    for n in names:
        conf.arch = n
        print("name: {}\n".format(n))
        model = LzUserModeling(conf)._build_model()
        if n == "lz-eff-org":
            # print(model.summary())
            pass
