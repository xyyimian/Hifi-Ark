from sequential import *
from settings import Config
import models
import utils


class LzUserModelingOrigin(Seq2VecForward):
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
            # logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

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

            # ------------   Orthogonal regularization is added to compression vectors   ------------

            if "lz-compress-plus" in user_model:
                channel_count = int(user_model.split("-")[-1])
                clicked_vec, weights, orth_reg = models.LzCompressionPredictor(channel_count=channel_count)(clicked_vec)
                clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

                # adjust the orthogonal regularization coefficient
                # self.config.l2_norm_coefficient /= (channel_count/3.0)**0.5

                logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
                self.model = keras.Model([clicked, candidate], logits)
                if "mean" in user_model:
                    self.config.l2_norm_coefficient = 0.1
                    # adjust the orthogonal regularization coefficient
                    self.model.add_loss(self.aux_loss(orth_reg * (channel_count/3.0)**0.75))
                    # self.model.add_loss(self.aux_loss(K.mean(orth_reg) * (channel_count/3.0)**0.75))
                else:
                    # adjust the orthogonal regularization coefficient
                    self.model.add_loss(self.aux_loss(K.sum(orth_reg) * (channel_count/3.0)**0.75))

                self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                                   loss=self.loss,
                                   metrics=[utils.auc_roc])
                # print("this is where metric tensor is added,\n no sure whether it works...\n")
                self.model.metrics_names += ['orth_reg']
                if "mean" in user_model:
                    self.model.metrics_tensors += [K.mean(orth_reg) * (channel_count/3.0)**0.75]
                else:
                    self.model.metrics_tensors += [K.sum(orth_reg) * (channel_count/3.0)**0.75]

            # ------------   Orthogonal regularization is added to pooling vectors   ------------

            elif "lz-compress-pre-plus" in user_model:
                channel_count = int(user_model.split("-")[-1])
                clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(clicked_vec)
                orth_reg = orth_reg[0]
                clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

                # adjust the orthogonal regularization coefficient
                # self.config.l2_norm_coefficient /= (channel_count / 3.0) ** 0.75

                logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
                self.model = keras.Model([clicked, candidate], logits)
                self.config.l2_norm_coefficient = 0.1

                # adjust the orthogonal regularization coefficient
                self.model.add_loss(self.aux_loss(orth_reg * (channel_count/3.0)**0.75))

                self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                                   loss=self.loss,
                                   metrics=[utils.auc_roc])
                self.model.metrics_names += ['orth_reg']
                self.model.metrics_tensors += [orth_reg]

            # -------------   Vanilla compression, use randomized pooling vectors   -------------

            else:
                if "-non" in user_model:
                    usr_model = models.LzQueryMapUserEncoder(history_len=self.config.window_size,
                                                             hidden_dim=self.config.hidden_dim)._build_model()
                    clicked_vec = usr_model([clicked_vec, candidate_vec])
                else:
                    channel_count = int(user_model.split("-")[-1])
                    #----------pretrain+preplus
                    self.config.enable_pretrain_attention = True
                    clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre", enable_pretrain_attention = self.config.enable_pretrain_attention)(clicked_vec)
                    orth_reg = orth_reg[0]
                    clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)                    




                    # clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(clicked_vec)
                    # orth_reg = orth_reg[0]                    
                    # usr_model = models.LzCompressQueryUserEncoder(history_len=self.config.window_size,
                    #                                               hidden_dim=self.config.hidden_dim,
                    #                                               channel_count=channel_count,
                    #                                               head_count=1,
                    #                                               enable_pretrain=self.config.enable_pretrain_attention)._build_model()
                    # clicked_vec = usr_model([clicked_vec, candidate_vec])

                logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
                self.model = keras.Model([clicked, candidate], logits)
                self.config.l2_norm_coefficient = 0.1

                # adjust the orthogonal regularization coefficient
                self.model.add_loss(self.aux_loss(orth_reg * (channel_count/3.0)**0.75))                
                self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                                   loss=self.loss,
                                   metrics=[utils.auc_roc])
                self.model.metrics_names += ['orth_reg']
                self.model.metrics_tensors += [orth_reg]                                   

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


class LzUserModelingSelf(LzUserModelingOrigin):
    # "----------------------------   with self attention added as extra dimensions   ----------------------------" #
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

        # ------------------------------------------    Task Compression    ------------------------------------------ #

        if "lz-compress-plus" in user_model:
            channel_count = int(user_model.split("-")[-1])

            o_click_vec, weights, orth_reg = models.LzCompressionPredictor(channel_count=channel_count)(clicked_vec)
            o_click_vec = models.LzQueryAttentionPooling()(o_click_vec, candidate_vec)

            x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
            x_click_vec, weights, _ = models.LzCompressionPredictor(channel_count=channel_count)(x_click_vec)
            x_click_vec = models.LzQueryAttentionPooling()(x_click_vec, candidate_vec)

            # ----------   uncommented for mlp   ---------- #
            # clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=-1)
            # logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])

            dim_expand = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))
            o_click_vec, x_click_vec = dim_expand(o_click_vec), dim_expand(x_click_vec)

            clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=1)
            clicked_vec = keras.layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(clicked_vec)
            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            if "mean" in user_model:
                self.config.l2_norm_coefficient = 0.1
                self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
            else:
                self.model.add_loss(self.aux_loss(K.sum(orth_reg) * (channel_count / 3.0) ** 0.75))

            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])

            self.model.metrics_names += ['orth_reg']
            if "mean" in user_model:
                self.model.metrics_tensors += [K.mean(orth_reg) * (channel_count / 3.0) ** 0.75]
            else:
                self.model.metrics_tensors += [K.sum(orth_reg) * (channel_count / 3.0) ** 0.75]

        # ------------   Orthogonal regularization is added to pooling vectors   ------------

        elif "lz-compress-pre-plus" in user_model:
            channel_count = int(user_model.split("-")[-1])
            o_click_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(clicked_vec)
            orth_reg = orth_reg[0]
            o_click_vec = models.LzQueryAttentionPooling()(o_click_vec, candidate_vec)

            x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
            x_click_vec, _ = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(x_click_vec)
            x_click_vec = models.LzQueryAttentionPooling()(x_click_vec, candidate_vec)

            # ----------   uncommented for mlp   ---------- #
            # clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=-1)
            # logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])

            dim_expand = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))
            o_click_vec, x_click_vec = dim_expand(o_click_vec), dim_expand(x_click_vec)
            clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=1)
            clicked_vec = keras.layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(clicked_vec)

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            self.config.l2_norm_coefficient = 0.1
            self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
            self.model.metrics_names += ['orth_reg']
            self.model.metrics_tensors += [orth_reg]

        # -------------   Vanilla compression, use randomized pooling vectors   -------------

        else:
            if "-non" in user_model:
                usr_model = models.LzQueryMapUserEncoder(history_len=self.config.window_size,
                                                         hidden_dim=self.config.hidden_dim)._build_model()
                o_click_vec = usr_model([clicked_vec, candidate_vec])
                x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
                x_click_vec = usr_model([x_click_vec, candidate_vec])
            else:
                channel_count = int(user_model.split("-")[-1])
                self.config.enable_pretrain_attention = True
                o_click_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre",
                                                                      enable_pretrain_attention=self.config.enable_pretrain_attention)(clicked_vec)
                o_click_vec = models.LzQueryAttentionPooling()(o_click_vec, candidate_vec)
                orth_reg = orth_reg[0]

                x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
                x_click_vec, _ = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre",
                                                               enable_pretrain_attention=self.config.enable_pretrain_attention)(x_click_vec)
                x_click_vec = models.LzQueryAttentionPooling()(x_click_vec, candidate_vec)

            # -------------------   uncommented for the use of mlp   -------------------
            # clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=-1)
            # logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])

            dim_expand = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))
            o_click_vec, x_click_vec = dim_expand(o_click_vec), dim_expand(x_click_vec)
            clicked_vec = keras.layers.concatenate([o_click_vec, x_click_vec], axis=1)
            clicked_vec = keras.layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(clicked_vec)
            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            self.config.l2_norm_coefficient = 0.1

            self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
            self.model.metrics_names += ['orth_reg']
            self.model.metrics_tensors += [orth_reg]

        return self.model


class LzUserModelingSelfPre(LzUserModelingOrigin):
    # "----------------------------   with self attention added as extra dimensions   ----------------------------" #
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

        # ------------------------------------------    Task Compression    ------------------------------------------ #
        if "pre-train" in user_model:
            channel_count = int(user_model.split("-")[-1])
            clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count,
                                                                           mode="pretrain",
                                                                           enable_pretrain_attention=True)(clicked_vec)
            clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)

            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
        # if "lz-compress-plus" in user_model:
        #     logging.info("post")
        #     channel_count = int(user_model.split("-")[-1])
        #
        #     x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
        #     clicked_vec = keras.layers.Average()([clicked_vec, x_click_vec])
        #
        #     clicked_vec, weights, orth_reg = models.LzCompressionPredictor(channel_count=channel_count)(clicked_vec)
        #     orth_reg = orth_reg[0]
        #     clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)
        #
        #     logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
        #
        #     self.model = keras.Model([clicked, candidate], logits)
        #     if "mean" in user_model:
        #         self.config.l2_norm_coefficient = 0.1
        #         self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
        #     else:
        #         self.model.add_loss(self.aux_loss(K.sum(orth_reg) * (channel_count / 3.0) ** 0.75))
        #
        #     self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
        #                        loss=self.loss,
        #                        metrics=[utils.auc_roc])
        #
        #     self.model.metrics_names += ['orth_reg']
        #     if "mean" in user_model:
        #         self.model.metrics_tensors += [K.mean(orth_reg) * (channel_count / 3.0) ** 0.75]
        #     else:
        #         self.model.metrics_tensors += [K.sum(orth_reg) * (channel_count / 3.0) ** 0.75]

        # ------------   Orthogonal regularization is added to pooling vectors   ------------

        elif "pre-plus" in user_model:
            logging.info("preplus")
            channel_count = int(user_model.split("-")[-1])
            if "self" in user_model:
                x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
                clicked_vec = keras.layers.Average()([clicked_vec, x_click_vec])

            clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(clicked_vec)
            orth_reg = orth_reg[0]
            clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            self.config.l2_norm_coefficient = 0.1
            self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                            loss=self.loss,
                            metrics=[utils.auc_roc])
            self.model.metrics_names += ['orth_reg']
            self.model.metrics_tensors += [orth_reg]
        # -------------   dimension augmentation   -------------            
        elif "dim-aug" in user_model:
            logging.info("dim-aug")
            # channel_count = int(user_model.split("-")[-1])
            # expand hidden_dim for comparision
            # x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
            # clicked_vec = keras.layers.Average()([clicked_vec, x_click_vec])

            # clicked_vec, _ = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre")(clicked_vec)           
            clicked_vec = models.LzInnerSingleHeadAttentionPooling()(clicked_vec)
            # clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)
            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                            loss=self.loss,
                            metrics=[utils.auc_roc])
            

        # -------------   Vanilla compression, use randomized pooling vectors   -------------

        elif "pretrain-preplus" in user_model:

            logging.info("pretrain-preplus")
            channel_count = int(user_model.split("-")[-1])
            self.config.enable_pretrain_attention = True
            if "self" in user_model:
                x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
                clicked_vec = keras.layers.Average()([clicked_vec, x_click_vec])
            clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre",
                                                                  enable_pretrain_attention=self.config.enable_pretrain_attention)(clicked_vec)
            orth_reg = orth_reg[0]
            clicked_vec = models.LzQueryAttentionPooling()(clicked_vec, candidate_vec)

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])

            self.model = keras.Model([clicked, candidate], logits)
            self.config.l2_norm_coefficient = 0.1

            self.model.add_loss(self.aux_loss(orth_reg * (channel_count / 3.0) ** 0.75))
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
            self.model.metrics_names += ['orth_reg']
            self.model.metrics_tensors += [orth_reg]
        elif "DIN" in user_model:
            logging.info("DIN")
            usr_model = models.LzQueryMapUserEncoder(history_len=self.config.window_size,
                                                     hidden_dim=self.config.hidden_dim)._build_model()
            clicked_vec = usr_model([clicked_vec, candidate_vec])
            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                                loss=self.loss,
                                # metrics=[utils.auc_roc],
                               metrics="mean_squared_error")
        elif "base-compress" in user_model:
            head_count=int(user_model.split('-')[-1])
            user_model = models.LzBaseCompress(head_count=head_count)
            clicked_vec = user_model(clicked_vec)
            logits = models.LzLogits(mode="mlp")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])
        else:

            if user_model == 'gru':
                clicked_vec = keras.layers.GRU(self.config.hidden_dim)(clicked_vec)
            elif user_model == 'cnn':
                clicked_vec = keras.layers.Conv1D(self.config.hidden_dim, 3, padding='same', activation='relu')(
                    clicked_vec)
                clicked_vec = keras.layers.Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(clicked_vec)
                # clicked_vec = keras.layers.Average()([clicked_vec[:, i, :] for i in range(clicked_vec.shape[1])])
            else:
                raise Exception("No available models. Please check param!")

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])            
        return self.model


# LzUserModeling = LzUserModelingSelf
class LzUserModeling(LzUserModelingSelfPre):
    pass


if __name__ == "__main__":

    conf = Config()
    names = ["lz-eff-org", "lz-eff-pos", "lz-eff-neg", "lz-eff-both",
             "lz-compress-1", "lz-compress-3", "lz-compress-10",
             "lz-compress-non", "lz-compress-pre-plus-5"]
    names_ = ["self-lz-compress-plus-3",
              "self-lz-compress-pre-plus-3",
              "self-lz-compress-pre-train-3"]
    for n in names_:
        conf.arch = n
        print("name: {}\n".format(n))
        model = LzUserModeling(conf)._build_model()
        if n == "lz-eff-org":
            # print(model.summary())
            pass
