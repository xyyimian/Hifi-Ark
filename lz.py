from sequential import *
from settings import Config
import models
import utils



class UserModeling(Seq2Vec):
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
        logging.info('[!] Selected User Model: {}'.format(user_model))

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
            

        elif "pretrain-preplus" in user_model:

            logging.info("pretrain-preplus")
            channel_count = int(user_model.split("-")[-1])
            self.config.enable_pretrain_attention = True
            if "self" in user_model:
                x_click_vec = models._LzSelfAttention(mapping=True)(clicked_vec)
                clicked_vec = keras.layers.Average()([clicked_vec, x_click_vec])
            clicked_vec, orth_reg = models.LzCompressionPredictor(channel_count=channel_count, mode="Pre",
                                                                     enable_pretrain_attention=True)(clicked_vec)
            with tf.name_scope('orth_reg_tensor'):
                orth_reg = orth_reg[0]
                tf.summary.scalar('orthreg',orth_reg)

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
        else:
            raise Exception("No available models. Please check param!")

            logits = models.LzLogits(mode="dot")([clicked_vec, candidate_vec])
            self.model = keras.Model([clicked, candidate], logits)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate, clipnorm=5.0),
                               loss=self.loss,
                               metrics=[utils.auc_roc])            
        return self.model


