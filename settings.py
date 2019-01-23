import os


class Config:

    def __init__(self,
                 input_training_data_path="data/",
                 input_validation_data_path="data/",
                 input_previous_model_path="models/",
                 output_model_path="models/",
                 log_dir="log/",
                 textual_embedding_trainable=False,
                 enable_baseline=True,
                 enable_pretrain_encoder=False, #True,
                 epochs=3,
                 batch_size=200,            #100,
                 rounds=1,
                 gain=1.0,
                 dropout=0.2,
                 learning_rate=0.001,
                 learning_rate_decay=0.2,
                 l2_norm_coefficient=0.02,
                 training_step=11801,       #10000,
                 validation_step=50,        #1000,
                 testing_impression=63793,  #1000,
                 user_embedding_dim=200,
                 compression_dim=5,
                 textual_embedding_dim=300,
                 title_filter_shape=(400, 3),
                 attention_head_count=4,
                 title_shape=20,
                 body_shape=200,
                 window_size=100,
                 recent_window_size=20,
                 hidden_dim=200,
                 negative_samples=4,
                 nonlocal_negative_samples=0,
                 pretrain_encoder_trainable=True,
                 pretrain_name="",
                 arch="lz-att",
                 name="",
                 debug=True,
                 background=True):

        self.input_validation_data_path = input_validation_data_path
        self.input_previous_model_path = input_previous_model_path
        self.input_training_data_path = input_training_data_path
        self.output_model_path = output_model_path
        self.log_dir = log_dir
        self.textual_embedding_trainable = textual_embedding_trainable
        self.pretrain_encoder_trainable = pretrain_encoder_trainable
        self.enable_pretrain_encoder = enable_pretrain_encoder
        self.enable_baseline = enable_baseline
        self.l2_norm_coefficient = l2_norm_coefficient
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.gain = gain
        self.testing_impression = testing_impression
        self.validation_step = validation_step
        self.training_step = training_step
        self.textual_embedding_dim = textual_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.compression_dim = compression_dim
        self.hidden_dim = hidden_dim
        self.attention_head_count = attention_head_count
        self.title_filter_shape = title_filter_shape
        self.title_shape = title_shape
        self.body_shape = body_shape
        self.recent_window_size = recent_window_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.rounds = rounds
        self.nonlocal_negative_samples = nonlocal_negative_samples
        self.negative_samples = negative_samples
        self.pretrain_name = pretrain_name
        self.arch = arch
        self.name = name
        self.debug = debug
        self.background = background

    @property
    def doc_meta_input(self):
        return os.path.join(self.input_training_data_path, 'DocMeta.tsv')

    @property
    def training_data_input(self):
        return os.path.join(self.input_training_data_path, 'ClickData.tsv')

    @property
    def title_embedding_input(self):
        return os.path.join(self.input_training_data_path, 'Vocab.tsv')

    @property
    def testing_data_input(self):
        return os.path.join(self.input_training_data_path, 'TestData.tsv')

    @property
    def model_input(self):
        return os.path.join(
            self.input_previous_model_path, 'model{}.json'.format(self.pretrain_name)), os.path.join(
            self.input_previous_model_path, 'model{}.pkl'.format(self.pretrain_name))

    @property
    def model_output(self):
        return os.path.join(
            self.output_model_path, 'model{}.json'.format(self.name)), os.path.join(
            self.output_model_path, 'model{}.pkl'.format(self.name))

    @property
    def encoder_input(self):
        return os.path.join(
            self.input_previous_model_path, 'encoder{}.json'.format(self.pretrain_name)), os.path.join(
            self.input_previous_model_path, 'encoder{}.pkl'.format(self.pretrain_name))

    @property
    def encoder_output(self):
        return os.path.join(
            self.output_model_path, 'encoder{}.json'.format(self.name)), os.path.join(
            self.output_model_path, 'encoder{}.pkl'.format(self.name))

    @property
    def log_output(self):
        return os.path.join(self.log_dir, 'log{}.txt'.format(self.name))

    @property
    def result_input(self):
        if self.debug:
            return os.path.join(self.output_model_path, 'model{}.tsv'.format(self.pretrain_name))
        else:
            return os.path.join(self.input_previous_model_path, 'model{}.tsv'.format(self.pretrain_name))

    @property
    def result_output(self):
        return os.path.join(self.output_model_path, 'model{}.tsv'.format(self.name))


if __name__ == "__main__":
    c = Config()
    print(hasattr(Config, "encoder_input"))

    """
    something to show
    """
    print(locals())