from lz import *
import keras
import pickle


def load_test_data(path):
    with utils.open(path) as file:
        for i, line in enumerate(file):
            session_id, user, doc, label, score = line.strip().split('\t')
            yield session_id, int(user), int(doc), label, score


def test(config: settings.Config):
    yield config.result_output
    yield utils.load_model(config.model_input)
    batch_info = []
    batch_data = []
    for session_id, user, doc, label, score in load_test_data(config.testing_data_input):
        batch_info.append((session_id, label, score))
        batch_data.append((user, doc))
        yield batch_info, [np.stack(x) for x in zip(*batch_data)]
        batch_info = []
        batch_data = []
    if batch_info:
        yield batch_info, [np.stack(x) for x in zip(*batch_data)]


class RunUserModel(LzUserModeling):
    def __init__(self, config: settings.Config):
        super(RunUserModel, self).__init__(config)
        self._recall_model()

    def _load_data(self):
        logging.info('[+] loading user data')
        self.users = {}
        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[2]:
                    ih = self._extract_impressions(line[2])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih:
                        for pos in impression.pos:
                            ch.push(pos)
                    self.users[(line[0], line[1])] = ch.get_title()
        logging.info('[-] loaded {} users'.format(len(self.users)))

    def _recall_model(self):
        logging.info('[+] loading models')
        self.model = super(RunUserModel, self)._build_model()
        with open(self.config.model_input[1], 'rb') as file:
            self.model.set_weights(pickle.load(file))
        logging.info('[-] loaded models')
        """
        python3 main.py users -p lt.ca.catt.2 --debug -w 100
        import keras
        self.model = keras.Model(self.model.layers[0].input, self.model.layers[8].output)
        exit()
        """
        # utils.interactive_console(locals())

        self.model = keras.Model(self.model.layers[0].input, self.model.layers[5].get_output_at(-1))

        logging.info('[+] generating user vectors')
        self.user_vectors = {
            k: self.model.predict(v[None, :])[0] for k, v in self.users.items()
        }
        logging.info('[-] generated user vectors')

    def save_result(self):
        logging.info('[+] writing user vectors')
        with utils.open(self.config.result_output, 'w') as file:
            for (user_id, id_type), vec in self.user_vectors.items():
                file.write('{}\t{}\t{}\n'.format(user_id, id_type, ' '.join(map(str, vec))))
        logging.info('[-] written user vectors')
