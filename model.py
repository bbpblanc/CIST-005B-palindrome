

__author__ = "Bertrand Blanc"

from sampling import Sampling
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import NewType, TypeAlias,Tuple


filename : TypeAlias = str
dimension : TypeAlias = int

class Model():
    TRAINING    = 'palindromes_100_1000.all.json.txt' # 1000 + samples for words, numbers and DNA
    TRAINING    = 'palindromes_100_100.json.txt'
    TRAINING    = 'palindromes_100_200.all.json.txt' # 200 + samples for words, numbers and DNA
    TRAINING    = 'palindromes_100_200.words.json.txt' # words only
    TRAINING    = 'palindromes_100_1000.words.json.txt' # 1000 + samples for words
    TRAINING    = 'palindromes_50_5000.words.json.txt' # 5000 + samples for words
    TRAINING    = 'palindromes_25_20000.words.json.txt' # 20_000 + samples for words
    TRAINING    = 'palindromes_10_50000.words.json.txt' # 50_000 + samples for short words 
    TRAINING    = 'palindromes_10_100000.words.json.txt' # 100_000 + samples for short words
    TRAINING    = 'palindromes_10_500000.words.json.txt'

    EPOCHS      = 4
    DROPOUT     = 0.08
    LEARNING_RATE = 0.001
    ADAM        = {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07} # default
    LR_DECAY    = {'initial_LR': 1e-2, 'steps':10000, 'rate':0.9} # default
    REGULARIZER = {'L1': 0.001, 'L2': 0.001}
    NEURONS     = [50,40]


    MODEL_FILENAME = f'model_{str(EPOCHS)}epochs_{"-".join(map(str,NEURONS))}.keras'
    TRAIN_PERCENTAGE= 1
    YHAT_THRESHOLD = 0.5
    VALIDATE_SET = [('Madam',1), ("Ma'am",1),
                    ('Dad!',1), ('Daddy!',0),
                    ('bop',0),
                    ('Mr. Owl ate etal worm.', 1), ('Mr. Owl ate etal wor.', 1),
                    ('Doc, note: et on cod.', 1), ('Doc, note: et on co.', 0), ('Doc, note: t on cod.', 0),
                    ("Dammit I'm mad", 1),
                    ('ATTTAAATTAAATTTA', 1), ('ATTTAAATTAAATTA', 0), ('ACAGCTGTTGTCGACA', 1)
                ]

    def __init__(self, auto:Tuple[filename,dimension]=(None,None)):
        self.model = None
        self.padding = 'X'

        if auto:
            # self.model = keras.saving.load_model(Model.MODEL_FILENAME)
            self.model = keras.saving.load_model(auto[0])
            self.data_length = auto[1]
            return
        
        self.samples = Sampling(Model.TRAINING, train_percentage=Model.TRAIN_PERCENTAGE, padding=self.padding)
        self.vectorization = None
        self.data_length = self.samples.data_length()

        self.create()
        self.train()

    def create(self):
        # StringLookup: https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/
        # TextVectorization: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
        # Sequential: https://keras.io/guides/sequential_model/
        # Dense: https://keras.io/api/layers/core_layers/dense/

        self.vectorization = keras.layers.StringLookup(
                max_tokens=len(self.samples.vocabulary())+1, # +1 for the padding token
                output_mode="int",
                vocabulary=self.samples.vocabulary(),
                encoding="utf-8",
                name='vectorization',
            )

        self.model = keras.Sequential()

        # the inputs are a np.array of characters (forming a palindrome or not)
        # the length of these strings varies, hence some padding is added to ensure all samples
        # have the same width

        # vectorization of the inputs to transform every letter from the vocabulary into integers
        # the vocabulary consists in a characters which can be found in the strings
        self.model.add(self.vectorization)

        # definition of the RNN per se
        for neurons in Model.NEURONS:
            layer = keras.layers.Dense(
                units=neurons,
                activation="relu",
                kernel_initializer="glorot_normal", # aka GlorotNormal, keras.initializers.GlorotNormal(seed=None),
                bias_initializer="zeros", # aka "keras.initializers.Zeros()",
                #kernel_regularizer = keras.regularizers.L1L2(l1=Model.REGULARIZER['L1'], l2=Model.REGULARIZER['L2']),
            )
            self.model.add(layer)

        # output layer: binary decision telling whether the input is a palindrome or not
        self.model.add(keras.layers.Dense(units=1, activation="sigmoid", name="output"))

        # SETTING THE LEARNING RATE
        # baseline constant learning rate
        learning_rate = Model.LEARNING_RATE

        # learning rate decay
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=Model.LR_DECAY['initial_LR'],
                decay_steps=Model.LR_DECAY['steps'],
                decay_rate=Model.LR_DECAY['rate'],
            )
        
        # SETTING THE KERNEL OPTIMIZER
        kernel_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=Model.ADAM['beta_1'],
                beta_2=Model.ADAM['beta_2'],
                epsilon=Model.ADAM['epsilon'],
            )


        # COMPILE THE MODEL
        self.model.compile(
            # optimizer
            optimizer=kernel_optimizer,

            # Loss function to minimize
            # since y is binary, the loss function is BinaryCrossentropy
            loss=keras.losses.BinaryCrossentropy(
                from_logits=False, # there's no logits for logistic regression
            ),

            # List of metrics to monitor
            metrics=[
                keras.metrics.Accuracy(),
                keras.metrics.BinaryAccuracy()
            ],
        )



    def train(self, resume_from_epoch=0):
        """Train the model
        :epoch: resume the training from epoch
        """

        # check the inputs have the proper shape, immediately fail otherwise
        self.input_check()

        # https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean
        # https://keras.io/api/models/model_training_apis/

        x_train, y_train = self.samples.train()
        x_test, y_test = self.samples.test()


        # nifty way to save time and resources by stopping at the epoch which doesn't keep decreasing the loss L
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

        # in case the execution of the training is aborted, restore from the last epoch
        abort_and_restore = keras.callbacks.BackupAndRestore(backup_dir="./tmp/backup", save_freq="epoch")

        history = self.model.fit(
            x = x_train,
            y = y_train,
            #batch_size=64,
            epochs          = Model.EPOCHS,
            shuffle         = True,
            initial_epoch   = resume_from_epoch,
            #validation_data = (x_test, y_test),
            validation_split=0.2, # if there's no validation_data, will save a percentage from the training set
            callbacks       = [early_stopping, abort_and_restore],
        )

        self.model.save(Model.MODEL_FILENAME, overwrite=True)


    def input_check(self):
        x,y = self.samples.train()
        assert x.shape[0] == y.shape[0], 'X_train and Y_train shall have the same length'
        assert x.shape[1] == self.data_length, f'X_train expected to be (None, {self.data_length})'
        assert y.shape[1] == 1, f'Y_train expected to be (None, 1)'


        x,y = self.samples.test()
        if x is None or y is None:
            #raise ValueError("X_test and Y_test are missing")
            return
        
        assert x.shape[0] == y.shape[0], 'X_test and Y_test shall have the same length'
        assert x.shape[1] == self.data_length, f'X_test expected to be (None, {self.data_length})'
        assert y.shape[1] == 1, f'Y_test expected to be (None, 1)'

        x,y = self.samples.validate()
        if x is None or y is None:
            return
        
        assert x.shape[0] == y.shape[0], 'X_validate and Y_validate shall have the same length'
        assert x.shape[1] == self.data_length, f'X_validate expected to be (None, {self.data_length})'
        assert y.shape[1] == 1, f'Y_validate expected to be (None, 1)'


    def evaluate(self):
        x_test, y_test = self.samples.test()
        self.model.evaluate(x_test, y_test)

    def is_palindrome(self, candidate):
        candidate_norm = [c for c in candidate.lower() if c.isalnum()]

        if len(candidate_norm) > self.data_length:
            raise ValueError(f'max length {self.data_length}, normalized "{candidate}" is {len(candidate_norm)}-character long')

        candidate = candidate_norm
        candidate += [self.padding]*(self.data_length-len(candidate))
        candidate = np.array([candidate])
        assert candidate.shape == (1,self.data_length), f'erroneous shape {candidate.shape}, (1,{self.data_length}) expected'
        #print(candidate)

        yhat = self.model.predict(x=candidate, verbose=0)
        return yhat > Model.YHAT_THRESHOLD, yhat.flatten()
    
    def validate(self):
        correct = 0
        for x,y in Model.VALIDATE_SET:
            yhat, _ = self.is_palindrome(x)
            correct += 1 if yhat == y else 0
        return round(correct / len(Model.VALIDATE_SET), 4)
    


if __name__ == "__main__":
    exit(0)
    m = Model()

