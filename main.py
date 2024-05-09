
__author__ = "Bertrand Blanc"

import argparse
import sys

from model import Model

class Main():
    def __init__(self, *args, **kargs):
        self.parser = None
        self._parser()

        if len(*args) == 0:
            args = [['-h']]
        self.args = self.parser.parse_args(*args)

        self._dispatch()

    def _parser(self):
        self.parser = argparse.ArgumentParser(
            prog='main.py',
            description='predict whether a word is a palindrome or not, basd on a NN algorithm',
            epilog="thank you for using this program"
        )
        self.parser.add_argument('--author', action='store_true', help='author of the program')
        self.parser.add_argument('--model', '-m', action='store', help='file name for a keras existing model')
        self.parser.add_argument('--data-length', '-dl', action='store', help='length of the data processed by the keras model')
        self.parser.add_argument('--create', action='store_true', help='creates a keras NN, train it and store it into a keras file')

    def _dispatch(self):
        if self.args.author:
            self._author()
            self._terminate()

        if self.args.create:
            self._create_model()
            self._terminate(0)
        
        if self.args.model and not self.args.data_length:
            print('please set a data length')
            self._terminate(-1)

        if not self.args.model and self.args.data_length:
            print('please provide a keras file')
            self._terminate(-1)

        try:
            int(self.args.data_length)
        except Exception:
            print(f'data length is expected to be an integer')
            self._terminate(-1)
        self._load_model()

    def _create_model(self):
        raise NotImplementedError('option not accessible from CLI')
        model = Model()
        model.evaluate()


    def _load_model(self):
        model = Model(auto=(self.args.model, int(self.args.data_length)))
        result = model.validate()
        print(f'model validation: {int(round(result*100, 0))}%')
        self._user_inputs(model)
                      

    def _user_inputs(self, model):
        candidate = ""
        while True:
            candidate = input('candidate (stop to exit): ')
            if candidate == 'stop':
                return
            
            try:
                yhat, prob = model.is_palindrome(candidate)
                if yhat:
                    print(f'"{candidate}" is likely to be a palindrome ({prob})')
                else:
                    print(f'"{candidate}" is unlikely to be a palindrome ({prob})')
            except ValueError as e:
                print(e)


    def _author(self):
        print('author: Bertrand Blanc')

    def _terminate(self, exit_=0):
        exit(exit_)


if __name__ == "__main__":
    try:
        Main(sys.argv[1:])
    except Exception as e:
        print(e)
    exit(-2)
