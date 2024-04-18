
from palindrome import Palindromes
import json
import numpy as np

class Sampling():
    def __init__(self, source, padding=None, train_percentage=.8, validate_percentage=0):
        self._source = source
        self._dict = None
        self._data_length = None
        self._padding = padding # padding value: 0, ... or None if no padding
        self._vocabulary = None

        if isinstance(source, str):
            # assuming it's a JSON file
            with open(source, 'r', encoding='utf-8') as fd:
                self._dict = json.loads(fd.read())
            self._source = [(v['x'],v['y']) for v in self._dict['samples']]

            self._data_length = self._dict['info']['max_length']
            self._vocabulary = self._dict['info']['vocabulary']


        #if not isinstance(source, list):
        #    raise AttributeError('the source shall be a list')

        if not self._data_length:
            self._data_length = max([len(x[0]) for x in self._source])
        
        set_ = set()
        for x in self._source:
            [set_.add(c) for c in x[0]]
        

        self._vocabulary = sorted(set_)

        if self._padding is not None:
            self._pad()

        train_max = int(len(self._source) * train_percentage)
        self._train_list = self._source[:train_max]
        

        if validate_percentage:
            validate_min = int((len(self._source)-train_max)*validate_percentage)
            self._test_list = self._source[train_max:-validate_min]
            self._validate_list = self._source[-validate_min:]
        else:
            self._test_list = self._source[train_max:]
            self._validate_list = None

        

    def _pad(self):
        for i,x in enumerate(self._source):
            padding = self._data_length - len(x[0])
            self._source[i] = (x[0] + self._padding*padding, x[1])


    def train(self):
        if self._train_list and len(self._train_list) > 0:
            return np.array([np.array(list(v[0])) for v in self._train_list]), np.array([np.array([v[1]]) for v in self._train_list])
        return None,None

    def test(self):
        if self._test_list and len(self._test_list) > 0:
            return np.array([np.array(list(v[0])) for v in self._test_list]), np.array([np.array([v[1]]) for v in self._test_list])
        return None,None
    
    def validate(self):
        if self._validate_list and len(self._validate_list) > 0:
            return np.array([np.array(list(v[0])) for v in self._validate_list]), np.array([np.array([v[1]]) for v in self._validate_list])
        return None,None
    
    def data_length(self):
        return self._data_length
    
    def vocabulary(self):
        return self._vocabulary

if __name__ == "__main__":
    p = Palindromes(auto=True)
    s = Sampling(p.list, train_percentage=.92, validate_percentage=.05)

    train_x, train_y = s.train()
    test_x, test_y = s.test()
    validate_x, validate_y = s.validate()

    print(len(train_x), train_x[0], train_y[0] , ' -> ', train_x[-1], train_y[-1] )
    print(len(test_x), test_x[0], test_y[0] , ' -> ', test_x[-1], test_y[-1] )
    print(len(validate_x), validate_x[0], validate_y[0] , ' -> ', validate_x[-1], validate_y[-1] )
    print('vocabulary:', s.vocabulary())
    print()


    #s = Sampling('palindromes_100_1000.json.txt', train_percentage=.92, validate_percentage=.05)
    s = Sampling('palindromes_100_1000.json.txt', train_percentage=.95)
    train_x, train_y = s.train()
    test_x, test_y = s.test()
    validate_x, validate_y = s.validate()
    print(len(train_x), train_x[0], train_y[0] , ' -> ', train_x[-1], train_y[-1] )
    print(len(test_x), test_x[0], test_y[0] , ' -> ', test_x[-1], test_y[-1] )
    #print(len(validate_x), validate_x[0], validate_y[0] , ' -> ', validate_x[-1], validate_y[-1] )

    print('vocabulary:', s.vocabulary())

