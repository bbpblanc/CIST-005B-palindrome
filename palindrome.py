

from random import randint,choice,shuffle
import json

class Palindromes():
    RATIO_F = 0.1
    MAX_LENGTH = 100
    POSITIVE_SAMPLES = 100

    SETS = {
        'words': 'abcdefghijklmnopqrstuvwxyz',
        'numbers': '0123456789',
        'DNA': 'ATCG',
        'RNA': 'AUCG',
    }


    class Sample():
        def __init__(self,x,y):
            self.x = x
            self.y = y

        def __str__(self):
            return f'({self.x},{1 if self.y else 0})'
        
        def to_tuple(self):
            return (self.x, 1 if self.y else 0)
        
        def to_dict(self):
            return {'x':self.x, 'y':1 if self.y else 0}



    def __init__(self, source = None, *, auto=False, max_length = MAX_LENGTH, positive_samples = POSITIVE_SAMPLES):
        self.list_ = list()
        self.dict_ = dict()
        self.vocabulary = set()

        if source:
            if not isinstance(source, Palindromes):
                raise ValueError(f'the source to copy is expected to be a Palindrome')
            for v in source:
                [self.vocabulary.add(x) for x in v]
                self.append(v)

            for _ in range(10):
                self.shuffle()
            return

        if not auto:
            return
        
        # automatically populate the palindromes
        for m in range(1,max_length):
            #self.generate_set('numbers', m, positive_samples)
            self.generate_set('words', m, positive_samples)
            #self.generate_set('DNA', m, positive_samples)
            #self.generate_set('RNA', m, positive_samples)
        for _ in range(10):
            self.shuffle()

    @staticmethod
    def is_palindrome(candidate):
        n = len(candidate)
        return all([candidate[i] == candidate[n-i-1] for i in range(n)])
    
    def __len__(self):
        return len(self.list_)
    
    """
    def __iter__(self):
        return iter(self.list_)
    """
    
    def __str__(self):
        return '[' + ", ".join(map(str,self.list_)) + ']'

    @property
    def list(self):
        return [v.to_tuple() for v in self.list_]

    def is_true(self):
        return [v.x for v in self.list_ if v.y]
    def is_false(self):
        return [v.x for v in self.list_ if not v.y]

    def shuffle(self):
        shuffle(self.list_)

    def append(self, item):
        if not isinstance(item, Palindromes.Sample):
            raise ValueError(f'item expected to be a {__class__.__name__}.Sample')
        
        self.list_.append(item)
        self.dict_[item.x] = True

    def _add_candidate(self, candidate):
        if self.dict_.get(candidate,False):
                return False
        
        self.dict_[candidate] = True
        self.list_.append(Palindromes.Sample(candidate, Palindromes.is_palindrome(candidate)))
        return True

    def generate_set(self,set_,n,times=1):
        range_ = Palindromes.SETS.get(set_,None)
        
        if not range_:
            raise IndexError(f'{set_} if not a valid key')
        
        range_ = list(range_)
        [self.vocabulary.add(x) for x in range_]
        shuffle(range_)
        for _ in range(times):
            self._generate(n,times,range_)


    def _generate(self,n,times,range_):
        retry = 20
        while retry:
            retry -= 1
            base = [choice(range_) for _ in range(n)]
            candidate = "".join(base + base[::-1])
            if not self._add_candidate(candidate):
                continue

            # augmentation
            bound = int(n*Palindromes.RATIO_F)
            for _ in range(randint(1,1 if not bound else bound)):
                start = randint(1,2*n-1)
                end = randint(start,2*n-1)

                if not self._add_candidate(candidate[start:end]):
                    continue
            break

    def __iter__(self):
        for v in self.list_:
            yield (v['x'], v['y'])

    def __getitem__(self,i):
        if not (0 <= i < len(self)):
            raise IndexError(f'index {i} out of range')
        return self.list_[i].to_tuple()
    
    def get_any(self):
        return choice(self.list_)
    def get_range(self,n):
        if n >= len(self.list_):
            raise IndexError(f'index out of range. Max is {len(self.list_)}')
        start = randint(0,len(self.list_)-n-1)

        p = Palindromes()
        [p.append(v) for v in self.list_[start:start+n]]
        return p


    def dump(self, filename, pretty=False):
        data = dict()
        info = dict()

        min_length = min([len(v.x) for v in self.list_])
        max_length = max([len(v.x) for v in self.list_])


        info['author'] = 'Bertrand Blanc'
        info['total'] = len(self.list_)
        info['positive'] = len([1 for v in self.list_ if v.y])
        info['negative'] = len([1 for v in self.list_ if not v.y])
        info['min_length'] = min_length
        info['max_length'] = max_length
        info['vocabulary'] = sorted(list(self.vocabulary))
        data['info'] = info
        data['samples'] = [x.to_dict() for x in self.list_]

        with open(filename, "w", encoding="utf-8") as fd:
            if pretty:
                fd.write(json.dumps(data, indent=3))
            else:
                fd.write(json.dumps(data))


if __name__ == "__main__":
    max_length = 10
    positive_samples = 500000
    pretty = False

    p = Palindromes(auto=True, max_length=max_length, positive_samples=positive_samples)
    p.dump(f'palindromes_{max_length}_{positive_samples}.json.txt', pretty=pretty)
