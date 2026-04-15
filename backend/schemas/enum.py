from enum import Enum

class Tags(str, Enum):
    ABSTRACT = "abstract"
    MODULE = "module"
    LIB = "lib"
    ACTIVATION = "activation"
    OPERATOR = "operator"
    FUNCTION = "function"
    COMMON = "common"
    NLP = "nlp"
    CV = "cv"
    CNN = "cnn" # CNN base model
    RNN = "rnn" # RNN base model
    ATTENTION = "attention" # Attention base model
    NORM = "norm" # Normalization
    POOL = "pool" # Pooling

    TORCH_MODULE = "torch_module"

    TEST = "test" # for internal testing purposes
