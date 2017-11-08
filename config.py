from basic_utils import get_logger
import os
from utils import import_vocab_dicts, load_word_embeddings


class Config():
    def __init__(self, only_config=True):


        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)
        self.load_vocab()

        if not only_config:

            self.load_embeddings()

    def load_vocab(self):
        
        # For loading vocabulary dictionaries
        self.vocab_words, self.vocab_chars, self.vocab_tags = import_vocab_dicts(self)
        
        # size of respective dictionaries
        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

    def load_embeddings(self):
        # Load Word embeddings from gensim saved binary file 
        # as 2-D numpy array to use for embedding_lookup function

        self.embeddings_matrix = load_word_embeddings(self)

    # Special Tokens
    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "$NONE$"

    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"



    # vocab (dictionaries saved as pkl files)
    filename_words_dict = "data/vocab_dict.pkl"
    filename_chars_dict = "data/char_dict.pkl"
    filename_tags_dict = "data/tags_dict.pkl"


    # Word Embeddings binary filename gensim pre-trained  
    filename_word2vec = "data/wikipedia-pubmed-and-PMC-w2v.bin"
    word2vec_size = 5443659
    word2vec_dim = 200
    use_pretrained = False



    # dataset
    #filename_dev = "data/ddiDataInCONLL_test.txt"
    filename_dev = "data/task_dev_conll.txt"
    filename_test = "data/task_test_conll.txt"
    #filename_train = "data/ddiDataInCONLL_train_2.txt"
    filename_train = "data/task_train_conll.txt"


    # Char Embedding variables
    use_chars = True
    dim_char = 100
    hidden_size_char = 100

    # Dropout
    use_dropout = True
    dropout_rate = 0.2


    # Model type
    use_bilstm = True
    hidden_size_lstm = 100
   # hidden_size_lstm_list = [100, 50, 150, 200, 300]
    hidden_size_lstm_list = [100]


    use_crf = True

    # Class Weights for highly imbalanced datasets
    use_class_weights = False
    class_0_weight = 0.0827799444854


    # Training Config variables
    default_lr = 0.001
    learning_rates_list = [0.01]
    learning_method = "adam"
    clip = 5
    nepochs = 50
    batch_size = 128
    lr_decay = 1
    nepoch_no_imprv = 10



    n_chars = 84
    n_tags = 3


    # Other variables
    word_index_padding = word2vec_size - 1
    padding_label = 0
    char_index_padding = n_chars -1 