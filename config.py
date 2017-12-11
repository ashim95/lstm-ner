from basic_utils import get_logger
import os
from utils import import_vocab_dicts, load_word_embeddings
import datetime

class Config():
    def __init__(self, phase="none"):

        if phase == "none":
            
            # Only using Configuration variables and vocabs

            self.load_vocab()

        if phase == "train":
            
            self.dir_model = self.dir_model + str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")) + "/"
            
            # For trainng phase
            # directory for training outputs
            if not os.path.exists(self.dir_model):
                os.makedirs(self.dir_model)
    
            # Pickle file inside dir_model to save hyperparameters and test performance stats
            self.hparmas_file = self.dir_model + self.hparmas_file
    
            self.wrong_predictions_file = self.dir_model + self.wrong_predictions_file
    
            # Directory inside dir_model to save tf model
            self.model_dir = self.dir_model + "model/"
    
    
            # create instance of logger
            self.logger = get_logger(self.dir_model + self.log_file)
            self.load_vocab()
            self.load_embeddings()

            self.load_dictionary()

        if phase == "restore":

            # Restoring the model
            self.logger = None
            self.load_vocab()

    def load_dictionary(self):
        if not self.use_hand_crafted:
            self.gazetteer = set()
            return
        names = []
        with open(self.dictionary_names_file, 'rb') as fp:
            for line in fp:
                names.append(line.strip().lower())
        self.gazetteer  = set(names)

    def load_vocab(self):
        
        # For loading vocabulary dictionaries
        self.vocab_words, self.vocab_chars, self.vocab_tags = import_vocab_dicts(self)
        self.update_size()
        self.vocabulary = set()
        # size of respective dictionaries

    def update_size(self):    
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

    special_tokens = [UNK, NUM, NONE]

    root_dir = "models/"
    dir_model  = root_dir + "model_"
    log_file   = "log.txt"
    hparmas_file = "logged_hparams.pkl"
    wrong_predictions_file = "wrong_predictions.txt"

    # space separated variables name of complex data structures like (list, dict, set, tuple) to log 
    # All other simple variables are logged by default
    special_variables_to_log = "hidden_size_lstm_list learning_rates_list features_index special_tokens"
    primitive_type_variables_to_log = (str, int, bool, float, long)


    # vocab (dictionaries saved as pkl files)
    filename_words_dict = "data/vocab_dict.pkl"
    filename_chars_dict = "data/char_dict.pkl"
    filename_tags_dict = "data/tags_dict.pkl"


    # Word Embeddings binary filename gensim pre-trained  
    filename_word2vec = "data/wikipedia-pubmed-and-PMC-w2v.bin"
    word2vec_size = 5443659
    word2vec_dim = 200
    retrain_embeddings = False
    reduce_embeddings = True
    use_word_embeddings = True


    # dataset
    #filename_dev = "data/ddiDataInCONLL_test.txt"
    filename_dev = "data/task_dev_conll.txt"
    filename_test = "data/task_test_conll.txt"
    #filename_train = "data/ddiDataInCONLL_train_2.txt"
    filename_train = "data/task_train_conll.txt"
    shuffle_data = True


    # Char Embedding variables
    use_chars = True
    dim_char = 50
    hidden_size_char = 50

    # Dropout
    use_dropout = True
    dropout_rate = 0.5


    # Model type
    use_bilstm = True
    hidden_size_lstm = 100
    # hidden_size_lstm_list = [100, 50, 150, 200, 300]
    hidden_size_lstm_list = [300]


    use_crf = True

    # Class Weights for highly imbalanced datasets
    use_class_weights = False
    class_0_weight = 0.0827799444854


    # Training Config variables
    default_lr = 0.001
    learning_rates_list = [0.001]
    learning_method = "adam"
    clip = -1
    nepochs = 150
    batch_size = 64
    lr_decay = 0.9
    nepoch_no_imprv = 30



    n_chars = 84
    n_tags = 3


    # Other variables
    word_index_padding = word2vec_size - 1
    padding_label = 0
    char_index_padding = n_chars -1 

    # For Hand Crafted and Dictionary
    use_hand_crafted = True
    features_size = 2
    features_padding = 0.0
    use_dictionary = True
    dictionary_names_file = "data/drug_names_wiki.txt"
    #features_index = [0]
    features_index = [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    features_size = len(features_index)

    # Path Variables For Java Program
    java_input_path = "logs/input/in.txt"
    java_output_path = "logs/output/out.txt"
    java_dictionary_file_1 = "data/drug_names_wiki.txt"
    java_dictionary_file_2 = "data/drug_names_long.txt" 
    java_jar_file = "drugner_java/jars/drugner_java_1.2.jar"


    print "Printing all parameter values ... "

    print "shuffle_data\t" + str(shuffle_data) 
    print "use_chars\t" + str(use_chars) 
    print "dim_char\t" + str(dim_char) 
    print "hidden_size_char\t" + str(hidden_size_char) 

    print "use_dropout\t" + str(use_dropout) 
    print "dropout_rate\t" + str(dropout_rate) 


    print "use_bilstm\t" + str(use_bilstm) 
    # print "hidden_size_lstm\t" + str(hidden_size_lstm) 
    print "hidden_size_lstm_list\t" + str(hidden_size_lstm_list) 


    print "use_crf\t" + str(use_crf) 

    # print "use_class_weights\t" + str(use_class_weights) 
    # print "class_0_weight\t" + str(class_0_weight) 


    # print "default_lr\t" + str(default_lr) 
    print "learning_rates_list\t" + str(learning_rates_list) 
    print "learning_method\t" + str(learning_method) 
    print "clip\t" + str(clip) 
    print "nepochs\t" + str(nepochs) 
    print "batch_size\t" + str(batch_size) 
    print "lr_decay\t" + str(lr_decay) 
    print "nepoch_no_imprv\t" + str(nepoch_no_imprv) 



    # print "n_chars\t" + str(n_chars) 
    # print "n_tags\t" + str(n_tags) 


    # print "word_index_padding\t" + str(word_index_padding) 
    # print "padding_label\t" + str(padding_label) 
    # print "char_index_padding\t" + str(char_index_padding) 

    print "use_hand_crafted\t" + str(use_hand_crafted) 
    print "features_size\t" + str(features_size) 
    print "features_padding\t" + str(features_padding) 
    print "use_dictionary\t" + str(use_dictionary) 
    print "dictionary_names_file\t" + str(dictionary_names_file) 
    print "features_index\t" + str(features_index) 
