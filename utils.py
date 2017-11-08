import numpy as np
import pickle
import gensim

# Special Tokens
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "$NONE$"


def load_dataset(filename):
    """Load dataset from a CoNLL format file

    Args:
        filename: name of the dataset file

    Returns:
        a tuple of words and tags.
        dataset_words : list of list of list of words
                        [Documents  [Sentences  [words]    ]    ]
        dataset_tags : same shape as dataset_words, contains
                       corresponding tags
    """


    with open(filename) as fp:
        dataset_words, dataset_tags = [], []
        doc_words, doc_tags = [], []
        sent_words, sent_tags = [], []
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                if len(sent_words) != 0:
                    doc_words.append(sent_words)
                    doc_tags.append(sent_tags)
                sent_words, sent_tags = [], []
            elif line.startswith("-DOCSTART"):
                if len(doc_words) !=0:
                    dataset_words.append(doc_words)
                    dataset_tags.append(doc_tags)
                doc_words, doc_tags = [], []
                sent_words, sent_tags = [], []
            else:
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                sent_words.append(line_content[0])
                sent_tags.append(line_content[1])
    print "Completed reading of the dataset from file : " + filename
    
    return dataset_words, dataset_tags

def load_pkl_file(filename):
    with open(filename, 'rb') as fp:
        pkl_file = pickle.load(fp)
    return pkl_file


def import_vocab_dicts(config):

    """ Load vocabulary dictionary for words, chars, and tags

    Args:
        None, since filenames are obtained from config file

    Returns:
            dictionary for words, chars, tags
    """
    # filename_words_dict = "data/vocab_dict.pkl"
    # filename_chars_dict = "data/char_dict.pkl"
    # filename_tags_dict = "data/tags_dict.pkl"
    # word_dict = load_pkl_file(filename_words_dict)
    # chars_dict = load_pkl_file(filename_chars_dict)
    # tags_dict = load_pkl_file(filename_tags_dict)
    word_dict = load_pkl_file(config.filename_words_dict)
    chars_dict = load_pkl_file(config.filename_chars_dict)
    tags_dict = load_pkl_file(config.filename_tags_dict)
    return word_dict, chars_dict, tags_dict


def vectorize_text_data(data, vocab_words, vocab_chars, vocab_tags, config):
    """ Vectorize text data, also replaces numbers with NUM token 
                            and converts all words to lowercase

    Args:
        data: tuple of words, tags (each as list of list of list)

        others: dictionary for words, tags

    Returns:
            Vectorized dataset: is a tuple (words, tags)
            which is a list of sentences, each sentence is a list of words
            each word is a tuple (list_of_char_ids, word_id)
    """

    dataset_words, dataset_tags = data
    vec_words = []
    vec_tags = []
    for i in range(len(dataset_words)):
        doc_words = dataset_words[i]
        doc_tags = dataset_tags[i]
        for j in range(len(doc_words)):
            w = []
            t = []
            # vec_sent = []
            sent_words = doc_words[j]
            sent_tags = doc_tags[j]
            for k in range(len(sent_words)):
                if sent_words[k].isdigit():
                    if config.use_chars:
                        w.append(  (  word_to_char_id_list(sent_words[k], vocab_chars)  , vocab_words.get(NUM, vocab_words[UNK])
                                    )     )
                        t.append(vocab_tags[sent_tags[k]])
                    else:
                        w.append(vocab_words.get(NUM, vocab_words[UNK]))
                        t.append(vocab_tags[sent_tags[k]])
                else:
                    if config.use_chars:
                        w.append( ( word_to_char_id_list(sent_words[k], vocab_chars) ,
                         vocab_words.get(sent_words[k].lower(),vocab_words[UNK] ) ) )
                        t.append(vocab_tags[sent_tags[k]])
                    else:
                        w.append(vocab_words.get(sent_words[k].lower(), vocab_words[UNK]))
                        t.append(vocab_tags[sent_tags[k]])
            vec_words.append(w)
            vec_tags.append(t)
    return vec_words, vec_tags

def word_to_char_id_list(word, vocab_chars):
    id_list = []
    default_char = "~"
    for c in word:
        id_list.append(vocab_chars.get( c, vocab_chars[default_char]))
    return id_list


def convert_to_list_of_sentences(dataset):

    """Convert vectorized data from list of docs to list of Sentences

    Args:
        dataset: a list of docs, doc is a list of sentences

    Returns
        a list of sentences
    """

    words = []
    tags = []
    for docs in dataset:
        for sent in docs:
            l = []
            t = []
            for s in sent:
                l.append(s[0])
                t.append(s[1])
            words.append(l)
            tags.append(t)
    return words, tags

def remove_large_sentences(data, maxlen):
    """ Removes sentences larger than specified length

    Args: 
        data: list of sentencesS
        maxlen: integer, max length of sentence allowed

    Returns:
        dataset: list of sentences (length less equal than maxlen) 
    """
    dataset = []
    for d in data:
        if len(d) <= maxlen:
            dataset.append(d)
    print "Remaining: " + str(len(dataset))
    return dataset


def load_word_embeddings(config):
    
    """Load Word Embeddings as numpy array

        Args:
            empty

        Returns:
                Numpy array

    """

    if not hasattr(config, 'vocab_words'):
        config.load_vocab()

    filename_word2vec = config.filename_word2vec
    i_UNK = config.nwords -3
    i_NUM = i_UNK + 1
    i_NONE = i_NUM + 1 # For padding
    word2vec_dim = config.word2vec_dim

    filename = filename_word2vec
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    emb_size = word2vec_dim
    embeddings = np.zeros((config.nwords, emb_size))
    i = 0
    for key in wv_model.wv.index2word:
        embeddings[i] = wv_model[key]
        i+=1
    embeddings[i_UNK] = calculate_UNK_embedding_value(wv_model)
    embeddings[i_NUM] = calculate_NUM_embedding_value(wv_model, config.vocab_words)
    embeddings[i_NONE] = np.zeros((1, 200))
    return embeddings

def calculate_NUM_embedding_value(wv_model, vocab_words):
    num_words = []

    num_embedding = np.zeros((1, 200))

    for word in vocab_words.keys():
        if word.isdigit():
            num_words.append(word)
    i = 0
    for word in num_words:
        if i < 1000:
            num_embedding += wv_model[word]

    return num_embedding/1000 

def calculate_UNK_embedding_value(wv_model):
    unk_embedding = np.zeros((1, 200))
    threshold = 80
    infrequent_words = []
    for word, vocab in wv_model.vocab.iteritems():
        if vocab.count < threshold:
            infrequent_words.append(word)

    for word in infrequent_words:
        unk_embedding += wv_model[word]
    # print len(infrequent_words)
    return unk_embedding/len(infrequent_words)


def get_next_batch(data, batch_size):
    
    x_batch, y_batch = [], []
    x_train = data[0]
    y_train = data[1]
    for i in range(len(x_train)):
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x_train[i]]
        y_batch += [y_train[i]]
    if len(x_batch) != 0:
        yield x_batch, y_batch

def pad_sequences(sentences, value1, value2, has_char=False):

    if has_char:
        
        lengths = [len(sent) for sent in sentences]
        max_sentence_len = max(lengths)

        lengths = [len(w[0]) for sent in sentences for w in sent]
        max_word_length = max(lengths)

        padded_words = []
        padded_sentences = []
        word_lengths = []
        sentence_lengths = []
        char_pad_words = [value2] * max_word_length

        for sent in sentences:
            char_ids, word_ids = map(list, zip(*sent))
            if len(sent) < max_sentence_len:
                padding = [value1] * (max_sentence_len - len(word_ids))
                new_sent = word_ids + padding
                padded_sentences.append(new_sent)
                sentence_lengths.append(len(word_ids))

                new_char_ids = []
                w_length = []
                for c in char_ids:
                    if len(c) < max_word_length:
                        padding_char = [value2] * (max_word_length - len(c))
                        new_c = c + padding_char
                        new_char_ids.append(new_c)
                        w_length.append(len(c))
                    else:
                        new_char_ids.append(c)
                        w_length.append(len(c))

                new_char_ids = new_char_ids + [char_pad_words] * (max_sentence_len - len(sent))
                w_length = w_length + [0] * (max_sentence_len - len(sent))
                padded_words.append(new_char_ids)
                word_lengths.append(w_length)


            else:
                padded_sentences.append(word_ids)
                sentence_lengths.append(len(word_ids))
                new_char_ids = []
                w_length = []
                for c in char_ids:
                    if len(c) < max_word_length:
                        padding_char = [value2] * (max_word_length - len(c))
                        new_c = c + padding_char
                        new_char_ids.append(new_c)
                        w_length.append(len(c))
                    else:
                        new_char_ids.append(c)
                        w_length.append(len(c))
                padded_words.append(new_char_ids)
                word_lengths.append(w_length)
    
    else:
        lengths = [len(sent) for sent in sentences]
        max_sentence_len = max(lengths)
        padded_sentences = []
        sentence_lengths = []
        padded_words = []
        word_lengths = []


        for sent in sentences:
            if len(sent) < max_sentence_len:
                padding = [value1] * (max_sentence_len - len(sent))
                new_sent = sent + padding
                padded_sentences.append(new_sent)
                sentence_lengths.append(len(sent))
            else:
                padded_sentences.append(sent)
                sentence_lengths.append(len(sent))

    return padded_sentences, sentence_lengths, padded_words, word_lengths



def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags, config):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = config.padding_label
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

