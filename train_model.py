from utils import load_dataset, vectorize_text_data, get_next_batch, generate_words_vocab
from ner_model_lstm import NerModelLstm
from config import Config

def main():

    config = Config(only_config=False)

    # Load data from txt files
    train_data = load_dataset(config.filename_train)

    dev_data = load_dataset(config.filename_dev)

    test_data = load_dataset(config.filename_test)

    # Reduce Size of embeddings matrix
    generate_words_vocab(train_data, dev_data, test_data, config)

    # Vectorize Text data and convert to indices
    print len(config.vocab_words)
    train_vectorize = vectorize_text_data(train_data, config.vocab_words, 
        config.vocab_chars, config.vocab_tags, config)

    print len(train_vectorize[0])
    print len(train_vectorize[1])

    count_dict = {0:0, 1:0, 2:0}

    for y in train_vectorize[1]:
        for i in y:
            count_dict[i] +=1
    print count_dict
    
    dev_vectorize = vectorize_text_data(dev_data, config.vocab_words, 
        config.vocab_chars, config.vocab_tags, config)

    test_vectorize = vectorize_text_data(test_data, config.vocab_words, 
        config.vocab_chars, config.vocab_tags, config)

    model = NerModelLstm(config)
    model.build_model()

    model.train(train_vectorize, dev_vectorize, test_vectorize)

    # print "Evaluating the best model on Test Set"

    # model.evaluate(test_vectorize)
    

if __name__ == "__main__":
    main()