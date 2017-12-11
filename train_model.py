from utils import load_dataset, vectorize_text_data, get_next_batch, generate_words_vocab
from new_ner_model_lstm import NerModelLstm
from config import Config

def main(run_number):

    print "Running the Code for run number : " + str(run_number)

    config = Config(phase="train")

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
    
    print "\n\nCompleted running the code for run number : " + str(run_number) +  " with following performance stats : "

    print "Precision : " + str(config.precision)
    print "Recall    : " + str(config.recall)
    print "F1 Score  : " + str(config.f1)
    print "\n\n"
    return config.precision, config.recall, config.f1, config.dir_model

if __name__ == "__main__":
    
    total_runs = 3
    precision = []
    recall = []
    f1 = []
    model_dirs = []
    for run_number in range(1,total_runs + 1):
        p, r, f, model = main(run_number)
        precision.append(p)
        recall.append(r)
        f1.append(f)
        model_dirs.append(model)

        print "Average After Run Number : " + str(run_number)
        print "Average Precision        : " + str(precision/run_number)
        print "Average Recall           : " + str(recall/run_number)
        print "Average F1 Score         : " + str(f1/run_number)
        
        print "Model Directories : "
        print model_dirs
    
    print "Completed all runs !!" 
    print "Precision List : "
    print precision
    print "\nRecall List : "
    print recall
    print "\nF1 List : "
    print f1
    print "\nModel Directories : "
    print model_dirs

