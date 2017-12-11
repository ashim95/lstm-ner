import os
import tensorflow as tf
from config import Config
from ner_model_lstm import NerModelLstm
from utils import load_pkl_file
import argparse

def restore_model(model_directory):
    if not os.path.exists(model_directory):
        raise OSError(errno.ENOTDIR, strerror(errno.ENOTDIR), model_directory)

    # model_hparams_file = model_directory + "logged_hparams.pkl"
    # model_hparams = load_pkl_file(model_hparams_file)
    
    model_path = model_directory 

    print "Restoring model from : " + str(model_directory)

    config = Config(phase="restore")

    model = NerModelLstm(config)
    model.restore(model_path)
    #model.build_model()
    #model.saver = tf.train.Saver()
    #model.saver.restore(model.sess, model_path)

    return model

def compare_models(model_directory_1, model_directory_2):
    
    print "Comapring models : " + model_directory_1 + " and " + model_directory_2
    if not os.path.exists(model_directory_1):
        raise OSError(errno.ENOTDIR, strerror(errno.ENOTDIR), model_directory_1)

    if not os.path.exists(model_directory_2):
        raise OSError(errno.ENOTDIR, strerror(errno.ENOTDIR), model_directory_2)

    hparams_filename = "logged_hparams.pkl"

    model_hparams_1 = load_pkl_file(model_directory_1 + hparams_filename)
    model_hparams_2 = load_pkl_file(model_directory_2 + hparams_filename)

    different_dict = {}
    dictionary_keys = set(model_hparams_1.keys() + model_hparams_2.keys())

    for key in dictionary_keys:
        if key not in model_hparams_1:
            val = ("---", str(model_hparams_2[key]))
            different_dict[key] = val
        elif key not in model_hparams_2:
            val = (str(model_hparams_1[key]), "---")
            different_dict[key] = val
        else:
            if model_hparams_1[key] != model_hparams_2[key]:
                val = (str(model_hparams_1[key]), str(model_hparams_2[key]))
		different_dict[key] = val

    
    print "{:<70} {:<70} {:<70}".format('Keys',model_directory_1, model_directory_2)

    print "\n\n"

    for key, val in different_dict.iteritems():
        print "{:<70} {:<70} {:<70}".format(key, val[0], val[1])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--function', help='Pass the function to run ')

    parser.add_argument('--model_directory', help='if method is compare_models, pass two directory paths separated by comma')

    args = parser.parse_args()

    if args.function == "compare_models":

        model_dirs = args.model_directory.split(",")

        model_directory_1 = model_dirs[0].strip()

        model_directory_2 = model_dirs[1].strip()

        compare_models(model_directory_1, model_directory_2)

    if args.function == "restore_model":
        model_dir = args.model_directory.strip()

        restore_model(model_dir)

    return

if __name__ == "__main__":
    main()
