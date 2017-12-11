import numpy as np
import os
import tensorflow as tf
from basic_utils import Progbar
from utils import get_next_batch, pad_sequences, get_chunks, pad_sequences_word_ids
from utils import pad_sequences_char_ids, pad_sequences_features, write_wrong_predictions_to_file, save_parameters_to_file
import pprint

class NerModelLstm():
    """Class of Model for NER"""

    def __init__(self, config):

        """Defines self.config, sess, saver

        Args:
            config: (Config instance) class with configuration variables

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None

        self.hidden_size_lstm = self.config.hidden_size_lstm


    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        """

        # shape = (batch size, max length of sentence in batch)
        # Since both these can change, so use None
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # Placeholder for loading word_embeddings (vocab_size, word_vec_dim)
        # Although known, still using None for convenience
        self.embd_place = tf.placeholder(tf.float32, shape=[None, None], 
                        name="embd_place")

        # Placeholder for input features not computed in graph (hand crafted)
        # shape = (batch_size, max length of sentence in batch, number_of__handcrafted_features)
        self.input_feats = tf.placeholder(tf.float32, shape=[None, None, self.config.features_size], 
                            name="input_feats")
        
        # Dropout is given as placeholder so that we can switch it off during
        # testing phase, the value of the variable is the keep_probability
        # A scalar
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], 
                                      name="dropout")

        # Learning rate is a placeholder because we want to be able to change it
        # at every epoch (learning_rate decay)
        # A scalar
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                            name="learning_rate")

    def add_embeddings_op(self):
        
        """   
        Adds embedding ops to computational graph
        Can add word_embeddings ops, char_embeddings ops and an op for other 'hand_crafted features'
        Specify the same in config file
        """

        with tf.variable_scope("word_embed"):

            if self.config.use_word_embeddings:
                _word_embeddings = tf.Variable(tf.constant(0.0, shape=[self.config.word2vec_size,
                                                            self.config.word2vec_dim]),
                                                            trainable=self.config.retrain_embeddings,
                                                             name="_word_embeddings")
    
                self.embedding_init = _word_embeddings.assign(self.embd_place)
    
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                        self.word_ids, name="word_embeddings")
    
                # Following node was used for debugging purposes
                # word_embeddings = tf.Print(word_embeddings, 
                #                           [word_embeddings], summarize=100)
    
        
        with tf.variable_scope("char_embed"):

            if self.config.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char , activation=tf.nn.relu,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char , activation=tf.nn.relu,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                if self.config.use_word_embeddings:
                    word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                else:
                    word_embeddings = output
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        with tf.variable_scope("other_features"):
            # print word_embeddings.shape
            if self.config.use_hand_crafted:
                if (not self.config.use_word_embeddings) and (not self.config.use_chars):
                    word_embeddings = self.input_feats
                else:
                    word_embeddings = tf.concat([word_embeddings, self.input_feats], axis=-1)
            # print word_embeddings.shape
        # Add for discrete feature sets

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

    def add_predictions_op(self):
        """ Takes input and transforms into predictions
        
        """
        if self.config.use_bilstm:
            with tf.variable_scope("bi-lstm"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw, cell_bw, self.word_embeddings,
                                            sequence_length=self.sequence_lengths, dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                
                output = tf.nn.dropout(output, self.dropout)

        else:
            with tf.variable_scope("lstm"):
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)

                output, _ = tf.nn.dynamic_rnn(cell, self.word_embeddings, 
                                            sequence_length = self.sequence_lengths,
                                            dtype=tf.float32)
                output = tf.nn.dropout(output, self.dropout)


        with tf.variable_scope("pred_scores"):

            if self.config.use_bilstm:
                W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.hidden_size_lstm, self.config.ntags])

                # Biases initialized with zeros
                b = tf.get_variable("b", shape=[self.config.ntags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())

                nsteps = tf.shape(output)[1]
                output = tf.reshape(output, [-1, 2*self.hidden_size_lstm])
                pred = tf.matmul(output, W) + b
                self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

            else:
                W = tf.get_variable("W", dtype=tf.float32,
                    shape=[self.hidden_size_lstm, self.config.ntags])

                # Biases initialized with zeros
                b = tf.get_variable("b", shape=[self.config.ntags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())

                nsteps = tf.shape(output)[1]
                output = tf.reshape(output, [-1, self.hidden_size_lstm])
                pred = tf.matmul(output, W) + b
                self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_loss_op(self):
        """Adds ops for the loss function to the computational graph
        """

        # If use Linear Chain CRF
        if self.config.use_crf:

            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)

            # For computing transition scores for tags
            self.trans_params = trans_params

            self.loss =  tf.reduce_mean(-log_likelihood)

        # If use softmax
        else:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                        tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)

            if self.config.use_class_weights:

                ratio_0 = self.config.class_0_weight
                class_weight = tf.constant([ratio_0, (1.0 - ratio_0)/2, (1.0 - ratio_0)/2])

                labels_one_hot = tf.one_hot(indices=self.labels, depth = 3)
                labels_one_hot_ = tf.reshape(labels_one_hot, [-1, labels_one_hot.shape[2].value])
                weight_per_class_ = tf.multiply(class_weight, labels_one_hot_)
                weight_per_class = tf.reshape(weight_per_class_, tf.shape(labels_one_hot))
                weight_classes = tf.reduce_sum(weight_per_class, axis=-1)
                losses = tf.multiply(weight_classes, losses)

            # Masking neccessary because padding has been done
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)



    def add_training_op(self, l_rate):
        """
        Adds training ops to the computational graph
        l_rate : is a placeholder 
        """

        lr_method = self.config.learning_method.lower().strip()
        clip = self.config.clip
        with tf.variable_scope("train_op"):

            if lr_method == 'adam':
                optimizer = tf.train.AdamOptimizer(l_rate)
            elif lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(l_rate)
            elif lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(l_rate)
            elif lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(l_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(lr_method))

            if  clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))

            else:
                self.train_op = optimizer.minimize(self.loss)

    def create_feed_dict(self, sentences, labels=None, keep_probability=None, lr=None):

        word_ids, sequence_lengths = pad_sequences_word_ids(sentences, self.config.word_index_padding, self.config)

        if self.config.use_chars:

            char_ids, word_lengths = pad_sequences_char_ids(sentences, self.config.char_index_padding, self.config)

        if self.config.use_hand_crafted:
            input_feats = pad_sequences_features(sentences, self.config.features_padding, self.config)

        # print word_ids
        feed = {}
        feed[self.sequence_lengths] = sequence_lengths
        if self.config.use_word_embeddings:
            feed[self.word_ids] = word_ids
    
        if labels is not None:

            labels, _, _, _ = pad_sequences(labels,
                                            self.config.padding_label, 
                                            self.config.char_index_padding, 
                                                                    )
            # print labels[:2]
            feed[self.labels] = labels

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if self.config.use_hand_crafted:
            arr = np.asarray(input_feats )
            # print arr.shape
            feed[self.input_feats] = arr


        if lr is not None:
            
            # Case when running evaluation or inference on the graph
            # for training phase only

            feed[self.learning_rate] = lr

        feed[self.dropout] = keep_probability

        return feed


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def build_model(self):

        self.add_placeholders()
        self.add_embeddings_op()
        self.add_predictions_op()
        self.add_loss_op()
        self.add_training_op(self.learning_rate)

    def restore(self, model_path):
        
        #self.build_model()
        self.sess = tf.Session()

        #init = tf.global_variables_initializer()

        #self.sess.run(init)
        self.saver = tf.train.import_meta_graph(model_path + '.meta`')

        self.saver.restore(self.sess, model_path + "checkpoint")

    def train(self, train_set, dev_set, test_set):

        for h_size_lstm in self.config.hidden_size_lstm_list:

            print "Using hidden layer size " +str(h_size_lstm) + "\n"


            for lr in self.config.learning_rates_list:

                print "Using Learning rate " + str(lr) + "\n"

                tf.reset_default_graph()

                self.hidden_size_lstm = h_size_lstm

                self.build_model()

                self.sess = tf.Session()

                init = tf.global_variables_initializer()

                self.sess.run(init)
                self.saver = tf.train.Saver()
                if self.config.use_word_embeddings:
                    self.sess.run([self.embedding_init], {self.embd_place:self.config.embeddings_matrix})

                best_score = 0
                nepoch_no_imprv = 0 # for early stopping

                for epoch in range(self.config.nepochs):

                    self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

                    score = self.run_epoch(train_set, dev_set, epoch, lr)

                    lr *= self.config.lr_decay # decay learning rate

                    # early stopping and saving best parameters
                    if score >= best_score:
                        nepoch_no_imprv = 0
                        self.save_session()
                        best_score = score
                        print "Best score achieved !! " + str(score)
                        self.logger.info("Best score achieved !! " + str(score))
                    else:
                        nepoch_no_imprv += 1
                        if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                            self.logger.info("- early stopping {} epochs without "\
                                    "improvement".format(nepoch_no_imprv))
                            break


                print "Evaluating the best model on Test Set ... "

                self.evaluate(test_set)


                test_set_performance = self.evaluate(test_set, is_test_set=True)
                self.config.precision = test_set_performance["Precision"]
                self.config.accuracy = test_set_performance["acc"]
                self.config.recall = test_set_performance["Recall"]
                self.config.f1 = test_set_performance["f1"]

                print "Saving all hyperparameters and performance statistics to file : " + str(self.config.hparmas_file)

                save_parameters_to_file(self.config)

    def run_epoch(self, train_set, dev_set, epoch, lr_for_epoch):
        """Performs one epoch on whole trainning set
        Args:
            train_set: tuple of words, tags for training
            dev_set: tuple of words, tags for validation
            epoch : current epoch number
        """

        batch_size = self.config.batch_size
        nbatches = (len(train_set[0]) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        x_train = train_set[0]
        y_train = train_set[1]
        for i, (sentences_batch, labels_batch) in enumerate(get_next_batch((x_train, y_train), self.config.batch_size)):

            feed_batch = self.create_feed_dict(sentences_batch, labels_batch, keep_probability=self.config.dropout_rate, lr=lr_for_epoch)
            # print feed_batch

            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_batch)


            prog.update(i + 1, [("train loss", train_loss)])


        metrics = self.evaluate(dev_set)

        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def predict_batch(self,sentences):

        # Use keep probability equal to 1.0 since we are in testing phase
        feed_batch = self.create_feed_dict(sentences, keep_probability=1.0)

        sequence_lengths_batch = feed_batch[self.sequence_lengths]

        if self.config.use_crf:
            
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=feed_batch)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths_batch):
                logit = logit[:sequence_length] # keep only the valid steps using masking
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths_batch

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_batch)

            return labels_pred, sequence_lengths_batch

    def evaluate(self, test, is_test_set=False):

        x_test = test[0]
        y_test = test[1]

        accs = []

        wrong_predictions = []
        """
        wrong_predictions is a list of tuples of type (sentence, fp_set, fn_set, lab, lab_pred)
        """

        correct_preds, total_correct, total_preds = 0., 0., 0.

        for sentences_batch, labels_batch in get_next_batch((x_test, y_test), self.config.batch_size):

            labels_pred_batch, sequence_lengths_batch = self.predict_batch(sentences_batch)

            sentence_index = 0

            for lab, lab_pred, length in zip(labels_batch, labels_pred_batch,
                                             sequence_lengths_batch):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags, self.config))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags, 
                                                 self.config))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                fp_preds = lab_pred_chunks - lab_chunks
                fn_preds = lab_chunks - lab_pred_chunks
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
                if is_test_set and (len(fp_preds) != 0 or len(fn_preds) != 0):
                    wrong_pred = (sentences_batch[sentence_index], fp_preds, fn_preds, lab, lab_pred)
                    # print len(fp_preds) + len(lab_chunks & lab_pred_chunks)
                    # print len(fn_preds) + len(lab_chunks & lab_pred_chunks)
                    # print len(lab_pred_chunks)
                    # print len(lab_chunks)
                    # print fp_preds
                    # print fn_preds
                    wrong_predictions.append(wrong_pred)
                sentence_index +=1

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        print "Correct: " + str(correct_preds)
        print "Total Pred: " + str(total_preds)
        print "Total Correct: " + str(total_correct)
        print "Precision: " + str(p)
        print "Recall: " + str(r)
        print "F1: " + str(f1)
        
        if is_test_set:
            write_wrong_predictions_to_file(wrong_predictions, self.config)

        return {"acc": 100*acc, "f1": 100*f1, "Precision": 100*p, "Recall": 100*r}
