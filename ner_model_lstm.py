import numpy as np
import os
import tensorflow as tf
from basic_utils import Progbar
from utils import get_next_batch, pad_sequences, get_chunks

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

        # Default Params
        self.hidden_size_lstm = self.config.hidden_size_lstm
        self.lr = self.config.default_lr
        self.batch_size = self.config.batch_size

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



    def add_embeddings_op(self):
        """Adds embedding ops to computational graph
        """


        with tf.variable_scope("word_embed"):

            _word_embeddings = tf.Variable(tf.constant(0.0, shape=[self.config.word2vec_size,
                                                        self.config.word2vec_dim]),
                                                        trainable=self.config.use_pretrained,
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
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        # with tf.variable_scope("other_features"):

        # Add for discrete feature sets

        # if self.config.use_dropout:
        #     self.word_embeddings =  tf.nn.dropout(word_embeddings, self.config.dropout_rate)
        # else:
        self.word_embeddings = word_embeddings

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
                
                if self.config.use_dropout:
                    output = tf.nn.dropout(output, self.config.dropout_rate)

        else:
            with tf.variable_scope("lstm"):
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)

                output, _ = tf.nn.dynamci_rnn(cell, self.word_embeddings, 
                                            sequence_length = self.sequence_lengths,
                                            dtype=tf.float32)
                if self.config.use_dropout:
                    output = tf.nn.dropout(output, self.config.dropout_rate)


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



    def add_training_op(self):
        """Adds training ops to the computational graph
        """

        _lr_m = self.config.learning_method.lower()
        clip = self.config.clip
        with tf.variable_scope("train_op"):

            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if  clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))

            else:
                self.train_op = optimizer.minimize(self.loss)

    def create_feed_dict(self, sentences, labels=None):

        if self.config.use_chars:
            
            word_ids, sequence_lengths, char_ids, word_lengths = pad_sequences(sentences, 
                                                                    self.config.word_index_padding, 
                                                                    self.config.char_index_padding, 
                                                                    has_char = True)

        else:
            word_ids, sequence_lengths, _, _ = pad_sequences(sentences, 
                                                                    self.config.word_index_padding, 
                                                                    self.config.char_index_padding, 
                                                                    )
        # print len(word_ids)

        # print word_ids[:2]
        # print sequence_lengths[:2]
        # Create feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

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
        self.add_training_op()

    def train(self, train_set, dev_set):

        for h_size_lstm in self.config.hidden_size_lstm_list:

            print "Using hidden layer size " +str(h_size_lstm) + "\n"


            for lr in self.config.learning_rates_list:

                tf.reset_default_graph()

                self.hidden_size_lstm = h_size_lstm
                self.lr = lr

                self.build_model()

                self.sess = tf.Session()

                init = tf.global_variables_initializer()

                self.sess.run(init)
                self.saver = tf.train.Saver()
                self.sess.run([self.embedding_init], {self.embd_place:self.config.embeddings_matrix})


                best_score = 0
                nepoch_no_imprv = 0 # for early stopping

                for epoch in range(self.config.nepochs):

                    self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

                    score = self.run_epoch(train_set, dev_set, epoch)

                    self.lr *= self.config.lr_decay # decay learning rate

                    # early stopping and saving best parameters
                    if score >= best_score:
                        nepoch_no_imprv = 0
                        self.sess.save_session()
                        best_score = score
                        self.logger.info("- new best score!")
                    else:
                        nepoch_no_imprv += 1
                        if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                            self.logger.info("- early stopping {} epochs without "\
                                    "improvement".format(nepoch_no_imprv))
                            break


    def run_epoch(self, train_set, dev_set, epoch):
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
        for i, (sentences_batch, labels_batch) in enumerate(get_next_batch(  (x_train, y_train), self.batch_size)):

            feed_batch = self.create_feed_dict(sentences_batch, labels_batch)

            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_batch)


            prog.update(i + 1, [("train loss", train_loss)])


        metrics = self.evaluate(dev_set)

        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def predict_batch(self,sentences):

        feed_batch = self.create_feed_dict(sentences)

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

    def evaluate(self, test):

        x_test = test[0]
        y_test = test[1]

        accs = []

        correct_preds, total_correct, total_preds = 0., 0., 0.

        for sentences_batch, labels_batch in get_next_batch((x_test, y_test), self.batch_size):

            labels_pred_batch, sequence_lengths_batch = self.predict_batch(sentences_batch)

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
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        print "Correct: " + str(correct_preds)
        print "Total Pred: " + str(total_preds)
        print "Total Correct: " + str(total_correct)
        print "Precision: " + str(p)
        print "Recall: " + str(r)
        
        return {"acc": 100*acc, "f1": 100*f1, "Precision": 100*p, "Recall": 100*r}