# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import argparse
import copy
import my_dropout


class Config(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.50
  learning_rate = 0.7
  init_scale = 0.05
  num_epochs = 65
  word_vocab_size = 0 # to be determined later
  weight_decay = 1e-7

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 300
  num_layers = 2
  drop_x = 0.10
  drop_i = 0.20
  drop_h = 0.10
  drop_o = 0.20

  # Pattern embedding hyperparameters
  pat_vocab_size = 0 # to be determined later
  pat_emb_dim = 300
  max_word_len = 0   # to be determined later
  highway_size = pat_emb_dim


def parse_args():
  '''Parse command line arguments'''
  parser = argparse.ArgumentParser(formatter_class=
                                   argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--is_train', default='1', 
                      help='mode. 1 = training, 0 = evaluation')
  parser.add_argument('--raw_data_dir', default='data_c16/',
                      help='raw data directory. Should have train.txt/valid.txt' \
                           '/test.txt with input data')
  parser.add_argument('--data_dir', default='data_c16/', 
                      help='converted data directory. Should have trainFSM.txt/validFSM.txt' \
                           '/testFSM.txt with converted data, and allFSM.txt which is a' \
                           'concatenation of these three *FSM.txt files')
  parser.add_argument('--save_dir', default='saves',
                      help='saves directory')
  parser.add_argument('--prefix', default='Pat-Sum',
                      help='prefix for filenames when saving')
  parser.add_argument('--eos', default='<eos>',
                      help='EOS marker')
  return parser.parse_args()


def read_data(args, config):
  '''read data sets, construct all needed structures and update the config'''
  def my_patterns(word):
    return word.split(':')

  if args.is_train == '1':
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, args.prefix + '-data.pkl'), 'wb') as data_file:
      word_data = open(os.path.join(args.data_dir, 'allFSM.txt'), 'r').read().replace('\n', '').split(':1:')
      words = list(set(word_data))
      
      patterns = set()
      word_lens_in_pat = []

      for word in words:
        pats = my_patterns(word)
        word_lens_in_pat.append(len(pats))
        for pat in pats:
          patterns.add(pat)

      pats_list = list(patterns)
      pickle.dump((word_data, words, word_lens_in_pat, pats_list), data_file)

  else:
    with open(os.path.join(args.save_dir, args.prefix + '-data.pkl'), 'rb') as data_file:
      word_data, words, word_lens_in_pat, pats_list = pickle.load(data_file)

  word_data_size, word_vocab_size = len(word_data), len(words)
  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', '').split(':1:')
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'trainFSM.txt'))
  valid_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'validFSM.txt'))
  test_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'testFSM.txt'))

  pat_vocab_size = len(pats_list)
  max_word_len = int(np.percentile(word_lens_in_pat, 100))
  config.max_word_len = max_word_len
  print('data has %d unique patterns' % pat_vocab_size)
  print('max word length in patterns is set to', max_word_len)

  # a fake patlable for zero-padding
  zero_pad_pat = ' '
  pats_list.insert(0, zero_pad_pat)
  pat_vocab_size += 1
  config.pat_vocab_size = pat_vocab_size

  pat_to_ix = { pat:i for i,pat in enumerate(pats_list) }
  ix_to_pat = { i:pat for i,pat in enumerate(pats_list) }

  word_ix_to_pat_ixs = {}
  for word in words:
    word_ix = word_to_ix[word]
    word_in_pats = my_patterns(word)
    if len(word_in_pats) > max_word_len:
      del word_in_pats[max_word_len:]
    else:
      word_in_pats += [zero_pad_pat] * (max_word_len - len(word_in_pats))
    word_ix_to_pat_ixs[word_ix] = [pat_to_ix[pat] for pat in word_in_pats]

  return train_raw_data, valid_raw_data, test_raw_data, word_ix_to_pat_ixs


def read_raw_data(args, config):
  '''read the raw data at word level'''

  if args.is_train == '1':
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, args.prefix + '-raw_data.pkl'), 'wb') as data_file:
      word_data = open(os.path.join(args.raw_data_dir, 'train.txt'), 'r').read().replace('\n', '').split('_')
      words = list(set(word_data))
      pickle.dump((word_data, words), data_file)

  else:
    with open(os.path.join(args.save_dir, args.save_name + '-raw_data.pkl'), 'rb') as data_file:
      word_data, words = pickle.load(data_file)

  word_data_size, word_vocab_size = len(word_data), len(words)
  print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
  config.word_vocab_size = word_vocab_size
  config.num_sampled = int(word_vocab_size * 0.2)

  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', '').split('_')
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data(os.path.join(args.raw_data_dir, 'train.txt'))
  valid_raw_data = get_word_raw_data(os.path.join(args.raw_data_dir, 'valid.txt'))
  test_raw_data = get_word_raw_data(os.path.join(args.raw_data_dir, 'test.txt'))

  return train_raw_data, valid_raw_data, test_raw_data


class batch_producer(object):
  '''Slice the raw data into batches'''
  def __init__(self, raw_data, ptb_raw_data, batch_size, num_steps):
    self.raw_data = raw_data
    self.ptb_raw_data = ptb_raw_data
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    self.batch_len = len(self.raw_data) // self.batch_size
    self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                           (self.batch_size, self.batch_len))
    self.ptb_data = np.reshape(self.ptb_raw_data[0 : self.batch_size * self.batch_len],
                               (self.batch_size, self.batch_len))
    
    self.epoch_size = (self.batch_len - 1) // self.num_steps
    self.i = 0
  
  def __next__(self):
    if self.i < self.epoch_size:
      # batch_x and batch_y are of shape [batch_size, num_steps]
      batch_x = self.data[::, self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
      batch_y = self.data[::, self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
      ptb_batch_x = self.ptb_data[::, self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
      ptb_batch_y = self.ptb_data[::, self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
      self.i += 1
      return (batch_x, batch_y, ptb_batch_x, ptb_batch_y)
    else:
      raise StopIteration()

  def __iter__(self):
    return self


class Model:
  '''pattern-aware language model'''
  def __init__(self, config, word_ix_to_pat_ixs, need_reuse=False):
    # get hyperparameters
    batch_size = config.batch_size
    num_steps = config.num_steps
    max_word_len = config.max_word_len
    pat_emb_dim = config.pat_emb_dim
    highway_size = config.highway_size
    init_scale = config.init_scale
    num_sampled = config.num_sampled
    pat_vocab_size = config.pat_vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    word_vocab_size = config.word_vocab_size
    drop_x = config.drop_x
    drop_i = config.drop_i
    drop_h = config.drop_h
    drop_o = config.drop_o
    weight_decay = config.weight_decay

    # pattern embedding matrix
    with tf.variable_scope('pat_emb', reuse=need_reuse):
      self.pat_embedding = tf.get_variable("pat_embedding", 
        [pat_vocab_size, pat_emb_dim], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(-init_scale, init_scale))
    
    # placeholders for training data and labels
    self.x = tf.placeholder(tf.int32, [batch_size, num_steps, max_word_len])
    self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
    y_float = tf.cast(self.y, tf.float32)
    
    # we first embed patterns ...
    words_embedded = tf.nn.embedding_lookup(self.pat_embedding, self.x)
    words_embedded = tf.reshape(words_embedded, [-1, max_word_len, pat_emb_dim])
    # ... and then sum pattern vectors to get a word vector
    words_embedded_sum = tf.reduce_sum(words_embedded, axis=1)    
    
    # we feed the word vector into a stack of two HW layers ...
    def highway_layer(highway_inputs):
      transf_weights = tf.get_variable('transf_weights', 
        [highway_size, highway_size],
        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
        dtype=tf.float32)
      transf_biases = tf.get_variable('transf_biases', [highway_size],
        initializer=tf.random_uniform_initializer(-2-0.01, -2+0.01),
        dtype=tf.float32)
      highw_weights = tf.get_variable('highw_weights', 
        [highway_size, highway_size],
        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
        dtype=tf.float32)
      highw_biases = tf.get_variable('highw_biases', [highway_size],
        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
        dtype=tf.float32)
      transf_gate = tf.nn.sigmoid(tf.matmul(highway_inputs, transf_weights)         + transf_biases)
      highw_output = tf.multiply(transf_gate, 
        tf.nn.relu(tf.matmul(highway_inputs, highw_weights) + highw_biases)) \
        + tf.multiply(tf.ones([highway_size], dtype=tf.float32) - transf_gate, 
        highway_inputs)
      return highw_output, transf_gate
    
    with tf.variable_scope('highway1', reuse=need_reuse):
      highw1_output, self.t1 = highway_layer(words_embedded_sum)
    
    with tf.variable_scope('highway2', reuse=need_reuse):
      highw2_output, self.t2 = highway_layer(highw1_output)
        
    highw_output_reshaped = tf.reshape(highw2_output, 
                                       [batch_size, num_steps, -1])
    if not need_reuse:
      highw_output_reshaped = tf.nn.dropout(highw_output_reshaped, 
                                            1-drop_x, [batch_size, num_steps, 1])
    
    # ... and then process it with a stack of two LSTMs
    lstm_input = tf.unstack(highw_output_reshaped, axis=1)
    # basic LSTM cell
    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, reuse=need_reuse)
    cells = []
    for i in range(num_layers):
      with tf.variable_scope('layer' + str(i)):
        if not need_reuse:
          if i == 0:
            cells.append(my_dropout.MyDropoutWrapper(lstm_cell(), 
                                          input_keep_prob=1-drop_i,
                                          state_keep_prob=1-drop_h,
                                          output_keep_prob=1-drop_o,
                                          variational_recurrent=True,
                                          input_size=highway_size,
                                          dtype=tf.float32))
          else:
            cells.append(my_dropout.MyDropoutWrapper(lstm_cell(),
                                          state_keep_prob=1-drop_h,
                                          output_keep_prob=1-drop_o,
                                          variational_recurrent=True,
                                          input_size=hidden_size,
                                          dtype=tf.float32))
        else:
          cells.append(lstm_cell())
    self.cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope('lstm_rnn', reuse=need_reuse):
      outputs, self.state = tf.contrib.rnn.static_rnn(self.cell, lstm_input, 
        dtype=tf.float32, initial_state=self.init_state)
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
      
    # finally we predict the next word according to a softmax normalization
    with tf.variable_scope('softmax_params', reuse=need_reuse):
      weights = tf.get_variable('weights', [word_vocab_size, hidden_size], 
              initializer=tf.random_uniform_initializer(-init_scale, init_scale), 
              dtype=tf.float32)
      biases = tf.get_variable('biases', [word_vocab_size], 
        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
        dtype=tf.float32)
    
    # and compute the cross-entropy between labels and predictions
    logits = tf.matmul(output, tf.transpose(weights)) + biases
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
              [logits],
              [tf.reshape(self.y, [-1])],
              [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self.cost = tf.reduce_sum(loss) / batch_size
    
    if not need_reuse:
      tvars = tf.trainable_variables()
      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars 
                          if 'bias' not in v.name and 'Bias' not in v.name]) * weight_decay
      self.full_cost = self.cost + l2_loss


class Train(Model):
  '''for training we need to compute gradients'''
  def __init__(self, config, word_ix_to_pat_ixs):
    super(Train, self).__init__(config, word_ix_to_pat_ixs)
    self.clear_pat_embedding_padding = tf.scatter_update(self.pat_embedding, 
      [0], tf.constant(0.0, shape=[1, config.pat_emb_dim], dtype=tf.float32))
    
    self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.full_cost, tvars), 
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars),
      global_step=tf.contrib.framework.get_or_create_global_step())
    
    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  # this will update the learning rate
  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def model_size():
  '''finds the total number of trainable variables a.k.a. model size'''
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


if __name__ == '__main__':
  config = Config()
  args = parse_args()
  train_data, valid_data, test_data, word_ix_to_pat_ixs = read_data(args, config)
  train_raw_data, valid_raw_data, test_raw_data = read_raw_data(args, config)
 
  train = Train(config, word_ix_to_pat_ixs)
  print('Model size is: ', model_size())

  valid = Model(config, word_ix_to_pat_ixs, need_reuse=True)

  test_config = copy.deepcopy(config)
  test_config.batch_size = 1
  test_config.ssm = 0
  test = Model(test_config, word_ix_to_pat_ixs, need_reuse=True)

  saver = tf.train.Saver()

  if args.is_train == '1':
    num_epochs = config.num_epochs
    display_freq = 100
    init = tf.global_variables_initializer()
    learning_rate = config.learning_rate

    with tf.Session() as sess:
      sess.run(init)
      sess.run(train.clear_pat_embedding_padding)
      prev_perplexity = float('inf')

      for epoch in range(num_epochs):
        start_time = time.time()
        train.assign_lr(sess, learning_rate)

        iters = 0
        costs = 0

        train_batches = batch_producer(train_data, train_raw_data, 
                                       config.batch_size, config.num_steps)
        training_state = None

        for batch in train_batches:
          my_x = np.empty(
            [config.batch_size, config.num_steps, config.max_word_len], 
            dtype=np.int32)

          # split words into patterns
          for t in range(config.num_steps):
            for i in range(config.batch_size):
              my_x[i, t] = word_ix_to_pat_ixs[batch[0][i, t]]

          # train the model on current batch
          if not training_state: training_state = sess.run(train.init_state)
          _, c, training_state, my_lr = sess.run(
            [train.train_op, train.cost, train.state, train.lr],
            feed_dict={train.x: my_x, train.y: batch[3], 
            train.init_state: training_state})
          sess.run(train.clear_pat_embedding_padding)

          costs += c
          if iters % (display_freq * config.num_steps) == 0 and iters != 0:
            print('step =', iters/config.num_steps, end=', ')
            print('perplexity =', np.exp(costs / iters), end=', ')
            print('learning rate =', my_lr, end=', ')
            print('speed =', 
              round(iters * config.batch_size / (time.time() - start_time)), 
              ' wps')

          iters += config.num_steps

        speed = round(iters * config.batch_size / (time.time() - start_time))
          print('epoch ', epoch + 1, end = ': ')
          print('train ppl =', np.exp(costs / iters), end=', ')
          print('lr =', my_lr, end=', ')

          # Get validation set perplexity
          valid_costs = 0
          valid_state = None
          valid_iters = 0

          valid_batches = batch_producer(valid_data, valid_raw_data,
                                         config.batch_size, config.num_steps)

          for valid_batch in valid_batches:
            my_valid_x = np.empty(
              [config.batch_size, config.num_steps, config.max_word_len], 
              dtype=np.int32)

            for t in range(config.num_steps):
              for i in range(config.batch_size):
                my_valid_x[i, t] = word_ix_to_pat_ixs[valid_batch[0][i, t]]

            if not valid_state: valid_state = sess.run(valid.init_state)
            c, valid_state = sess.run([valid.cost, valid.state], 
               feed_dict={valid.x: my_valid_x, valid.y: valid_batch[3], 
               valid.init_state: valid_state})

            valid_costs += c
            valid_iters += config.num_steps

          cur_perplexity = np.exp(valid_costs / valid_iters)
          print('valid ppl =', cur_perplexity, end=', ')
          print('speed =', speed)

          if prev_perplexity - cur_perplexity < 0:
            learning_rate *= config.lr_decay
          prev_perplexity = cur_perplexity

        # Get test set perplexity after training is done
        test_costs = 0
        test_state = None
        test_iters = 0

        test_batches = batch_producer(test_data, test_raw_data,
                                      test_config.batch_size, test_config.num_steps)

        for test_batch in test_batches:
          my_test_x = np.empty(
            [test_config.batch_size, test_config.num_steps, 
             test_config.max_word_len], 
            dtype=np.int32)

          for t in range(config.num_steps):
            for i in range(1):
              my_test_x[i, t] = word_ix_to_pat_ixs[test_batch[0][i, t]]

          if not test_state: test_state = sess.run(test.init_state)
          c, test_state = sess.run([test.cost, test.state], 
            feed_dict={test.x: my_test_x, test.y: test_batch[3], 
            test.init_state: test_state})

          test_costs += c
          test_iters += test_config.num_steps

        print('-' * 80)
        print('Test set perplexity =', np.exp(test_costs / test_iters))

        save_path = saver.save(sess, os.path.join(args.save_dir, args.prefix + '-model.ckpt'))
        print('Model saved in file: %s' % save_path)

    else:
      with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, os.path.join(args.save_dir, args.prefix + '-model.ckpt'))
        print('Model restored.')

        # Get test set perplexity
        test_costs = 0
        test_state = None
        test_iters = 0

        test_batches = batch_producer(test_data, test_raw_data,
                                      test_config.batch_size, test_config.num_steps)

        for test_batch in test_batches:
          my_test_x = np.empty(
            [test_config.batch_size, test_config.num_steps, 
             test_config.max_word_len], 
            dtype=np.int32)

          for t in range(test_config.num_steps):
            for i in range(test_config.batch_size):
              my_test_x[i, t] = word_ix_to_pat_ixs[test_batch[0][i, t]]

          if not test_state: test_state = sess.run(test.init_state)
          c, test_state = sess.run([test.cost, test.state], 
            feed_dict={test.x: my_test_x, test.y: test_batch[3], 
                       test.init_state: test_state})

          test_costs += c
          test_iters += test_config.num_steps

        print('Test set perplexity =', np.exp(test_costs / test_iters))