import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    """
    Class for the generative model.
    """

    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.002, reward_gamma=0.95, grad_clip=5.0):
        """Sets parameters and defines the model architecture."""

        """
        Set specified parameters
        """
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.1
        self.grad_clip = 5.0

        # for tensorboard
        self.g_count = 0

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))
        self.x = tf.placeholder(  # true data, not including start token
            tf.int32, shape=[self.batch_size, self.sequence_length])
        self.rewards = tf.placeholder(  # rom rollout policy and discriminator
            tf.float32, shape=[self.batch_size, self.sequence_length])

        """
        Define generative model
        """
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length]) # get from rollout policy and discriminator

        # processed for batch
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                              value=tf.nn.embedding_lookup(self.g_embeddings,
                                                           self.x))
            self.processed_x = tf.stack(  # seq_length x batch_size x emb_dim
                [tf.squeeze(input_, [1]) for input_ in inputs])
        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t/self.temperature))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        """
        Supervized Pretraining
        """
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        g_logits = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions, g_logits):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(
                i, tf.nn.softmax(o_t))  # batch x vocab_size
            g_logits = g_logits.write(i, o_t)  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions, g_logits

        _, _, _, self.g_predictions, self.g_logits = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(
                           self.g_embeddings, self.start_token),
                       self.h0, g_predictions, g_logits))

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.g_logits = tf.transpose(
            self.g_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions,
                                            [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)


        self.s_pretrain_loss = tf.summary.scalar('gen_pretrain_loss', self.pretrain_loss)

        pretrain_opt = self.g_optimizer(self.learning_rate)  # training updates
        self.pretrain_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(
            zip(self.pretrain_grad, self.g_params))

        """
        Unsupervised Training
        """

        with tf.name_scope('gen_training'):
            self.g_loss = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) *
                    tf.log(
                        tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                    )
                    , 1
                ) * tf.reshape(self.rewards, [-1])
            )

            self.s_g_loss = tf.summary.scalar('gen_g_loss', self.g_loss)

            g_opt = self.g_optimizer(self.learning_rate)

            self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
            self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

        return


    def generate_gan_summary(self, sess, x, reward):
        _summ = sess.run(
            tf.summary.merge([
                self.s_g_loss
            ]),
            feed_dict={
                self.x: x,
                self.rewards: reward
            }
        )
        cur_g_count = self.g_count
        #self.g_count += 1
        return cur_g_count, _summ

    def generate(self, session, class_labels=None, label_input=False):
        """generate samples
        
        Arguments:
            session: TensorFlow session
            class_labels: list of class labels
            label_input: whether to use class labels as input
        """
        if label_input:
            # if using class labels as input, modify start_token
            start_token_value = session.run(self.start_token)  # get actual token value first
            start_tokens = [start_token_value[i] * int(class_labels[i]) for i in range(self.batch_size)]
            feed_dict = {
                self.start_token: start_tokens
            }
        else:
            feed_dict = {}
        
        # add class_label to feed_dict
        if hasattr(self, 'class_label_ph'):
            feed_dict[self.class_label_ph] = class_labels
            
        outputs = session.run([self.gen_x], feed_dict=feed_dict)
        return outputs[0]

    def pretrain_step(self, session, x, class_labels=None):
        """Performs a pretraining step on the generator."""
        if class_labels is not None:
            start_token_value = session.run(self.start_token)  # get actual token value first
            start_tokens = [start_token_value[i] * int(class_labels[i]) for i in range(self.batch_size)]
            feed_dict = {
                self.start_token: start_tokens,
                self.x: x
            }
        else:
            feed_dict = {self.x: x}
        
        if hasattr(self, 'class_label_ph'):
            feed_dict[self.class_label_ph] = class_labels
            
        outputs = session.run([self.pretrain_updates, self.pretrain_loss,
                               self.g_predictions], feed_dict=feed_dict)
        return outputs

    def generator_step(self, sess, samples, rewards, class_labels=None):
        """Performs a training step on the generator."""
        if class_labels is not None:
            start_token_value = sess.run(self.start_token)  # get actual token value first
            start_tokens = [start_token_value[i] * int(class_labels[i]) for i in range(self.batch_size)]
            feed_dict = {
                self.start_token: start_tokens,
                self.x: samples,
                self.rewards: rewards
            }
        else:
            feed_dict = {self.x: samples, self.rewards: rewards}  
            
        if hasattr(self, 'class_label_ph'):
            feed_dict[self.class_label_ph] = class_labels
            
        _, g_loss = sess.run([self.g_updates, self.g_loss], feed_dict=feed_dict)
        return g_loss

    def init_matrix(self, shape):
        """Returns a normally initialized matrix of a given shape."""
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        """Returns a vector of zeros of a given shape."""
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        """Defines the recurrent process in the LSTM."""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        """Defines the output part of the LSTM."""

        self.Wo = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        """Sets the optimizer."""
        return tf.train.AdamOptimizer(*args, **kwargs)
