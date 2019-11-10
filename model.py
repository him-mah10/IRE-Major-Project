import tensorflow as tf

class Attention:
    def __init__(self, sequence_length, trigger_length, num_classes, vocab_size, word_embedding_size ,dist_embedding_size, hidden_size, attention_size, co_ref_size):
        #sentence1
        self.input1_text1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input1_text1')
        self.input1_text2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input1_text2')
        text1_length = self._length(self.input1_text1)

        #trigger1
        self.trigger1_text1 = tf.placeholder(tf.int32, shape=[None, trigger_length], name='trigger1_text1')
        self.trigger1_text2 = tf.placeholder(tf.int32, shape=[None, trigger_length], name='trigger1_text2')
        trigger1_text_length =  self._length(self.trigger1_text1)

        #sentence2
        self.input2_text1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input2_text1')
        self.input2_text2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input2_text2')
        text2_length = self._length(self.input2_text1)

        #trigger2
        self.trigger2_text1 = tf.placeholder(tf.int32, shape=[None, trigger_length], name='trigger2_text1')
        self.trigger2_text2 = tf.placeholder(tf.int32, shape=[None, trigger_length], name='trigger2_text2')
        trigger2_text_length =  self._length(self.trigger2_text1)

        #labels
        self.labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        self.bsz_size = tf.placeholder(tf.int32, name="bsz_size")

        #common words between triggers
        self.V_w = tf.placeholder(tf.float32, shape=[None, 1], name='trigger_common_words')
        self.V_d = tf.placeholder(tf.float32, shape=[None, 1], name='events_date_diff')

        initializer = tf.contrib.layers.xavier_initializer()

        #Embeddings
        self.W_em1 = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embedding_size]),trainable=True, name="W_em1")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])
        self.embedding_init = self.W_em1.assign(self.embedding_placeholder)
        self.W_em2 = tf.Variable(tf.random_uniform([77, dist_embedding_size], -1.0, 1.0), name="W_em2")

        #Embeddings for sent1
        self.embedded_words1 = tf.nn.embedding_lookup(self.W_em1, self.input1_text1)
        self.embedded_words2 = tf.nn.embedding_lookup(self.W_em2, self.input1_text2)
        self.embedded1_words = tf.concat([self.embedded_words1, self.embedded_words2], axis = 2)

        #Embeddings for sent2
        self.embedded_words1 = tf.nn.embedding_lookup(self.W_em1, self.input2_text1)
        self.embedded_words2 = tf.nn.embedding_lookup(self.W_em2, self.input2_text2)
        self.embedded2_words = tf.concat([self.embedded_words1, self.embedded_words2], axis = 2)

        #Embeddings for trigger1
        self.embedded_trigger_words1 = tf.nn.embedding_lookup(self.W_em1, self.trigger1_text1)
        self.embedded_trigger_words2 = tf.nn.embedding_lookup(self.W_em2, self.trigger1_text2)
        self.embedded1_trigger_words = tf.concat([self.embedded_trigger_words1, self.embedded_trigger_words2], axis = 2)

        #Embeddings for trigger2
        self.embedded_trigger_words1 = tf.nn.embedding_lookup(self.W_em1, self.trigger2_text1)
        self.embedded_trigger_words2 = tf.nn.embedding_lookup(self.W_em2, self.trigger2_text2)
        self.embedded2_trigger_words = tf.concat([self.embedded_trigger_words1, self.embedded_trigger_words2], axis = 2)

        #BiLSTM for sent1 and sent2
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        (self.output1_fw, self.output1_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded1_words,sequence_length=text1_length,dtype=tf.float32)
        (self.output2_fw, self.output2_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded2_words,sequence_length=text2_length,dtype=tf.float32)
        self.H1 = tf.concat([self.output1_fw, self.output1_bw], axis=2)
        # H1_reshape = tf.reshape(self.H1, [-1, 2 * hidden_size])
        self.H2 = tf.concat([self.output2_fw, self.output2_bw], axis=2)
        # H2_reshape = tf.reshape(self.H2, [-1, 2 * hidden_size])

        #BiLSTM for event trigger1 and event trigger2
        print("\n\n\n\n\n")
        (self.output1_fw, self.output1_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded1_trigger_words,sequence_length=trigger1_text_length,dtype=tf.float32)
        (self.output2_fw, self.output2_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded2_trigger_words,sequence_length=trigger2_text_length,dtype=tf.float32)

        #TODO : original length instead of trigger_length
        val1 = tf.slice(self.output1_fw, [0,trigger_length-1,0],[self.bsz_size, 1, hidden_size], name="val1")
        val2 = tf.slice(self.output1_bw, [0,0,0],[self.bsz_size, 1, hidden_size], name="val2")
        self.H1_trigger = tf.concat([val1, val2], axis=2)
        # H1_trigger_reshape = tf.reshape(self.H1, [-1, 2*hidden_size])
        val3 = tf.slice(self.output2_fw, [0,trigger_length-1,0],[self.bsz_size, 1, hidden_size], name="val3")
        val4 = tf.slice(self.output2_bw, [0,0,0],[self.bsz_size, 1, hidden_size], name="val4")
        self.H2_trigger = tf.concat([val3, val4], axis=2)
        # H2_trigger_reshape = tf.reshape(self.H2, [-1, 2*hidden_size])

        #-----------------------------------------------------------------
        #selectivegate (is it bsz*seq*2h or (bsz*seq)*2h??)
        w1 = tf.Variable(tf.random_normal([2*hidden_size, 2*hidden_size], stddev=0.01), name='w1')
        b1 = tf.Variable(tf.constant(0.1, shape=(1, 2*hidden_size)), name='b1')

        R1_c = self.H1 * tf.reshape(tf.tile(tf.reshape(self.H1_trigger, [1, -1, 2*hidden_size]), [sequence_length, 1, 1]), [-1, sequence_length, 2*hidden_size])
        # R1_c_reshaped = tf.reshape(R1_c, [sequence_length, -1, 2*hidden_size])
        alpha1 = tf.map_fn(lambda x: tf.math.tanh(tf.add(tf.matmul(x, w1), tf.tile(b1, [sequence_length, 1]))), R1_c)
        select1 = alpha1 * self.H1

        R2_c = self.H2 * tf.reshape(tf.tile(tf.reshape(self.H2_trigger, [1, -1, 2*hidden_size]), [sequence_length,1,1]), [-1, sequence_length, 2*hidden_size])
        # R2_c_reshaped = tf.reshape(R2_c, [sequence_length, -1, 2*hidden_size])
        alpha2 = tf.map_fn(lambda x: tf.math.tanh(tf.add(tf.matmul(x, w1),tf.tile(b1, [sequence_length, 1]))), R2_c)
        select2 = alpha2 * self.H2
        #-----------------------------------------------------------------------
        #attentive mechanism
        W_alpha = tf.Variable(tf.random_normal([2*hidden_size, attention_size], stddev=0.01), name='w_alpha')
        b_alpha = tf.Variable(tf.constant(0.1, shape=(1, attention_size)), name='b_alpha')
        V_alpha = tf.Variable(tf.random_normal([attention_size, 1], stddev=0.01), name='v_alpha')

        # select1_reshape = tf.reshape(select1, [sequence_length, -1, 2*hidden_size])
        u1 = tf.map_fn(lambda x : tf.matmul(tf.math.tanh(tf.add(tf.matmul(x, W_alpha), tf.tile(b_alpha, [sequence_length, 1]))), V_alpha), select1)
        beta1 = tf.nn.softmax(u1, axis = 1)
        latent1 = tf.reduce_sum(select1*beta1, axis = 1)
        V_em1 = tf.concat([latent1, tf.squeeze(self.H1_trigger)], axis=1)

        u2 = tf.map_fn(lambda x : tf.matmul(tf.math.tanh(tf.add(tf.matmul(x, W_alpha), tf.tile(b_alpha, [sequence_length, 1]))), V_alpha), select2)
        beta2 = tf.nn.softmax(u2, axis = 1)
        latent2 = tf.reduce_sum(select2*beta2, axis = 1)
        V_em2 = tf.concat([latent2, tf.squeeze(self.H2_trigger)], axis=1)

        #--------------------------------------------------------------------
        #co-reference decision
        W_ds = tf.Variable(tf.random_normal([8*hidden_size+2, co_ref_size], stddev=0.01), name='w_ds')
        b_ds = tf.Variable(tf.constant(0.1, shape=(1, co_ref_size)), name='b_ds')
        W_pro = tf.Variable(tf.random_normal([co_ref_size, 2], stddev=0.01), name='w_pro')
        b_pro = tf.Variable(tf.constant(0.1, shape=(1, 2)), name='b_pro')

        V_local = tf.concat([self.V_w, self.V_d], axis=1)
        V_pair = tf.concat([V_em1, V_em2, V_local], axis=1)
        # print("V_pair sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        # print(V_pair.get_shape().as_list())
        V_ds = tf.nn.relu(tf.add(tf.matmul(V_pair, W_ds), tf.tile(b_ds, [self.bsz_size, 1])))
        V_pro = tf.add(tf.matmul(V_ds, W_pro), tf.tile(b_pro, [self.bsz_size,1]))
        # print("V_em2 sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        # print(V_pro.get_shape().as_list())
        self.score = tf.argmax(V_pro, axis=1, output_type=tf.dtypes.int32, name="predictions")
        # print("sizeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        # print(self.score.get_shape().as_list())
        #loss and accuracy
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=V_pro, labels=self.labels)
        self.loss = tf.reduce_mean(losses)
        correct_predictions = tf.equal(self.score, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
