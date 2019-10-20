import tensorflow as tf

class Attention:
    def __init__(self, sequence_length, trigger_length, num_classes, vocab_size, word_embedding_size ,dist_embedding_size, hidden_size, attention_size, decay_rate):
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

        #Embeddings
        self.W_em1 = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embedding_size]),trainable=True, name="W_em1")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])
        embedding_init = W.assign(embedding_placeholder)
        self.W_em2 = tf.Variable(tf.random_uniform([20, dist_embedding_size], -1.0, 1.0), name="W_em2")

        #Embeddings for sent1
        self.embedded_words1 = tf.nn.embedding_lookup(self.W_em1, self.input1_text1)
        self.embedded_words2 = tf.nn.embedding_lookup(self.W_em2, self.input1_text2)
        self.embedded1_words = tf.concat(self.embedded_words1, self.embedded_words2, axis = 1)

        #Embeddings for sent2
        self.embedded_words1 = tf.nn.embedding_lookup(self.W_em1, self.input2_text1)
        self.embedded_words2 = tf.nn.embedding_lookup(self.W_em2, self.input2_text2)
        self.embedded2_words = tf.concat(self.embedded_words1, self.embedded_words2, axis = 1)

        #Embeddings for trigger1
        self.embedded_trigger_words1 = tf.nn.embedding_lookup(self.W_em1, self.trigger1_text1)
        self.embedded_trigger_words2 = tf.nn.embedding_lookup(self.W_em2, self.trigger1_text2)
        self.embedded1_trigger_words = tf.concat(self.embedded_trigger_words1, self.embedded_trigger_words2, axis = 1)

        #Embeddings for trigger2
        self.embedded_trigger_words1 = tf.nn.embedding_lookup(self.W_em1, self.trigger2_text1)
        self.embedded_trigger_words2 = tf.nn.embedding_lookup(self.W_em2, self.trigger2_text2)
        self.embedded2_trigger_words = tf.concat(self.embedded_trigger_words1, self.embedded_trigger_words2, axis = 1)

        #BiLSTM for sent1 and sent2
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        (self.output1_fw, self.output1_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded1_words,sequence_length=text_length,dtype=tf.float32)
        (self.output2_fw, self.output2_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded2_words,sequence_length=text_length,dtype=tf.float32)
        self.H1 = tf.concat([self.output1_fw, self.output1_bw], axis=2)
        H1_reshape = tf.reshape(self.H1, [-1, 2 * hidden_size])
        self.H2 = tf.concat([self.output2_fw, self.output2_bw], axis=2)
        H2_reshape = tf.reshape(self.H2, [-1, 2 * hidden_size])

        #BiLSTM for event trigger1 and event trigger2
        (self.output1_fw, self.output1_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded1_trigger_words,sequence_length=trigger_text_length,dtype=tf.float32)
        (self.output2_fw, self.output2_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embedded2_trigger_words,sequence_length=trigger_text_length,dtype=tf.float32)
        self.H1 = tf.concat([self.output1_fw[trigger_text_length-1], self.output1_bw[0]], axis=2)
        H1_trigger_reshape = tf.reshape(self.H1, [-1, 2])
        self.H2 = tf.concat([self.output2_fw[trigger_text_length-1], self.output2_bw[0]], axis=2)
        H2_trigger_reshape = tf.reshape(self.H2, [-1, 2])

        #selectivegate
        

    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
