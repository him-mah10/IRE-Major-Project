import sys
import string
import numpy as np
import _pickle as cPickle

embed = {}

def get_train_file(train, output):
    train_data = []
    f = open(output,'w')
    with open(train, 'r') as f1:
        for line in f1:
            tokens = line.strip().split('\t')
            if(len(tokens) > 5):
                train_data.append(line)
    for i in range(len(train_data)):
        cnt = 0
        sent1 = train_data[i].strip().split('\t')
        for j in range(i+1, len(train_data)):
            sent2 = train_data[j].strip().split('\t')
            val = int(sent2[4]) - int(sent1[4])
            if (val <= 604800000):
                label = 0
                # print(str(sent1[2]) + "     " + str(sent2[2]))
                if (int(sent1[2]) == int(sent2[2])):
                    # print("hereee")
                    label = 1
                line = sent1[0] + '\t' + sent1[5] + '\t' + sent1[3] + '\t' + sent2[0] + '\t' + sent2[5] + '\t' + sent2[3] + '\t' + str(label) +'\t' + str(int(val/1000)) + '\n'
                f.write(line)
                cnt += 1
        # /print(str(cnt) + "------" + str(len(train_data) - i))
    f.close()


def get_word_embeddings(file1, file2, index, vocab_cnt):
    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_cnt, 100))
    with open(file1, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if (len(tokens) > 5):
                words = tokens[5].strip().split(' ')
                for word in words:
                    if word in index:
                        if index[word] not in embedding_matrix:
                            if word in embed:
                                embedding_matrix[index[word]] = embed[word]
                            else:
                                embedding_matrix[index[word]] = embed["<unknown>"]
    with open(file2, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if (len(tokens) > 5):
                words = tokens[5].strip().split(' ')
                for word in words:
                    if word in index:
                        if index[word] not in embedding_matrix:
                            if word in embed:
                                embedding_matrix[index[word]] = embed[word]
                            else:
                                embedding_matrix[index[word]] = embed["<unknown>"]
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix


def get_word_index(sent, index, vocab_cnt):
    ans = []
    sent = sent.strip().split(' ')
    for word in sent:
        if word not in index:
            index[word] = vocab_cnt
            vocab_cnt += 1
        ans.append(index[word])
    return ans, index, vocab_cnt


def get_distance_index(sent1, trigger1):
    index = {}
    cnt = 0
    ans = []
    sent = sent1.strip().split(' ')
    trigger = trigger1.strip().split(' ')
    # print(sent)
    for word2 in trigger:
        cnt = 0
        for word in sent:
            if word == word2:
                break
            cnt += 1
    prev = 0
    i = cnt
    if(i == len(sent)):
        # print("hello")
        return ans
    while i>=0:
        if sent[i] in trigger:
            index[sent[i]] = 0
        else:
            index[sent[i]] = prev-1
            prev -=1
        i-=1
    prev = 0
    for i in range(cnt+1, len(sent)):
        if sent[i] in trigger:
            index[sent[i]] = 0
        else:
            index[sent[i]] = prev+1
            prev +=1
    for word in sent:
        ans.append(index[word])
    return ans

def read_files(file, embed, index, vocab_cnt):
    sents =[]
    cnt = 0
    with open(file, 'r') as f:
        for line in f:
            cnt += 1
            tokens = line.strip().split('\t')
            sent1, index, vocab_cnt = get_word_index(tokens[1], index, vocab_cnt)
            sent1_dist = get_distance_index(tokens[1], tokens[2])
            trigger1, index, vocab_cnt = get_word_index(tokens[2], index, vocab_cnt)
            trigger1_dist = get_distance_index(tokens[2], tokens[2])
            sent2, index, vocab_cnt = get_word_index(tokens[4], index, vocab_cnt)
            sent2_dist = get_distance_index(tokens[4], tokens[5])
            trigger2, index, vocab_cnt = get_word_index(tokens[5], index, vocab_cnt)
            trigger2_dist = get_distance_index(tokens[5], tokens[5])
            if(sent1_dist == [] or sent2_dist == [] or trigger1_dist == [] or trigger2_dist == []):
                continue
            trigger_common_words = 0
            trig1 = tokens[2].strip().split(' ')
            trig2 = tokens[5].strip().split(' ')
            for val in trig1:
                if val in trig2:
                    trigger_common_words += 1
            sents.append((np.asarray(sent1).astype('int32'),np.asarray(sent1_dist).astype('int32'), np.asarray(trigger1).astype('int32'),np.asarray(trigger1_dist).astype('int32'), np.asarray(sent2).astype('int32'), np.asarray(sent2_dist).astype('int32'),
             np.asarray(trigger2).astype('int32'), np.asarray(trigger2_dist).astype('int32'), np.asarray(int(tokens[6])).astype('int32'), np.asarray(trigger_common_words).astype('int32'), np.asarray(int(tokens[7])).astype('int32')))
    return sents, index, vocab_cnt


with open('./glove.twitter.27B/glove.twitter.27B.100d.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        embedding = [float(x) for x in line[1:]]
        embed[line[0]] = np.asarray(embedding).astype('float32')
get_train_file('./train.txt', './final_train.txt')
get_train_file('./test.txt', './final_test.txt')
train = './final_train.txt'
test = './final_test.txt'
vocab_cnt = 0
index = {}
print("train,test created")
embed["<unknown>"] = np.zeros(100).astype('float32')
train_sents, index, vocab_cnt = read_files(train, embed, index, vocab_cnt)
print("train done")
test_sents, index, vocab_cnt = read_files(test, embed, index, vocab_cnt)
print("test done")
embedding_matrix = get_word_embeddings('./train.txt', './test.txt', index, vocab_cnt)
print("embedding matrix done")
with open('./dump_file', 'wb') as f:
    cPickle.dump(train_sents, f)
    cPickle.dump(test_sents, f)
    cPickle.dump(embedding_matrix, f)
