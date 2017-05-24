import tensorflow as tf
import numpy as np
import prepare, sys

X_train,X_lengths, Y_train, dimen, n_steps, X_test, X_test_lengths, y_test, sentences = prepare.main()


def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (tf.cast(length, tf.int32) - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

def attention(Y, dimen):
    # [batch_size x seq_len x dim]  -- hidden states
    #Y = tf.constant(np.random.randn(batch_size, seq_len, dim), tf.float32)
    # [batch_size x dim]            -- h_N
    #h = tf.constant(np.random.randn(batch_size, dim), tf.float32)
    initializer = tf.random_uniform_initializer()
    W = tf.get_variable("weights_Y", [dimen, dimen], initializer=initializer)
    w = tf.get_variable("weights_w", [dimen], initializer=initializer)

    # [batch_size x seq_len x dim]  -- tanh(W^{Y}Y)
    M = tf.tanh(tf.einsum("aij,jk->aik", Y, W))
    # [batch_size x seq_len]        -- softmax(Y w^T)
    a = tf.nn.softmax(tf.einsum("aij,j->ai", M, w))
    # [batch_size x dim]            -- Ya^T
    r = tf.einsum("aij,ai->aj", Y, a)
    return r, a

classes =2
target = tf.placeholder(tf.float32, [None, classes],name ="y_true")
data = tf.placeholder(tf.float32, [None, n_steps,dimen], name ="x_true") #Number of examples, number of input, dimension of each input
sl = tf.placeholder(tf.float32, [None])
num_hidden = 4
learning_rate =5e-3
#cell = tf.contrib.rnn.GRUCell(num_hidden)
cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_hidden)
#cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0,  output_keep_prob=.90)
outputs, last_states = tf.nn.dynamic_rnn(cell = cell, dtype = tf.float32, sequence_length = sl, inputs = data)
val = tf.transpose(outputs, [1, 0, 2])#[10,32]
#last = _last_relevant(outputs, sl)#[1,32]
last, att = attention(outputs, num_hidden)
W2 = tf.get_variable("M",shape=[num_hidden, classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
bias = tf.Variable(tf.random_normal([classes]))
comp = tf.matmul(last,W2) + bias#[1,32][32,10]=[1,10]
wes = tf.nn.softmax(comp)



def coster(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=1)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=1))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=0)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=0)
    return tf.reduce_mean(cross_entropy)

cost = coster(wes, target)
correct_prediction = tf.equal(tf.argmax(wes, 1), tf.argmax(target, 1))
# Calculate accuracy for  test set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(cost)
# clip gradients
for i, (grad, var) in enumerate(gradients):
    # clip gradients by norm
    if grad is not None:
        gradients[i] = (tf.clip_by_norm(grad, 5), var)
train_op = optimizer.apply_gradients(gradients)'''

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess = tf.Session()
summary_writer = tf.summary.FileWriter("/tmp/charLevel")
summary_writer.add_graph(sess.graph)
sess.run(init_op)
batch_size = int(len(X_train))
no_of_batches = int(len(X_train) / batch_size)
epoch = 1000
modelName = 'inflationModel'
try:
    saver = tf.train.import_meta_graph(modelName+".meta")
    print("Loading variables from '%s'." % modelName)
    saver.restore(sess, tf.train.latest_checkpoint('./trainedModel'))
    print("Success: Loaded variables from '%s'." % modelName)
except IOError:
    print("Not found: Creating new '%s'." % modelName)
#training & testing modes
mode = 1
if mode == 0:
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out, leno = X_train[ptr:ptr+batch_size], Y_train[ptr:ptr+batch_size], X_lengths[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(train_op, {data: inp, target: out, sl:leno})
        costsss =float(sess.run(cost,{data: X_train, target: Y_train, sl:X_lengths}))
        testAcc = sess.run(accuracy, {data: X_test, target: y_test, sl:X_test_lengths})
        print ("Epoch "+str(i)+" Train set cost: "+str(costsss)," Test cost: ", sess.run(cost, {data: X_test, target: y_test, sl:X_test_lengths}),"Accuracy:", testAcc)
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=costsss, tag="training_cost")
        summary_writer.add_summary(episode_summary, i)
        summary_writer.flush()
        if testAcc >0.54:
            saver.save(sess, modelName)
            #sys.exit();
if mode == 1:
    print("Test set Accuracy:", sess.run(accuracy, {data: X_test, target: y_test, sl:X_test_lengths}))
    for i in range(400):
            #for all corect predictions in the training set
            #find where attantion was concentrated on and print that
            words = sentences[i].split()
            accur = sess.run(accuracy, {data: [X_train[i]], target: [Y_train[i]], sl:[X_lengths[i]]})
            if accur>0.0:
                attention = sess.run(att, {data: [X_train[i]], target: [Y_train[i]], sl:[X_lengths[i]]})
                index = np.argmax(attention[0])
                if len(words)>index:
                    attendedWord = words[index]
                    print(prepare.underline(sentences[i], attendedWord))
    

