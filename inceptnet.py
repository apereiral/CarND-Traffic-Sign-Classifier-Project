import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


EPOCHS = 10
BATCH_SIZE = 100

n_classes = 43


# InceptNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC ### NEED TO CHANGE HERE!!!!!! ###
#
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper with bias
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool( x, ksize=[1, k, k, 1],
        strides=[1, k, k, 1], padding='SAME')

def InceptNet(x, weights, biases):
    
    x = tf.reshape(x, (-1, 32, 32, 1))
        
    # Convolution 1x1 layer 1 - 32*32*1 to 32*32*10
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    # ReLu activation function
    conv1 = tf.nn.relu(conv1)
    
    # Convolution 3x3 layer 2 - 32*32*10 to 30*30*20
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    # ReLu activation function
    conv2 = tf.nn.relu(conv2)
    # Pooling layer 2 - 30*30*20 to 15*15*20
    conv2 = maxpool2d(conv2)
    
    # Convolution 6x6 layer 3 - 15*15*20 to 10*10*25
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    # ReLu activation function
    conv3 = tf.nn.relu(conv3)
    # Pooling layer 3 - 10*10*25 to 5*5*25
    conv3 = maxpool2d(conv3)
    
    # Branching and pooling layer 2 - 15*15*20 to 8*8*20
    conv2_branch = maxpool2d(conv2)
    
    # Flatten layer 3
    flt1 = flatten(conv3)
    # Flatten layer 2 branch
    flt2 = flatten(conv2_branch)
    # Stack flattened layers together
    flt = tf.concat(1, [flt1, flt2])
    #flt = flatten(conv3)
    
    # 1st Fully connected layer - 5*5*25 + 8*8*20 to 200
    fc1 = tf.add(tf.matmul(flt, weights['fully_connected_1']), biases['fully_connected_1'])
    # ReLu activation function
    fc1 = tf.nn.relu(fc1)
    
    # 2nd Fully connected layer - 200 to 100
    fc2 = tf.add(tf.matmul(fc1, weights['fully_connected_2']), biases['fully_connected_2'])
    # ReLu activation function
    fc2 = tf.nn.relu(fc2)
    
    # Final fully connected layer - 100 to 43
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

mu = 0
sigma = 0.1

layer_width = {
    'layer_1': 10,
    'layer_2': 20,
    'layer_3': 25,
    'fully_connected_1': 200,
    'fully_connected_2': 100 }
    
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
            shape=(1, 1, 1, layer_width['layer_1']), mean=mu, stddev=sigma)),
    'layer_2': tf.Variable(tf.truncated_normal(
            shape=(3, 3, layer_width['layer_1'], layer_width['layer_2']), mean=mu, stddev=sigma)),
    'layer_3': tf.Variable(tf.truncated_normal(
            shape=(6, 6, layer_width['layer_2'], layer_width['layer_3']), mean=mu, stddev=sigma)),
    'fully_connected_1': tf.Variable(tf.truncated_normal(
            shape=(5*5*25+8*8*20, layer_width['fully_connected_1']), mean=mu, stddev=sigma)),
    'fully_connected_2': tf.Variable(tf.truncated_normal(
            shape=(layer_width['fully_connected_1'], layer_width['fully_connected_2']), mean=mu, stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(
            shape=(layer_width['fully_connected_2'], n_classes), mean=mu, stddev=sigma)) }

biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1'])),
    'fully_connected_2': tf.Variable(tf.zeros(layer_width['fully_connected_2'])),
    'out': tf.Variable(tf.zeros(n_classes)) }

x = tf.placeholder(tf.float32, (None, 32, 32))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

fc2 = InceptNet(x, weights, biases)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, one_hot_y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    num_examples = len(X_data)
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples

def draw_subset(rate_train_val = 0.7):

    for label in range(n_classes):
        pickle_file = 'label' + str(label) + '.pickle'
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                dataset = data['dataset']
                dataset_label = data['dataset_label']
                X_train, X_val, y_train, y_val = train_test_split(
                    dataset, dataset_label, test_size=(1-rate_train_val))
                if label == 0:
                    X_train_subset = X_train
                    y_train_subset = y_train
                    X_val_subset = X_val
                    y_val_subset = y_val
                else:
                    X_train_subset = np.append(X_train_subset, X_train, axis=0)
                    y_train_subset = np.append(y_train_subset, y_train, axis=0)
                    X_val_subset = np.append(X_val_subset, X_val, axis=0)
                    y_val_subset = np.append(y_val_subset, y_val, axis=0)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    X_train_subset, y_train_subset = shuffle(X_train_subset, y_train_subset)
    X_val_subset, y_val_subset = shuffle(X_val_subset, y_val_subset)
    
    return X_train_subset, y_train_subset, X_val_subset, y_val_subset

if __name__ == '__main__':

    # Load data
    X_train, y_train, X_val, y_val = draw_subset()
    num_examples = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train model
        for i in range(EPOCHS):
            
            X_train, y_train = shuffle(X_train, y_train)
            
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            print("EPOCH {} ...".format(i+1))
            train_loss, train_acc = evaluate(X_train, y_train)
            print("Train loss = {:.3f}".format(train_loss))
            print("Train accuracy = {:.3f}".format(train_acc))
            val_loss, val_acc = evaluate(X_val, y_val)
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        ## Evaluate on the test data
        #test_loss, test_acc = eval_data(mnist.test) ### CHANGE HERE!!! ###
        #print("Test loss = {:.3f}".format(test_loss)) ### CHANGE HERE!!! ###
        #print("Test accuracy = {:.3f}".format(test_acc)) ### CHANGE HERE!!! ###