import numpy as np
import tensorflow as tf

#####################################################################################
#-----Creat Data(training data, validation data and testing data)---------------------


#function to transform a number(str type) to thirty bits(more details in readme.txt)
def format_number(n):
    high = []
    middle = []
    low = []
    for i in range(10):
        high.append(0.)
        middle.append(0.)
        low.append(0.)

    s = str(n)

    if len(s) == 1:
        high[0] = 1.
        middle[0] = 1.
        low[int(s[0])] = 1.
    elif len(s) == 2:
        high[0] = 1.
        middle[int(s[0])] = 1.
        low[int(s[1])] = 1.
    else:
        high[int(s[0])] = 1.
        middle[int(s[1])] = 1.
        low[int(s[2])] = 1.

    high.extend(middle)
    high.extend(low)

    return high

#function to transform each of the two numbers(str type)
# to thirty bits and join them together as a 60 bits list
def format_two_number(n,m):
    a = format_number(n)
    b = format_number(m)
    a.extend(b)
    return a

#function to transform the output(str type) to 32 bits
def format_output_number(n):
    first = []
    second = []
    third = []
    forth = []

    first.append(0.)
    first.append(0.)

    for i in range(10):
        second.append(0.)
        third.append(0.)
        forth.append(0.)

    s = str(n)

    if len(s) == 1:
        first[0] =1.
        second[0] = 1.
        third[0] = 1.
        forth[int(s[0])] = 1.
    elif len(s) == 2:
        first[0] = 1.
        second[0] = 1.
        third[int(s[0])] = 1.
        forth[int(s[1])] = 1.
    elif len(s) == 3:
        first[0] = 1.
        second[int(s[0])] = 1.
        third[int(s[1])] = 1.
        forth[int(s[2])] = 1.
    else:
        first[1] = 1.
        second[int(s[1])] = 1.
        third[int(s[2])] = 1.
        forth[int(s[3])] = 1.

    first.extend(second)
    first.extend(third)
    first.extend(forth)
    return first

X = []


#creat the data
for i in range(1000):
    if i%100 == 0:
        print "loading data",i/10.0,"%"
    for j in range(1000):
        a = [i, j, i+j]
        #the length of X is 1000*1000    and X[k][0]+X[k][1] = X[k][2]
        X.append(a)

X_format = []


#transform the created data to bits type as describing in "readme.txt"
for i in range(1000*1000):
    if i % 10000 == 0:
        print "format data", i / 10000.0, "%"
    data = []
    format_input = format_two_number(X[i][0],X[i][1])
    format_output = format_output_number(X[i][2])

    data.append(format_input)
    data.append(format_output)

    X_format.append(data)
np.random.shuffle(X_format)

del X


#training, validation and testing data
X_training = X_format[0:600000]
X_validation = X_format[600000:800000]
X_testing = X_format[800000:]

del X_format




#transform the data to the numpy array type
train_X = np.zeros((600000,60))
train_Y = np.zeros((600000,32))
validation_X = np.zeros((200000,60))
validation_Y = np.zeros((200000,32))
testing_X = np.zeros((200000,60))
testing_Y = np.zeros((200000,32))

print "please wait a minute for data transformation"
for i in range(600000):
    for j in range(60):
        train_X[i][j] = X_training[i][0][j]
for i in range(600000):
    for j in range(32):
        train_Y[i][j] = X_training[i][1][j]
for i in range(200000):
    for j in range(60):
        validation_X[i][j] = X_validation[i][0][j]
for i in range(200000):
    for j in range(32):
        validation_Y[i][j] = X_validation[i][1][j]
for i in range(200000):
    for j in range(60):
        testing_X[i][j] = X_testing[i][0][j]
for i in range(200000):
    for j in range(32):
        testing_Y[i][j] = X_testing[i][1][j]

del X_training
del X_testing
del X_validation
#######################################################################################
#-----------------------------------------------------------------------------------




#caculate if the two np array type number(output, 32bits) are same
def equalList(a,b):
    a1 = a[0:2]
    a2 = a[2:12]
    a3 = a[12:22]
    a4 = a[22:32]
    b1 = b[0:2]
    b2 = b[2:12]
    b3 = b[12:22]
    b4 = b[22:32]
    if a1.argmax()== b1.argmax() \
        and a2.argmax() == b2.argmax() \
        and a3.argmax() == b3.argmax() \
        and a4.argmax()== b4.argmax():
        return True

#caculate the accuracy of between the predict and the real value of output
def accuracy(predict_y, real_y):
    p = 0.
    l = len(predict_y)
    for i in range(l):
        if equalList(predict_y[i],real_y[i]):
            p +=1
    return p/l


epoch = 50
batch_size = 50
sess = tf.InteractiveSession()


#the structure of networks
x = tf.placeholder("float", shape=[None, 60])
y_ = tf.placeholder("float", shape=[None, 32])

W1 = tf.Variable(tf.truncated_normal([60,2048], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[2048]))

h1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

W2 = tf.Variable(tf.truncated_normal([2048,32], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[32]))

y = tf.nn.sigmoid(tf.matmul(h1,W2) + b2)

#loss function
loss = tf.reduce_sum(0.5*(y_-y)*(y_-y))

#optimizer
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

sess.run(tf.initialize_all_variables())

#train
for i in range(epoch):
    for j in range(600000/batch_size):
        if (j%100 ==0):
            print "epoch:",i, "iteration:", j, "loss:", sess.run(loss,feed_dict={x: train_X[j*batch_size:j*batch_size+batch_size],
                                                                        y_: train_Y[j*batch_size:j*batch_size+batch_size]})

        train_step.run(feed_dict={x: train_X[j*batch_size:j*batch_size+batch_size],
                                  y_: train_Y[j*batch_size:j*batch_size+batch_size]})

    acc = 0.


    # caculate the accuracy of validtion set, because of my machine,
    # the time to caculate huge data is too long
    # so , the data was devided to ten parts,
    # the result is the mean value of the ten accuracy
    for j in range(10):
        acc = acc + accuracy(sess.run(y, feed_dict={x: validation_X[20000 * j:20000 * j + 20000]}),
                             validation_Y[20000 * j:20000 * j + 20000])
    print "validation accuracy = ", acc / 10


acc = 0.



#caculate the accuracy of testing set, because of my machine,
#the time to caculate huge data is too long
#so , the data was devided to ten parts,
# the result is the mean value of the ten accuracy
for j in range(10):
    acc = acc+accuracy(sess.run(y, feed_dict={x: testing_X[20000*j:20000*j+20000]}),
                       testing_Y[20000*j:20000*j+20000])
print "testing accuracy = ", acc/10
sess.close()


