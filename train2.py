#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Rewards function and such
import numpy

# Info given: 
# {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}
def reward(ninfo, pinfo):
    # coins : N/A
    # flag : endgame, big plus
    # life : >0, big minus
    # score : N/A
    # stage : N/A
    # status : N/A
    # time : lose points for losing time, minus
    # world : N/A
    # x_pos : move right, plus
    flag = 0
    if ninfo['flag_get']:
        flag = 15
    if ninfo['life'] != pinfo['life']:
        return -15
    return ((ninfo['time']-pinfo['time']) + 3*(ninfo['x_pos']-pinfo['x_pos']) + flag)

# 0.95 discount rate
def discountrewards(rewards):
    discrewards = numpy.empty(len(rewards))
    cumreward = 0
    discrate = 0.95
    for i in reversed(range(len(rewards))):
        cumreward = rewards[i] + (cumreward * discrate)
        discrewards[i] = cumreward
    return discrewards

def discnormrewards(allrewards):
    alldiscrewards = []
    for rewards in allrewards:
        alldiscrewards.append(discountrewards(rewards))
    fullrewards = numpy.concatenate(alldiscrewards)
    rmean = fullrewards.mean()
    rstd = fullrewards.std()
    return [(discrewards - rmean)/rstd
for discrewards in alldiscrewards]


# In[2]:



## Base model to run the game, using random movements
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env.reset()


# In[3]:


#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
#import gym_super_mario_bros
#from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
#env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
#

# In[4]:


import numpy as np
import skimage 


def reset_graph(seed=828):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def preprocess_frame(frame):
    gray = skimage.color.rgb2gray(frame)

    normalized_frame = gray/255.0

    return normalized_frame



# In[8]:


import tensorflow as tf

with tf.device("gpu"):

    #architecture

    reset_graph()

    input_shape = env.observation_space.shape

    n1_hidden = 10
    n2_hidden = 10
    n3_hidden = 100
    n_outputs = 7# 7 stuff
    initializer = tf.variance_scaling_initializer()

    learning_rate = 0.01

    #build

    X = tf.placeholder(tf.float32, shape=[None, 240, 256, 3])



    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size=[8,8], padding ="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[4,4], strides=(4,4))

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size=[4,4], padding ="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2], strides=(2,2))

    pool2_flat = tf.contrib.layers.flatten(pool2)

    hidden = tf.layers.dense(pool2_flat,n3_hidden, activation=tf.nn.elu,
                            kernel_initializer=initializer)

    logits = tf.layers.dense(hidden, n_outputs,
                            kernel_initializer=initializer)
    outputs = tf.nn.softmax(logits)

    print(tf.shape(outputs))

    #Sampling

    action = tf.multinomial(tf.log(outputs),num_samples=1)

    init = tf.global_variables_initializer()
    #train
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = outputs, logits = logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)

    gradients = [grad for grad,variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []

    for grad,variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder,variable))
        training_op = optimizer.apply_gradients(grads_and_vars_feed)
        init =tf.global_variables_initializer()
        saver = tf.train.Saver()


# In[6]:





# In[9]:


n_games_per_update = 1
n_max_steps = 1000
n_iterations = 1000000
save_iterations = 25
discount_rate = 0.95
current_rewards = 0
with tf.Session() as sess:
    init.run()
    #saver.restore(sess, 'iter25_score_4893_mario.ckp.meta')
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration),end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            done = False
            oldi = {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}
            while oldi['life'] == 2:
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(-1, 240,256,3)})
                obs, rwd, done, info = env.step(action_val[0][0])
                creward = reward(info, oldi)
                oldi = info
                current_rewards.append(creward)
                current_gradients.append(gradients_val)
                #env.render()
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        all_rewards = discnormrewards(all_rewards)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./"+"iter"+str(iteration)+"_score_"+str(sum(current_rewards))+"_mario.ckpt")
env.close()


# In[ ]:





# In[ ]:




