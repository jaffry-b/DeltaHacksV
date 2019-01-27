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
	return ((ninfo['time']-pinfo['time']) + (ninfo['x_pos']-pinfo['x_pos']) + flag)

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


from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)


# In[4]:


import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[8]:


import tensorflow as tf

with tf.device("gpu"):

    #architecture

    reset_graph()

    n_inputs = env.observation_space.shape[0]
    n1_hidden = 10
    n2_hidden = 10
    n3_hidden = 10
    n_outputs = 7# 7 stuff
    initializer = tf.variance_scaling_initializer()

    learning_rate = 0.01

    #build

    X = tf.placeholder(tf.float32, shape=[None,n_inputs])
    hidden1 = tf.layers.dense(X, n1_hidden, activation=tf.nn.elu,
                             kernel_initializer=initializer)
    hidden2 = tf.layers.dense(hidden1,n2_hidden ,activation=tf.nn.elu,
                            kernel_initializer=initializer)
    hidden3 = tf.layers.dense(hidden2,n3_hidden, activation=tf.nn.elu,
                            kernel_initializer=initializer)

    logits = tf.layers.dense(hidden3, n_outputs,
                            kernel_initializer=initializer)
    outputs = tf.nn.softmax(logits)


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


n_games_per_update = 10
n_max_steps = 1000
n_iterations = 250
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            done = False
            oldi = {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}
            while not done:
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(768, n_inputs)})
                obs, rwd, done, info = env.step(action_val[0][0])
                creward = reward(info, oldi)
                oldi = info
                current_rewards.append(creward)
                current_gradients.append(gradients_val)
                env.render()
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
            saver.save(sess, "./"+"iteration_"+Str(iteration)+"_mario".ckpt"+)
env.close()


# In[ ]:





# In[ ]:




