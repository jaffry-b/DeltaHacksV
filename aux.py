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
	return ((ninfo['time']-pinfo['time']) + 0.5*(ninfo['x_pos']-pinfo['x_pos']) + flag)

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


