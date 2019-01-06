import os
import sys, getopt
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings

warnings.filterwarnings('ignore')
np.random.seed(8455)  # Let numpy initialize the seed from the machine random entropy source
st0 = np.random.get_state()  # Retrieve the initial state to be able to reproduce the results

# state = array [x,y,z,t] with x,y,z,t in {0, ... ,9}
# predator 1 location - prey location = [x,y]
# predator 2 location - prey location = [z,t]
# action = tuple (a,b) with a,b in {0, ... ,4}
# with action 0 = (0,0), action 1 = (1,0), action 2 = (-1,0), action 3 = (0,1), action 4 = (0,-1)
# mapping that simulates the experiment
# input = state + action
# output = new state + reward + boolean = true if capture (has to ne in coordination)

possibleactions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def manhattandistance(a, b):
    return min(a, 10 - a) + min(b, 10 - b)


def mapping(state, action):
    captured = False
    statefinal = [0, 0, 0, 0]
    reward = -1

    # checking the state has all its values in {0, ... ,9}
    if (state[0] < 0) or (state[0] > 9) or (state[1] < 0) or (state[1] > 9) or (state[2] < 0) or \
            (state[2] > 9) or (state[3] < 0) or (state[3] > 9):
        print("Relative positions out of bound")

    # moving the predators on the torus according to action
    statefinal[0] = (state[0] + possibleactions[action[0]][0]) % 10
    statefinal[1] = (state[1] + possibleactions[action[0]][1]) % 10
    statefinal[2] = (state[2] + possibleactions[action[1]][0]) % 10
    statefinal[3] = (state[3] + possibleactions[action[1]][1]) % 10

    # reward for catching the prey in coordination
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0) \
            and (manhattandistance(statefinal[2], statefinal[3]) != 0):
        reward = 75
        captured = True

    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) != 0) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = 75
        captured = True

    # penalty for both predators jumping on the prey
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = -75
        for i in range(0, 4):
            statefinal[i] = int(
                np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1, replace=True,
                                 p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    # penalty for one predators jumping on the prey while being alone
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) > 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0):
        reward = -75
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1, \
                                                 replace=True, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    if (manhattandistance(state[0], state[1]) > 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = -75
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1, \
                                                 replace=True, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    # moving the prey randomly
    r = int(
        np.random.choice([0, 1, 2, 3, 4], size=1, replace=True,
                         p=[0.2, 0.2, 0.2, 0.2, 0.2]))
    statefinal[0] = (statefinal[0] - possibleactions[r][0]) % 10
    statefinal[1] = (statefinal[1] - possibleactions[r][1]) % 10
    statefinal[2] = (statefinal[2] - possibleactions[r][0]) % 10
    statefinal[3] = (statefinal[3] - possibleactions[r][1]) % 10

    return (statefinal, reward, captured)


def q_learning(rseed):
    # Let numpy initialize the seed from the machine random entropy source
    np.random.seed(rseed)
    # Retrieve the initial state to be able to reproduce the results
    st0 = np.random.get_state()
    # generating the 100 starting states
    startingstates = np.zeros((100, 4), dtype=np.int)
    for i in range(0, 100):
        for j in range(0, 4):
            startingstates[i][j] = int(np.random.choice(range(0, 10), size=1, replace=True))

    # params of the RL algo
    N = 500000
    alpha = 0.3
    gamma = 0.9

    stepsarray = np.zeros(N, dtype=np.float)
    captured = False
    start_time = time.time()

    # initialisation of the Q values
    # Q = numpy array where multi index = ( state[0], ... , state[3], action predator 1, action predator 2 )
    Q = np.zeros((10, 10, 10, 10, 5, 5), dtype=np.float)
    for k in range(0, 10):
        for l in range(0, 10):
            for m in range(0, 10):
                for n in range(0, 10):
                    for q in range(0, 5):
                        for r in range(0, 5):
                            Q[k][l][m][n][q][r] = 75

    # best action and Q value derived from Q in a given state
    def bestaction(state):
        M = Q[state[0]][state[1]][state[2]][state[3]][0][0]
        act = (0, 0)
        for k in range(0, 5):
            for l in range(0, 5):
                if (Q[state[0]][state[1]][state[2]][state[3]][k][l] > M):
                    M = Q[state[0]][state[1]][state[2]][state[3]][k][l]
                    act = (k, l)
        return (act, M)

    # Q learning algo
    for i in range(0, N):

        # initial state and epsilon for the runs 0 to 99, 500 to 599, 1000 to 1099, ...
        if (i % 1000 < 500):
            state = startingstates[i % 100]
            epsilon = 0.2
        # initial state and epsilon for the runs 100 to 499, 600 to 999, 1100 to 1499, ...
        else:
            # for j in range(0,4):
            # state[j] = int ( np.random.choice( [0,1,2,3,4,5,6,7,8,9] , size=1 , \
            # replace=True , p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] ) )
            state = startingstates[i % 100]
            epsilon = 0

        captured = False
        steps = 0

        while (captured != True):

            # choose action according to epsilon greedy policy
            if (np.random.uniform() < epsilon):
                a = (int(np.random.choice([0, 1, 2, 3, 4], size=1, replace=False, p=[0.2, 0.2, 0.2, 0.2, 0.2])), \
                     int(np.random.choice([0, 1, 2, 3, 4], size=1, replace=False, p=[0.2, 0.2, 0.2, 0.2, 0.2])))
            else:
                a = bestaction(state)[0]

            # observe result from action a
            (state2, reward, captured) = mapping(state, a)

            # update Q
            Q[state[0]][state[1]][state[2]][state[3]][a[0]][a[1]] = \
                (1 - alpha) * Q[state[0]][state[1]][state[2]][state[3]][a[0]][a[1]] + \
                alpha * reward + alpha * gamma * bestaction(state2)[1]

            state = state2

            steps += 1

        stepsarray[i] = steps

        if (i % 10000 == 0):
            print("Episode", i + 1, "finished in", steps, "steps.")

        i += 1

    print("The Q learning algo took --- %s seconds ---" % (time.time() - start_time))

    return stepsarray


def main():
    import multiprocessing as mp
    seeds = [8642, 1489, 9952, 8995, 1962, 8483, 2021, 2161, 9628, 7462]
    trials = 10
    episodes = 500000
    results = []
    pool = mp.Pool(processes=2)
    for i in range(trials):
        results.append(pool.apply_async(q_learning, (seeds[i],)))
    pool.close()
    pool.join()

    count_result = 0
    episodes_step_arr = np.zeros((trials, episodes))
    for tr in range(trials):
        episodes_step_arr[tr] = results[count_result].get()
        count_result += 1

    avg_steps = []
    for ts in range(episodes):
        avg_steps.append(np.average(episodes_step_arr[:, ts]))

    # charts
    style = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-',
             'k-', 'b--', 'g--', 'r--', 'c--', ]
    plt.plot(range(0, episodes), avg_steps, style[2])
    M1 = np.amin(avg_steps)
    M2 = np.amax(avg_steps)
    plt.xlabel('Time')
    plt.ylabel('Steps')
    plt.title('MDP - Number of steps before capture per episode over time')
    plt.axis([0, episodes, M1, M2])
    plt.show()


if __name__ == "__main__":
    main()