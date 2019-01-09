import os
import sys
import getopt
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings
from collections import *
import multiprocessing as mp

warnings.filterwarnings('ignore')

# state = array [x,y,z,t] with x,y,z,t in {0, ... ,9}
# predator 1 location - prey location = [x,y]
# predator 2 location - prey location = [z,t]
# action = tuple (a,b) with a,b in {0, ... ,4}
# with action 0 = (0,0), action 1 = (1,0), action 2 = (-1,0), action 3 = (0,1), action 4 = (0,-1)
# mapping that simulates the experiment
# input = state + action
# output = new state + reward + boolean = true if capture (has to be in coordination)
# reward [ r1 , r2 ] = one for each predator

possibleactions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
possibleactions2 = [
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 2), (-1, 1), (1, 1), (-2, 0),
    (2, 0), (-1, -1), (1, -1), (0, -2)
]


def manhattandistance(a, b):
    """
    function to calculate the manhattandistance
    :param a: x
    :param b: y
    :return: value of manhattan
    """
    return min(a, 10 - a) + min(b, 10 - b)


def mapping(state, action):
    """
    Function to move the predator and prey , plus calculate the reward
    :param state:
    :param action:
    :return: new state,rewards, and the informaiton about the capture
    """
    captured = False
    statefinal = [0, 0, 0, 0]
    reward = [-0.5, -0.5]

    # checking the state has all its values in {0, ... ,9}
    if (state[0] < 0) or (state[0] > 9) or (state[1] < 0) \
            or (state[1] > 9) or (state[2] < 0) \
            or (state[2] > 9) or (state[3] < 0) or (state[3] > 9):
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
        reward = [37.5, 37.5]
        captured = True

    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) != 0) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = [37.5, 37.5]
        captured = True

    # penalty for both predators jumping on the prey
    # penalty for both predators jumping on the same cell
    if (statefinal[0] == statefinal[2]) and (statefinal[1] == statefinal[3]):
        reward = [-50, -50]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(range(0, 10), size=1, replace=True))

    # penalty for one predators jumping on the prey while being alone
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) > 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0):
        reward = [-5, -5]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(
                range(0, 10), size=1, replace=True))

    if (manhattandistance(state[0], state[1]) > 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = [-5, -5]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(
                range(0, 10), size=1, replace=True))

    # moving the prey randomly
    r = int(np.random.choice(range(0, 5), size=1, replace=True))

    statefinal[0] = (statefinal[0] - possibleactions[r][0]) % 10
    statefinal[1] = (statefinal[1] - possibleactions[r][1]) % 10
    statefinal[2] = (statefinal[2] - possibleactions[r][0]) % 10
    statefinal[3] = (statefinal[3] - possibleactions[r][1]) % 10

    return (statefinal, reward, captured)


# generation of the rules
# Rule = a class with attributes: Pred1, Pred2, Action1, Action2, Value, nj


class Rule(object):
    def __init__(self, pred1, pred2, action1, action2, value, nj):
        self.Pred1 = pred1
        self.Pred2 = pred2
        self.Action1 = action1
        self.Action2 = action2
        self.Value = value
        self.nj = nj


def to_str_key(pred1, pred2, action1="", action2=""):
    """
    Function to return a string
    :param pred1:
    :param pred2:
    :param action1:
    :param action2:
    :return: string of key
    """
    key = str(pred1) + str(pred2) + str(action1) + str(action2)
    return key


def generate_rules():
    """
    Fucntion to generate the rules
    :return: dictionnary of rules
    """
    dict_rules = dict()

    # Coordinated states
    for x_agent_1 in range(0, 10):
        for y_agent_1 in range(0, 10):
            if x_agent_1 != 0 or y_agent_1 != 0:
                for x_agent_2 in range(0, 10):
                    for y_agent_2 in range(0, 10):
                        if ((x_agent_2 != 0) or (y_agent_2 != 0)) and (
                                (x_agent_2 != x_agent_1) or (y_agent_2 != y_agent_1)):
                            for action_1 in range(0, 5):
                                for action_2 in range(0, 5):
                                    if (manhattandistance((x_agent_2 - x_agent_1) % 10,
                                                          (y_agent_2 - y_agent_1) % 10) <= 2) \
                                            or ((manhattandistance(x_agent_1, y_agent_1) <= 2) and (
                                            manhattandistance(x_agent_2, y_agent_2) <= 2)):
                                        r = Rule((x_agent_1, y_agent_1), (x_agent_2, y_agent_2), action_1, action_2, 75,
                                                 2)
                                        key = to_str_key(x_agent_1, y_agent_1, x_agent_2, y_agent_2)
                                        dict_rules.setdefault(key, {}).update({to_str_key(action_1, action_2): r})

    # Uncoordinated states
    for x_agent in range(0, 10):
        for y_agent in range(0, 10):
            if ((x_agent != 0) or (y_agent != 0)):
                for action in range(0, 5):
                    r1 = Rule((x_agent, y_agent), (None, None), action, None, 75, 1)
                    r2 = Rule((None, None), (x_agent, y_agent), None, action, 75, 1)
                    key1 = to_str_key(x_agent, y_agent, None, None)
                    key2 = to_str_key(None, None, x_agent, y_agent)
                    dict_rules.setdefault(key1, {}).update({to_str_key(action, None): r1})
                    dict_rules.setdefault(key2, {}).update({to_str_key(None, action_1): r2})

    return dict_rules


# definition of the functions Q1 and Q2
# list_r is the list of the rules that are consistent with (state,action)
# and where predator 1 (resp. 2) is involved


def Q1(state, action, dic):
    """
    Function to update Q value 1
    :param state:
    :param action:
    :param dic: dictionnary of rules
    :return: new Q value 1
    """
    S = 0
    list_r = [v for k, v in dic.items()
              if v.Pred1 == (state[0], state[1])
              and v.Action1 == action[0]
              and (v.Action2 == action[1] or v.Action2 == None)]
    for rule in list_r:
        S += rule.Value / rule.nj
    return S


def Q2(state, action, dic):
    """
    Function to update Q value 2
    :param state:
    :param action:
    :param dic: dictionnary of rules
    :return: new Q value 2
    """
    S = 0
    list_r = [v for k, v in dic.items()
              if v.Pred2 == (state[2], state[3])
              and (v.Action1 == action[0] or v.Action1 == None)
              and v.Action2 == action[1]]
    for rule in list_r:
        S += rule.Value / rule.nj
    return S


# best combined action a that maximizes global payoff in a given state
def bestaction(dic):
    """
    Function to found the best action
    :param dic:
    :return: best action combined
    """
    tab = [[0, 0, 0.0], [1, 0, 0.0], [2, 0, 0.0], [3, 0, 0.0], [4, 0, 0.0]]
    ro = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    for action_1 in range(0, 5):
        poM = 0
        for action_2 in range(0, 5):
            po = 0

            rules = variable_elimination_type_1(dic, action_1, action_2)

            for rule in rules:
                if rule != None:
                    po += rule.Value
            if po > poM:
                poM = po
                tab[i][1] = j
                tab[i][2] = poM
                ro[i] = poM
        # finding best combined action
        m = np.argmax(ro)
        # act = [tab[m][0], tab[m][1]]
        max_indexes = np.where(ro == ro[m])[0]
        if len(max_indexes) == 1:
            action = max_indexes[0]
            act = [tab[action][0], tab[action][1]]
        else:
            action = np.random.choice(max_indexes)
            act = [tab[action][0], tab[action][1]]

    return act


def variable_elimination_type_1(dic, i, j):
    """
    Function to compute the variable elimination
    :param dic:
    :param i: action 1
    :param j: action 2
    :return:dictionnary containing our actions
    """
    rules = []
    key1 = to_str_key(i, j)
    key2 = to_str_key(i, None)
    key3 = to_str_key(None, j)
    dico_1 = dic[key1] if key1 in dic else None
    dico_2 = dic[key2] if key2 in dic else None
    dico_3 = dic[key3] if key3 in dic else None
    rules.extend((dico_1, dico_2, dico_3))
    return rules


def variable_elimination_type_2(dic, state):
    """
    Function to compute the variable elimination
    :param dic:
    :param state:
    :return: dictionnary containing our state
    """
    key1 = to_str_key(state[0], state[1], None, None)
    key2 = to_str_key(None, None, state[2], state[3])
    key3 = to_str_key(state[0], state[1], state[2], state[3])
    dico_1 = dic[key1] if key1 in dic else {}
    dico_2 = dic[key2] if key2 in dic else {}
    dico_3 = dic[key3] if key3 in dic else {}
    dico_final = {**dico_1, **dico_2, **dico_3}
    return dico_final


def q_learning(rseed, startingstates):
    """
    Function who running the algorithm of Q-learning
    :param rseed:
    :param startingstates: state from a random list
    :return: list countain the number of step for each episode
    """
    # Let numpy initialize the seed from the machine random entropy source
    np.random.seed(rseed)
    # Generate rules
    dict_rules = generate_rules()

    # params of the RL algo
    N = 500000
    alpha = 0.3
    gamma = 0.9

    stepsarray = np.zeros(N, dtype=np.float)
    start_time = time.time()

    # Q learning algo
    for i in range(0, N):

        if (i % 1000 < 500):
            state = startingstates[i % 100]
            epsilon = 0.2
        else:
            state = startingstates[i % 100]
            epsilon = 0.0

        captured = False
        steps = 0

        while (captured != True):

            # list_r5 is the list of the rules that are consistent with state
            dict_r5 = variable_elimination_type_2(dict_rules, state)

            if np.random.uniform() < epsilon:
                a1 = np.random.randint(0, 5)
                a2 = np.random.randint(0, 5)
                a = [a1, a2]
            else:
                a = bestaction(dict_r5)

            # observe result from action a
            (state2, reward, captured) = mapping(state, a)

            if i % 1000 < 500:
                # update of the rules

                # list_r2 is the list of the rules that are consistent with (state,a)
                dict_r2 = variable_elimination_type_1(dict_r5, a[0], a[1])

                # list_r6 is the list of the rules that are consistent with state2
                dict_r6 = variable_elimination_type_2(dict_rules, state2)

                a2 = bestaction(dict_r6)

                for rule in dict_r2:
                    if rule != None:
                        if rule.Pred1 == (None, None):
                            rule.Value += \
                                alpha * \
                                (reward[1] + gamma * Q2(state2, a2,
                                                        dict_r6) - Q2(state, a, dict_r5))

                        elif rule.Pred2 == (None, None):
                            rule.Value += \
                                alpha * \
                                (reward[0] + gamma * Q1(state2, a2,
                                                        dict_r6) - Q1(state, a, dict_r5))

                        else:
                            rule.Value += \
                                alpha * \
                                ((reward[1] + gamma * Q2(state2, a2, dict_r6) - Q2(state, a, dict_r5)) + (
                                        reward[0] + gamma * Q1(state2, a2, dict_r6) - Q1(state, a, dict_r5)))

            state = state2

            steps += 1

        stepsarray[i] = steps

        if (i % 100 == 0):
            print("Episode", i + 1, "finished in", steps,
                  "steps and", time.time() - start_time, "seconds.")

    print("The Q learning algo took --- %s seconds ---" %
          (time.time() - start_time))

    return stepsarray


def main():
    """
    Function main
    :return: /
    """
    np.random.seed(2161)
    seeds = [np.random.randint(100, 10000) for _ in range(0, 100)]
    # generating the 100 starting states
    startingstates = np.zeros((100, 4), dtype=np.int)
    for i in range(0, 100):
        for j in range(0, 4):
            startingstates[i][j] = int(np.random.choice(
                range(0, 10), size=1, replace=True))
    trials = 10
    episodes = 500000
    results = []
    pool = mp.Pool(processes=10)
    for i in range(trials):
        results.append(pool.apply_async(q_learning, (seeds[i], startingstates,)))
    pool.close()
    pool.join()
    # q_learning(8642)
    count_result = 0
    episodes_step_arr = np.zeros((trials, episodes))
    for tr in range(trials):
        episodes_step_arr[tr] = results[count_result].get()
        count_result += 1

    avg_steps = []
    for ts in range(episodes):
        avg_steps.append(np.average(episodes_step_arr[:, ts]))

    import csv
    filename = "sparse.csv"
    with open(filename, mode='w+') as outfile:
        owriter = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, episodes):
            owriter.writerow([avg_steps[i]])

    # charts
    style = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-',
             'k-', 'b--', 'g--', 'r--', 'c--', ]
    plt.plot(range(0, episodes), avg_steps, style[0])
    M1 = np.amin(avg_steps)
    M2 = np.amax(avg_steps)
    plt.xlabel('Time')
    plt.ylabel('Steps')
    plt.title('SCQ - Number of steps before capture per episode over time')
    plt.axis([0, episodes, M1, M2])
    plt.savefig("sparse_max_50x10_5.png")
    plt.axis([0, episodes, M1, 100])
    plt.savefig("sparse_100_50x10_5.png")
    plt.axis([0, 10000, M1, 100])
    plt.savefig("sparse_100_10000.png")
    plt.axis([0, 100000, M1, 100])
    plt.savefig("sparse_100_10x10_5.png")
    plt.axis([0, 200000, M1, 100])
    plt.savefig("sparse_100_20x10_5.png")


if __name__ == "__main__":
    main()
