import getopt
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings
from collections import *

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
possibleactions2 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 2), (-1, 1), (1, 1),
                    (-2, 0), (2, 0), (-1, -1), (-1, 1), (0, -2)]


def manhattandistance(a, b):
    return min(a, 10 - a) + min(b, 10 - b)


def mapping(state, action):
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
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = [-50, -50]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(range(0, 10), size=1, replace=True))

    # penalty for one predators jumping on the prey while being alone
    if (manhattandistance(state[0], state[1]) == 1) \
            and (manhattandistance(state[2], state[3]) > 1) \
            and (manhattandistance(statefinal[0], statefinal[1]) == 0):
        reward = [-5, -5]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(range(0, 10), size=1, replace=True))

    if (manhattandistance(state[0], state[1]) > 1) \
            and (manhattandistance(state[2], state[3]) == 1) \
            and (manhattandistance(statefinal[2], statefinal[3]) == 0):
        reward = [-5, -5]
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice(range(0, 10), size=1, replace=True))

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
    """
    key = str(pred1) + str(pred2) + str(action1) + str(action2)
    return key


def generate_rules():
    """
    """
    dict_rules = dict()
    for i in range(0, 10):
        for j in range(0, 10):
            if ((i != 0) or (j != 0)):
                for k in range(0, 5):
                    for l in range(0, 5):
                        for m in range(0, 13):
                            x = (i + possibleactions2[m][0]) % 10
                            y = (j + possibleactions2[m][1]) % 10
                            if ((x != 0) or (y != 0)):
                                r = Rule((i, j), (x, y), k, l, 75, 2)
                                key = to_str_key(i, j, x, y)
                                dict_rules.setdefault(key, {}).update(
                                    {to_str_key(k, l): r})

    for m in range(1, 13):
        for n in range(1, 13):
            for k in range(0, 5):
                for l in range(0, 5):
                    x = (possibleactions2[m][0]) % 10
                    y = (possibleactions2[m][1]) % 10
                    z = (possibleactions2[n][0]) % 10
                    t = (possibleactions2[n][1]) % 10
                    r = Rule((x, y), (z, t), k, l, 75, 2)
                    key = to_str_key(x, y, z, t)
                    dict_rules.setdefault(key, {}).update({to_str_key(k, l): r})

    for i in range(0, 10):
        for j in range(0, 10):
            if ((i != 0) or (j != 0)):
                for k in range(0, 5):
                    r1 = Rule((i, j), (None, None), k, None, 75, 1)
                    r2 = Rule((None, None), (i, j), None, k, 75, 1)
                    key1 = to_str_key(i, j, None, None)
                    key2 = to_str_key(None, None, i, j)
                    dict_rules.setdefault(key1, {}).update(
                        {to_str_key(k, None): r1})
                    dict_rules.setdefault(key2, {}).update(
                        {to_str_key(None, k): r2})

    return dict_rules


# definition of the functions Q1 and Q2
# list_r is the list of the rules that are consistent with (state,action)
# and where predator 1 (resp. 2) is involved

def Q1(state, action, dic):
    S = 0
    list_r = [v for k, v in dic.items() \
              if v.Pred1 == (state[0], state[1])
              and v.Action1 == action[0]
              and (v.Action2 == action[1] or v.Action2 == None)]
    for rule in list_r:
        S += rule.Value / rule.nj
    return S


def Q2(state, action, dic):
    S = 0
    list_r = [v for k, v in dic.items() \
              if v.Pred2 == (state[2], state[3])
              and (v.Action1 == action[0] or v.Action1 == None)
              and v.Action2 == action[1]]
    for rule in list_r:
        S += rule.Value / rule.nj
    return S


# best combined action a that maximizes global payoff in a given state
def bestaction(state, dic):
    act = [0, 0]
    # for each action of predator 1 find the best action of predator 2
    tab = [[0, 0, 0.0], [1, 0, 0.0], [2, 0, 0.0], [3, 0, 0.0], [4, 0, 0.0]]
    ro = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(0, 5):
        poM = 0
        for j in range(0, 5):
            po = 0

            rules = []
            key1 = to_str_key(i, j)
            key2 = to_str_key(i, None)
            key3 = to_str_key(None, j)
            dict_r4_1 = dic[key1] if key1 in dic else None
            dict_r4_2 = dic[key2] if key2 in dic else None
            dict_r4_3 = dic[key3] if key3 in dic else None

            rules.extend((dict_r4_1, dict_r4_2, dict_r4_3))

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
    max_indexes = np.where(ro == ro[m])[0]
    if len(max_indexes) == 1:
        action = max_indexes[0]
        act = [tab[action][0], tab[action][1]]
    else:
        action = np.random.choice(max_indexes)
        act = [tab[action][0], tab[action][1]]

    return act


def q_learning(rseed):
    # Let numpy initialize the seed from the machine random entropy source
    np.random.seed(rseed)
    # Retrieve the initial state to be able to reproduce the results
    st0 = np.random.get_state()
    # Generate rules
    dict_rules = generate_rules()
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

    # Q learning algo
    for i in range(0, N):

        if (i % 1000 < 500):
            state = startingstates[i % 100]
            epsilon = 0.2
        else:
            # for j in range(0, 4):
            #    state[j] = int(np.random.choice(range(0,10), size=1, replace=True))
            state = startingstates[i % 100]
            epsilon = 0

        captured = False
        steps = 0

        while (captured != True):

            # list_r5 is the list of the rules that are consistent with state
            key1 = to_str_key(state[0], state[1], None, None)
            key2 = to_str_key(None, None, state[2], state[3])
            key3 = to_str_key(state[0], state[1], state[2], state[3])
            dict_r5_1 = dict_rules[key1] if key1 in dict_rules else {}
            dict_r5_2 = dict_rules[key2] if key2 in dict_rules else {}
            dict_r5_3 = dict_rules[key3] if key3 in dict_rules else {}

            dict_r5 = {**dict_r5_1, **dict_r5_2, **dict_r5_3}

            # choose combined action according to epsilon greedy policy
            if (np.random.uniform() < epsilon):
                a1 = int(np.random.choice(range(0, 5), size=1, replace=False))
                a2 = int(np.random.choice(range(0, 5), size=1, replace=False))
                a = [a1, a2]
            else:
                a = bestaction(state, dict_r5)

            # observe result from action a
            (state2, reward, captured) = mapping(state, a)

            # update of the rules

            # list_r2 is the list of the rules that are consistent with (state,a)
            dict_r2 = []
            key1 = to_str_key(a[0], None)
            key2 = to_str_key(None, a[1])
            key3 = to_str_key(a[0], a[1])
            rule1 = dict_r5[key1] if key1 in dict_r5 else None
            rule2 = dict_r5[key2] if key2 in dict_r5 else None
            rule3 = dict_r5[key3] if key3 in dict_r5 else None
            dict_r2.extend((rule1, rule2, rule3))

            # list_r6 is the list of the rules that are consistent with state2
            key1 = to_str_key(state2[0], state2[1], None, None)
            key2 = to_str_key(None, None, state2[2], state2[3])
            key3 = to_str_key(state2[0], state2[1], state2[2], state2[3])
            dict_r1_6 = dict_rules[key1] if key1 in dict_rules else {}
            dict_r2_6 = dict_rules[key2] if key2 in dict_rules else {}
            dict_r3_6 = dict_rules[key3] if key3 in dict_rules else {}

            dict_r6 = {**dict_r1_6, **dict_r2_6, **dict_r3_6}

            a2 = bestaction(state2, dict_r6)

            for rule in dict_r2:
                if rule != None:
                    if rule.Pred1 == (None, None):
                        rule.Value += \
                            alpha * \
                            (reward[1] + gamma * Q2(state2, a2, dict_r6) - Q2(state, a, dict_r5))
                    elif rule.Pred2 == (None, None):
                        rule.Value += \
                            alpha * \
                            (reward[0] + gamma * Q1(state2, a2, dict_r6) - Q1(state, a, dict_r5))
                    else:
                        rule.Value += \
                            alpha * \
                            ((reward[1] + gamma * Q2(state2, a2, dict_r6) - Q2(state, a, dict_r5)) + (
                                        reward[0] + gamma * Q1(state2, a2, dict_r6) - Q1(state, a, dict_r5)))

            state = state2

            steps += 1

        stepsarray[i] = steps

        if (i % 10000 == 0):
            print("Episode", i + 1, "finished in", steps, "steps and", time.time() - start_time, "seconds.")

        i += 1

    print("The Q learning algo took --- %s seconds ---" %
          (time.time() - start_time))

    return stepsarray


def main():
    import multiprocessing as mp
    seeds = [8642, 1489, 9952, 8995, 1962, 8483, 2021, 2161, 9628, 7462]
    trials = 10
    episodes = 500000
    results = []
    pool = mp.Pool(processes=4)
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
    plt.plot(range(0, episodes), avg_steps, style[0])
    M1 = np.amin(avg_steps)
    M2 = np.amax(avg_steps)
    plt.xlabel('Time')
    plt.ylabel('Steps')
    plt.title('Sparse Cooperative Q-learning - Number of steps before capture per episode over time')
    plt.axis([0, episodes, M1, M2])
    plt.show()


if __name__ == "__main__":
    main()