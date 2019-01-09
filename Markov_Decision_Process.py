import os
import sys, getopt
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp

# state = array [x,y,z,t] with x,y,z,t in {0, ... ,9}
# predator 1 location - prey location = [x,y]
# predator 2 location - prey location = [z,t]
# action = tuple (a,b) with a,b in {0, ... ,4}
# with action 0 = (0,0), action 1 = (1,0), action 2 = (-1,0), action 3 = (0,1), action 4 = (0,-1)
# mapping that simulates the experiment
# input = state + action
# output = new state + reward + boolean = true if capture (has to ne in coordination)

possibleactions = [(0,0) , (1,0) , (-1,0) , (0,1) , (0,-1)]

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
    statefinal = [0,0,0,0]
    reward = -1

    #checking the state has all its values in {0, ... ,9}
    if (state[0] < 0) or (state[0] > 9) or (state[1] < 0) or (state[1] > 9)\
    or (state[2] < 0) or (state[2] > 9) or (state[3] < 0) or (state[3] > 9):
        print("Relative positions out of bound")

    #moving the predators on the torus according to action
    statefinal[0] = ( state[0] + possibleactions[action[0]][0]) % 10
    statefinal[1] = ( state[1] + possibleactions[action[0]][1]) % 10
    statefinal[2] = ( state[2] + possibleactions[action[1]][0]) % 10
    statefinal[3] = ( state[3] + possibleactions[action[1]][1]) % 10

    #reward for catching the prey in coordination
    if  ( manhattandistance (state[0],state[1]) == 1 ) \
    and ( manhattandistance (state[2],state[3]) == 1 ) \
    and ( manhattandistance (statefinal[0],statefinal[1] ) == 0 ) \
    and ( manhattandistance (statefinal[2],statefinal[3] ) != 0 ):
        reward = 75
        captured = True

    if  ( manhattandistance ( state[0],state[1] ) == 1 ) \
    and ( manhattandistance ( state[2],state[3] ) == 1 ) \
    and ( manhattandistance ( statefinal[0],statefinal[1] ) != 0 ) \
    and ( manhattandistance ( statefinal[2],statefinal[3] ) == 0 ):
        reward = 75
        captured = True

    #penalty for both predators jumping on the prey
    # penalty for both predators jumping on the same cell
    if (statefinal[0]==statefinal[2]) and (statefinal[1]==statefinal[3]):
        reward = -50
        for i in range(0, 4):
            statefinal[i] = int(np.random.choice( range(0,10), size=1,replace=True))
    
    #penalty for one predators jumping on the prey while being alone
    if  ( manhattandistance ( state[0],state[1] ) == 1) \
    and ( manhattandistance ( state[2],state[3] ) > 1 ) \
    and ( manhattandistance ( statefinal[0],statefinal[1] ) == 0 ):
        reward = -5
        for i in range(0,4):
            statefinal[i] = int ( np.random.choice( range(0,10) , size=1 , \
            replace=True , p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] ) )

    if  ( manhattandistance ( state[0],state[1] ) > 1 ) \
    and ( manhattandistance ( state[2],state[3] ) == 1 ) \
    and ( manhattandistance ( statefinal[2],statefinal[3] ) == 0 ):
        reward = -5
        for i in range(0,4):
            statefinal[i] = int ( np.random.choice( range(0,10) , size=1 , \
            replace=True , p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] ) )

    #moving the prey randomly
    r = int (
         np.random.choice( range(0,5) , size=1 , replace=True ,
         p=[0.2,0.2,0.2,0.2,0.2] ) )
    statefinal[0] = ( statefinal[0] - possibleactions[r][0] ) % 10
    statefinal[1] = ( statefinal[1] - possibleactions[r][1] ) % 10
    statefinal[2] = ( statefinal[2] - possibleactions[r][0] ) % 10
    statefinal[3] = ( statefinal[3] - possibleactions[r][1] ) % 10

    return ( statefinal , reward , captured )

    #best action and Q value derived from Q in a given state
def bestaction(Q, state):
    """
    Function to found the best action
    :param dic:
    :return: best action combined
    """
    M=Q[ state[0] ][ state[1] ][ state[2] ][ state[3] ][0][0]
    act = (0 , 0)
    for k in range(0,5):
        for l in range(0,5):
            if ( Q[ state[0] ][ state[1] ][ state[2] ][ state[3] ][k][l] > M ):
                M = Q[ state[0] ][ state[1] ][ state[2] ][ state[3] ][k][l]
                act = (k,l)
    return ( act , M )

def q_learning(rseed, startingstates):
    """
    Function who running the algorithm of Q-learning
    :param rseed:
    :param startingstates: state from a random list
    :return: list countain the number of step for each episode
    """
    # Let numpy initialize the seed from the machine random entropy source
    np.random.seed(rseed)

    # params of the RL algo
    N = 100000
    alpha = 0.3
    gamma = 0.9

    stepsarray = np.zeros(N, dtype=np.float)
    start_time = time.time()

    # initialisation of the Q values
    # Q = numpy array where multi index = ( state[0], ... , state[3], action predator 1, action predator 2 )
    Q = np.zeros( (10,10,10,10,5,5) ,dtype=np.float)
    for agent1_x in range(0,10):
        for agent1_y in range(0,10):
            for agent2_x in range(0,10):
                for agent2_y in range(0,10):
                    for action_1 in range(0,5):
                        for action_2 in range(0,5):
                            Q[agent1_x][agent1_y][agent2_x][agent2_y][action_1][action_2]=75


    #Q learning algo
    for i in range(0,N):
        if (i % 1000 < 500):
            state = startingstates[i % 100]
            epsilon = 0.2
        else:
            state = startingstates[i % 100]
            epsilon = 0

        captured = False
        steps = 0

        while( captured != True ):

            #choose action according to epsilon greedy policy
            if ( np.random.uniform() < epsilon ):
                a = ( int ( np.random.choice(range(0,5),size=1,replace=False,p=[0.2,0.2,0.2,0.2,0.2]) ) , \
                int ( np.random.choice(range(0,5),size=1,replace=False,p=[0.2,0.2,0.2,0.2,0.2]) ) )    
            else:
                a = bestaction(Q, state)[0]
                
            #observe result from action a
            (state2,reward,captured) = mapping(state,a)
            
            #update Q
            Q[state[0]][state[1]][state[2]][state[3]][a[0]][a[1]] = \
            (1-alpha)*Q[state[0]][state[1]][state[2]][state[3]][a[0]][a[1]] + \
            alpha*reward + alpha*gamma*bestaction(Q, state2)[1]

            state = state2

            steps+=1

        stepsarray[i]=steps
        
        if(i%1000 == 0):
            print("Episode",i+1,"finished in",steps,"steps.")

    print("The Q learning algo took --- %s seconds ---" % (time.time() - start_time))

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
            startingstates[i][j] = int(np.random.choice(range(0,10), size=1, replace=True))
    trials = 1
    episodes = 100000
    results = []
    pool = mp.Pool(processes=1)
    for i in range(trials):
        results.append(pool.apply_async(q_learning,(seeds[i], startingstates,)))
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

    times = int(episodes/1000)
    for i in range(0, times):
        avg = np.average(avg_steps[i*1000:i*1000+1000])
        print("avg from %d to %d : %f " % (i*1000,i*1000+1000,avg))
    
    print("avg from %d to %d : %f " % (99500,100000,np.average(avg_steps[99500:100000])))


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
