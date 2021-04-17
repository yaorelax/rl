import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


np.random.seed(1024)
pd.set_option('display.max_rows', None)

NPY_RESTORE_FILENAME = '21.npy'

N_STATES = 21
ACTIONS = ['hit', 'stand']
EPSILON = 0.001
GAMMA = 0.9
ALPHA = 0.2
MAX_EPISODES = 10000
MIRROR_UPDATE_INTERVAL = 1000
POKE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

global q_table, q_table_mirror

def build_q_table():
    global q_table, q_table_backup

    if os.path.isfile(NPY_RESTORE_FILENAME):
        q_table = q_table_backup = pd.DataFrame(
            np.load(NPY_RESTORE_FILENAME),
            columns=ACTIONS
        )
        print('Model Found:', NPY_RESTORE_FILENAME)
        print(q_table)
    else:
        q_table = q_table_backup = pd.DataFrame(
            np.zeros((N_STATES, len(ACTIONS))),
            columns=ACTIONS
        )

def choose_action(state):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() < EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        # state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
        action_name = state_actions.idxmax()
    return action_name

def get_feedback(state, action):
    is_terminated = True
    if action == 'stand':
        state_ = 'stand'
    elif action == 'hit':
        hit = np.random.choice(POKE)
        hited = state + hit
        if hited == 21:
            state_ = '21point'
        elif hited > 21:
            state_ = 'bust'
        elif hited < 21:
            state_ = hited
            reward = 0
            is_terminated = False
    if is_terminated:
        mirror_state_, mirror_state = mirror_player()
        winer_table = {'stand':   {'stand': -99, 'bust':  3, '21point': -2},
                       'bust':    {'stand':  -2, 'bust': -1, '21point': -2},
                       '21point': {'stand':   3, 'bust':  3, '21point': -1}}
        if winer_table[state_][mirror_state_] == -99:
            if state > mirror_state:
                reward = 3
            elif state == mirror_state:
                reward = -1
            elif state < mirror_state:
                reward = -2
        else:
            reward = winer_table[state_][mirror_state_]

    return state_, reward

def mirror_player():
    state = 0
    is_terminated = False
    while not is_terminated:
        state_actions = q_table_mirror.iloc[state, :]
        if (np.random.uniform() < EPSILON) or ((state_actions == 0).all()):
            action = np.random.choice(ACTIONS)
        else:
            # state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action = state_actions.idxmax()

        if action == 'stand':
            state_ = 'stand'
            is_terminated = True
        elif action == 'hit':
            hit = np.random.choice(POKE)
            hited = state + hit
            if hited == 21:
                state_ = '21point'
                is_terminated = True
            elif hited > 21:
                state_ = 'bust'
                is_terminated = True
            elif hited < 21:
                state_ = hited

        if is_terminated:
            break

        state = state_

    return state_, state



def run():
    global q_table, q_table_mirror
    build_q_table()
    win_rate_set = []
    loss_rate_set = []
    plain_rate_set = []
    stand_set = []
    cnt_win = 0
    cnt_loss = 0
    cnt_plain = 0
    for episode in range(MAX_EPISODES):
        if episode % 1000 == 0:
            print('Episode %s' % episode)
            q_table_mirror = q_table
        state = 0
        is_terminated = False
        while not is_terminated:
            action = choose_action(state)
            state_, reward = get_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if state_ != 'bust' and state_ != 'stand' and state_ != '21point':
                q_target = reward + GAMMA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)

            if is_terminated:
                '''
                if state_ == 'bust':
                    interaction = '[bust]Episode %s: reward = %s' % (episode + 1, reward)
                    print(interaction)
                elif state_ == 'stand':
                    interaction = '[stand:%s]Episode %s: reward = %s' % (state, episode + 1, reward)
                    print(interaction)
                elif state_ == '21point':
                    interaction = '[stand:%s]Episode %s: reward = %s' % (21, episode + 1, reward)
                    print(interaction)
                '''
                if reward == 3:
                    cnt_win += 1
                elif reward == -2:
                    cnt_loss += 1
                elif reward == -1:
                    cnt_plain += 1



            state = state_

        for i in range(N_STATES):
            if q_table.loc[20 - i, 'hit'] >= q_table.loc[20 - i, 'stand']:
                stand_set.append(20 - i + 1)
                break

        win_rate_set.append(cnt_win / (episode + 1))
        loss_rate_set.append(cnt_loss / (episode + 1))
        plain_rate_set.append(cnt_plain / (episode + 1))
    return win_rate_set, loss_rate_set, plain_rate_set, stand_set

if __name__ == '__main__':
    win_rate_set, loss_rate_set, plain_rate_set, stand_set = run()

    plt.subplot(121)
    plt.plot([x for x in range(1, MAX_EPISODES + 1)], win_rate_set, 'r', lw=1)
    plt.plot([x for x in range(1, MAX_EPISODES + 1)], loss_rate_set, 'g', lw=1)
    plt.plot([x for x in range(1, MAX_EPISODES + 1)], plain_rate_set, 'b', lw=1)
    plt.legend(['win_rate', 'loss_rate', 'plain_rate'])

    plt.subplot(122)
    plt.plot([x for x in range(1, MAX_EPISODES + 1)], stand_set, 'r', lw=1)
    plt.legend(['stand'])

    plt.show()

    strategy = pd.DataFrame(
        np.zeros((N_STATES, 1)),
        columns=['strategy']
    )
    for i in range(N_STATES):
        if q_table.loc[i, 'hit'] >= q_table.loc[i, 'stand']:
            strategy.loc[i, 'strategy'] = 'hit'
        else:
            strategy.loc[i, 'strategy'] = 'stand'


    print('\r\nQ-table:\n')
    print(q_table)

    print('\r\nStrategy:\n')
    print(strategy)

    np.save(NPY_RESTORE_FILENAME, q_table)


