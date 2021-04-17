import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import sys
import time
import random
from maze import map_selecter, target_selecter

np.random.seed(333)
pd.set_option('display.max_rows', None)

MAP_ID = 4

map = map_selecter[MAP_ID]


WHO_IS_RUNNING = ''
N_HEIGHT = len(map)
N_WIDTH = len(map[0])
N_STATES = N_HEIGHT * N_WIDTH
ACTIONS = ['up', 'down', 'left', 'right']
EPSILON = 0.01
GAMMA = 0.9
ALPHA = 0.3
MAX_EPISODES = 50
FRESH_TIME = 0
START_POS = (1, 1)
TARGET_POS = target_selecter[MAP_ID]


WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
WALL_WIDTH = 550 // N_WIDTH
WALL_HEIGHT = 550 // N_HEIGHT
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
map[TARGET_POS[0]][TARGET_POS[1]] = 2
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
screen.fill(WHITE)




def state_to_pos(state):
    x = int(state / N_WIDTH)
    y = int(state % N_WIDTH)
    pos = (x, y)
    return pos

def pos_to_state(pos):
    state = pos[0] * N_WIDTH + pos[1]
    return state

def build_q_table():
    table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )
    return table

def fux(state_actions):
    return state_actions.reindex(np.random.permutation(state_actions.index))

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() < EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = fux(state_actions).idxmax()
    return action_name

def get_feedback(state, action):
    pos = state_to_pos(state)
    x_ = int(pos[0])
    y_ = int(pos[1])
    if action == 'up':
        x_ -= 1
    elif action == 'down':
        x_ += 1
    elif action == 'left':
        y_ -= 1
    elif action == 'right':
        y_ += 1
    if map[x_][y_] == 0:
        state_ = state
        reward = -2
    elif map[x_][y_] == 2:
        state_ = 'arrived'
        reward = 2000
    elif map[x_][y_] == 1:
        state_ = pos_to_state((x_, y_))
        reward = -1
    return state_, reward

def update_env(state, episode, cnt_step):
    if state == 'dead':
        interaction = '[%s:dead]Episode %s: total_steps = %s' % (WHO_IS_RUNNING, episode + 1, cnt_step)
        print(interaction)
    elif state == 'arrived':
        interaction = '[%s]Episode %s: total_steps = %s' % (WHO_IS_RUNNING, episode + 1, cnt_step)
        print(interaction)
    x_start = (WINDOW_HEIGHT - N_HEIGHT * WALL_HEIGHT) / 2
    y_start = (WINDOW_WIDTH - N_WIDTH * WALL_WIDTH) / 2
    for i in range(N_HEIGHT):
        for j in range(N_WIDTH):
            x_pos = x_start + WALL_HEIGHT * i
            y_pos = y_start + WALL_WIDTH * j
            if i * N_HEIGHT + j == state:
                pygame.draw.rect(screen, GREEN, (y_pos, x_pos, WALL_WIDTH, WALL_HEIGHT), 0)
            elif i * N_HEIGHT + j == pos_to_state(START_POS):
                pygame.draw.rect(screen, BLUE, (y_pos, x_pos, WALL_WIDTH, WALL_HEIGHT), 0)
            elif map[i][j] == 0:
                pygame.draw.rect(screen, BLACK, (y_pos, x_pos, WALL_WIDTH, WALL_HEIGHT), 0)
            elif map[i][j] == 1:
                pygame.draw.rect(screen, WHITE, (y_pos, x_pos, WALL_WIDTH, WALL_HEIGHT), 0)
            elif map[i][j] == 2:
                pygame.draw.rect(screen, RED, (y_pos, x_pos, WALL_WIDTH, WALL_HEIGHT), 0)
    pygame.display.flip()
    time.sleep(FRESH_TIME)

def Sarsa():
    global WHO_IS_RUNNING
    WHO_IS_RUNNING = 'Sarsa'
    q_table = build_q_table()
    data_set = []
    for episode in range(MAX_EPISODES):
        cnt_step = 0
        state = pos_to_state(START_POS)
        is_terminated = False
        update_env(state, episode, cnt_step)
        action = choose_action(state, q_table)
        while not is_terminated:
            state_, reward = get_feedback(state, action)
            action_ = ''
            q_predict = q_table.loc[state, action]
            if state_ != 'dead' and state_ != 'arrived':
                action_ = choose_action(state_, q_table)
                q_target = reward + GAMMA * q_table.loc[state_, action_]
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = state_
            action = action_

            update_env(state, episode, cnt_step + 1)
            cnt_step += 1
            pygame.event.pump()
        data_set.append(cnt_step)
    return data_set, q_table

def Q_learning():
    global WHO_IS_RUNNING
    WHO_IS_RUNNING = 'Q learning'
    q_table = build_q_table()
    data_set = []
    for episode in range(MAX_EPISODES):
        cnt_step = 0
        state = pos_to_state(START_POS)
        is_terminated = False
        update_env(state, episode, cnt_step)
        while not is_terminated:
            action = choose_action(state, q_table)
            state_, reward = get_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if state_ != 'dead' and state_ != 'arrived':
                q_target = reward + GAMMA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = state_

            update_env(state, episode, cnt_step + 1)
            cnt_step += 1
            pygame.event.pump()
        data_set.append(cnt_step)
    return data_set, q_table

def Double_Q_learning():
    global WHO_IS_RUNNING
    WHO_IS_RUNNING = 'Double Q learning'
    q_table1 = build_q_table()
    q_table2 = build_q_table()
    data_set = []
    for episode in range(MAX_EPISODES):
        cnt_step = 0
        state = pos_to_state(START_POS)
        is_terminated = False
        update_env(state, episode, cnt_step)
        while not is_terminated:
            action = choose_action(state, q_table1 + q_table2)
            state_, reward = get_feedback(state, action)

            if np.random.uniform() < 0.5:
                q_predict = q_table1.loc[state, action]
                if state_ != 'dead' and state_ != 'arrived':
                    q_target = reward + GAMMA * q_table2.loc[state_, fux(q_table1.iloc[state_, :]).idxmax()]
                else:
                    q_target = reward
                    is_terminated = True

                q_table1.loc[state, action] += ALPHA * (q_target - q_predict)
            else:
                q_predict = q_table2.loc[state, action]
                if state_ != 'dead' and state_ != 'arrived':
                    q_target = reward + GAMMA * q_table1.loc[state_, fux(q_table2.iloc[state_, :]).idxmax()]
                else:
                    q_target = reward
                    is_terminated = True

                q_table2.loc[state, action] += ALPHA * (q_target - q_predict)

            state = state_

            update_env(state, episode, cnt_step + 1)
            cnt_step += 1
            pygame.event.pump()
        data_set.append(cnt_step)
    return data_set, q_table1 + q_table2

def Dyna_Q(n, dynamic=False):
    global WHO_IS_RUNNING
    WHO_IS_RUNNING = 'Dyna Q(%s)' % n
    MAX_MEMORY_SIZE = 1000
    q_table = build_q_table()
    model_r = build_q_table()
    model_s = build_q_table()
    experience = []
    data_set = []
    for episode in range(MAX_EPISODES):
        cnt_step = 0
        state = pos_to_state(START_POS)
        is_terminated = False
        update_env(state, episode, cnt_step)
        while not is_terminated:
            action = choose_action(state, q_table)
            state_, reward = get_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if state_ != 'dead' and state_ != 'arrived':
                q_target = reward + GAMMA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)

            model_r.loc[state, action] = reward
            model_s.loc[state, action] = state_
            if len(experience) >= MAX_MEMORY_SIZE:
                experience.pop(0)
            experience.append((state, action))
            for ex_state, ex_action in random.choices(experience, k=n):
                ex_reward = model_r.loc[ex_state, ex_action]
                ex_state_ = model_s.loc[ex_state, ex_action]
                ex_q_predict = q_table.loc[ex_state, ex_action]
                if ex_state_ != 'dead' and ex_state_ != 'arrived':
                    ex_state_ = int(ex_state_)
                    ex_q_target = ex_reward + GAMMA * q_table.iloc[ex_state_, :].max()
                else:
                    ex_q_target = ex_reward

                q_table.loc[ex_state, ex_action] += ALPHA * (ex_q_target - ex_q_predict)

            state = state_

            update_env(state, episode, cnt_step + 1)
            cnt_step += 1
            pygame.event.pump()
        data_set.append(cnt_step)
        if episode > 0:
            ln.remove()

        ln, = plt.plot(range(len(data_set)), data_set, color='b', lw=1)


        if dynamic:
            if len(data_set) == 3:
                _var = int(np.var(data_set))
            elif len(data_set) > 3:
                var = int(np.var(data_set))
                if var <= _var and n > 5:
                    n = n // 2 + 2
                elif var > _var:
                    n = n + 2
                WHO_IS_RUNNING = 'Dyna Q(%s)' % n

    plt.clf()
    plt_init()
    return data_set, q_table


def run_dyna_Q(n, legend, datasets, dynamic=False):
    data_set, q_table = Dyna_Q(n, dynamic=dynamic)
    datasets.append(data_set)
    legend.append('Q(%s)' % ('dynamic' if dynamic else n))

def plt_init():
    plt.rcParams['font.sans-serif'] = 'simhei'
    plt.xlim(-2, MAX_EPISODES + 2)
    plt.ylim(-20, 1000)

if __name__ == '__main__':

    plt.ion()
    plt_init()

    legend = []
    datasets = []
    dyna_test_pool = [5, 10, 20]

    for i, n in enumerate(dyna_test_pool):
        run_dyna_Q(n, legend, datasets)

    # run_dyna_Q(20, legend, color=4, dynamic=True)


    # data_set, q_table = Sarsa()
    # plt.plot([x for x in range(1, MAX_EPISODES + 1)], data_set, 'r', lw=1)
    # legend.append('Sarsa')

    # data_set, q_table = Q_learning()
    # plt.plot([x for x in range(1, MAX_EPISODES + 1)], data_set, 'b', lw=1)
    # legend.append('Q')

    # data_set, q_table = Double_Q_learning()
    # plt.plot([x for x in range(1, MAX_EPISODES + 1)], data_set, 'g', lw=1)
    # legend.append('Double Q')

    # data_set, q_table = Dyna_Q(0)
    # plt.plot([x for x in range(1, MAX_EPISODES + 1)], data_set, 'b', lw=1)
    # legend.append('Q(0)')

    plt.ioff()
    plt.close()

    for i in range(len(datasets)):
        plt.plot([x for x in range(1, MAX_EPISODES + 1)], datasets[i], lw=1)
    plt.legend(legend)
    plt.show()

    pygame.quit()
    sys.exit()
