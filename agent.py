import torch
import random
import numpy as np
from collections import deque
from game import Snake, Direction, point, BLOCK_SIZE
from model import QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n = 0
        self.ep = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_r = point(head.x + BLOCK_SIZE, head.y)
        point_d = point(head.x, head.y + BLOCK_SIZE)
        point_l = point(head.x - BLOCK_SIZE, head.y)
        point_u = point(head.x, head.y - BLOCK_SIZE)

        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP

        state = [
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_mem(self):
        if len(self.memory) > BATCH_SIZE:
            small = random.sample(self.memory, BATCH_SIZE)
        else:
            small = self.memory

        states, actions, rewards, next_states, dones = zip(*small)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_s_mem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.ep = 80 - self.n
        final_move = [0,0,0]
        if random.randint(0,200)<self.ep:
            move = random.randint(0,2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train():
    total_score = 0
    agent = Agent()
    game = Snake()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.step(final_move)
        state_new = agent.get_state(game)

        agent.train_s_mem(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n += 1
            agent.train_mem()
            if score>reward:
                reward = score
                agent.model.save()
            print(f'Game: {agent.n}, Score: {score}')
            total_score+=score

train()