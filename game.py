import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('Arial', 20)

class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4

point = namedtuple('Point', 'x, y')

WHITE = (255,255,255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80

class Snake():
    def __init__(self):
        self.display = pygame.display.set_mode((480, 480))
        pygame.display.set_caption("Snake Reinforcement Learning")
        self.time = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = point(240, 240)
        self.snake = [self.head, point(self.head.x-BLOCK_SIZE, self.head.y), point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.add_food()
        self.iterate = 0

    def add_food(self):
        x = random.randint(0, (480-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (480-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = point(x, y)
        if self.food in self.snake:
            self.add_food()

    def step(self, action):
        self.iterate += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        over = False
        if self.is_collision() or self.iterate > 100*len(self.snake):
            over = True
            reward -= 10
            return reward, over, self.score
        if self.head == self.food:
            self.score += 1
            reward += 10
            self.add_food()
        else:
            self.snake.pop()
        self.ui()
        self.time.tick(SPEED)
        return reward, over, self.score

    def is_collision(self, a_point = None):
        if a_point is None:
            a_point = self.head
        if a_point.x>480 - BLOCK_SIZE or a_point.x < 0 or a_point.y>480 - BLOCK_SIZE or a_point.y < 0:
            return True
        if a_point in self.snake[1:]:
            return True

        return False

    def ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,450])
        pygame.display.flip()

    def move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = point(x, y)