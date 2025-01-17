import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from ResNet import ResNetAgent
from random import randint
import random
import time
import statistics
import torch.optim as optim
import torch 
from GPyOpt.methods import BayesianOptimization
from bayesOpt import *
import datetime
import distutils.util
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt
import cv2 as cv
import gif

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.0013629    #/10
    params['first_layer_size'] = 128    # neurons in the first layer
    params['second_layer_size'] = 256   # neurons in the second layer
    params['third_layer_size'] = 64    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    params['gif_path'] = 'logs/gifs_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + '/'
    if not os.path.exists(params['gif_path']+'train'):
        os.makedirs(params['gif_path']+'train')
    if not os.path.exists(params['gif_path']+'test'):
        os.makedirs(params['gif_path']+'test')
    return params


class Game:
    """ Initialize PyGAME """
    
    def __init__(self, game_width, game_height, params):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self, params)
        self.food = Food(self, self.player)
        self.score = 0


class Player(object):
    def __init__(self, game, params):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.advisedPosition = []
        self.position.append((self.x, self.y))
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0
        
        self.obstacle = []
        obsIm = cv.imread(f"img/obs{params['obs']}.png", cv.IMREAD_GRAYSCALE)
        for i in range(0, game.game_width, 20):
            for j in range(0, game.game_height, 20):
                if obsIm[int(j / 20), int(i / 20)] < 128:
                    self.obstacle.append((i, j))
        
        self.obsPotential = np.zeros((int(game.game_width // 20), int(game.game_height // 20)))
        visited = np.zeros_like(self.obsPotential)
        direct = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        q1, q2 = [], []
        for pos in self.obstacle:
            q1.append([int(pos[0] // 20), int(pos[1] // 20)])
        potVal = 0
        while len(q1) > 0:
            while len(q1) > 0:
                self.obsPotential[q1[-1][0], q1[-1][1]] = potVal
                visited[q1[-1][0], q1[-1][1]] = 1
                for dir in direct:
                    tx, ty = q1[-1][0] + dir[0], q1[-1][1] + dir[1]
                    if tx < 0 or tx >= game.game_width // 20 or ty < 0 or ty >= game.game_height // 20 or visited[tx, ty] == 1:
                        continue
                    q2.append([tx, ty])
                q1.pop()
            q1, q2 = q2, q1
            potVal += 1
        x = np.array(range(game.game_width // 20))
        y = np.array(range(game.game_height // 20))
        X, Y = np.meshgrid(x, y)
        Z = self.obsPotential[X, Y]
        plt.contourf(X, Y, Z)
        plt.savefig(params['gif_path'] + "obsPotential")

        self.obsImage = pygame.image.load('img/obstacle.png')
        self.adviseImage = pygame.image.load('img/advise.png')

        self.graph = np.zeros((7, int(game.game_width // 20), int(game.game_height // 20)))
        self.graph[4] = np.ones((int(game.game_width // 20), int(game.game_height // 20)))
        '''
            0: head
            1: body before head
            2: body
            3: obstacle
            4: ground
            5: food
            6: advised position
        '''
        for pos in self.obstacle:
            self.graph[3][int(pos[0] // 20), int(pos[1] // 20)] = 1
            self.graph[4][int(pos[0] // 20), int(pos[1] // 20)] = 0
        self.graph[0][int(self.x // 20), int(self.y // 20)] = 1


    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            self.graph[0][int(self.position[-1][0] // 20), int(self.position[-1][1] // 20)] = 0
            self.graph[4][int(self.position[-1][0] // 20), int(self.position[-1][1] // 20)] = 0
            self.graph[4][int(self.position[0][0] // 20), int(self.position[0][1] // 20)] = 1
            if self.food > 1:
                if self.food == 2:
                    self.graph[1][int(self.position[0][0] // 20), int(self.position[0][1] // 20)] = 0
                for i in range(0, self.food - 1):
                    self.position[i] = (self.position[i + 1][0], self.position[i + 1][1])
                if self.food > 2:
                    self.graph[2][int(self.position[0][0] // 20), int(self.position[0][1] // 20)] = 0
                    self.graph[2][int(self.position[-3][0] // 20), int(self.position[-3][1] // 20)] = 1
                    self.graph[1][int(self.position[-3][0] // 20), int(self.position[-3][1] // 20)] = 0
                self.graph[1][int(self.position[-2][0] // 20), int(self.position[-2][1] // 20)] = 1
            self.position[-1] = (x, y)
            self.graph[0][int(self.position[-1][0] // 20), int(self.position[-1][1] // 20)] = 1

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append((self.x, self.y))
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or (self.x, self.y) in self.position \
                or (self.x, self.y) in self.obstacle:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def advisePosition(self, game_width, game_height, x_food, y_food):
        done = False
        directions = [[0, 20], [0, -20], [20, 0], [-20, 0]]
        while not done:
            self.advisedPosition = []
            pos = (x_food, y_food)
            r = randint(0,3)
            for index in range(self.food):
                if pos in self.advisedPosition or pos in self.obstacle:
                    if fail == 2:
                        break
                    pos = (pos[0] - directions[r][0], pos[1] - directions[r][1])
                    
                    if fail == 0:
                        if r < 2:
                            r = 2 + randint(0,1)
                        else:
                            r = randint(0,1)
                    elif fail == 1:
                        if r % 2 == 0:
                            r += 1
                        else:
                            r -= 1
                            
                    pos = (pos[0] + directions[r][0], pos[1] + directions[r][1])
                    fail += 1
                    index -= 1
                    continue
                self.advisedPosition.append(pos)
                pos = (pos[0] + directions[r][0], pos[1] + directions[r][1])
                fail = 0

            if self.headTailWay(game_width, game_height, self.advisedPosition):
                done = True
        for adp in self.advisedPosition:
            self.graph[6][int(adp[0] // 20), int(adp[1] // 20)] = 1

    def headTailWay(self, game_width, game_height, target):
        visited = np.zeros_like(self.obsPotential)
        direct = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        q1, q2 = [[int(target[0][0] // 20), int(target[0][1] // 20)]], []
        while len(q1) > 0:
            while len(q1) > 0:
                visited[q1[-1][0], q1[-1][1]] = 1
                for dir in direct:
                    tx, ty = q1[-1][0] + dir[0], q1[-1][1] + dir[1]
                    if tx * 20 == target[-1][0] and ty * 20 == target[-1][1]:
                        return True
                    if tx < 0 or tx >= game_width // 20 or ty < 0 or ty >= game_height // 20 or visited[tx, ty] == 1 or (tx * 20, ty * 20) in target or (tx * 20, ty * 20) in self.obstacle:
                        continue
                    q2.append([tx, ty])
                q1.pop()
            q1, q2 = q2, q1
        return False

    def getBodyPotential(self):
        now = np.array([self.x, self.y])
        pot = 0
        for pos in self.position:
            temp = np.array(pos)
            pot += np.sum(np.abs(temp - now))
        pot /= 20 * self.food
        return pot

    def display_player(self, x, y, food, game):
        self.position[-1] = (x, y)

        if game.crash == False:
            for i in range(1, len(self.advisedPosition)):
                game.gameDisplay.blit(self.adviseImage, (self.advisedPosition[i][0], self.advisedPosition[i][1]))
                
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
            
            for obs in self.obstacle:
                game.gameDisplay.blit(self.obsImage, (obs[0], obs[1]))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):
    def __init__(self, game, player):
        self.x_food = 240
        self.y_food = 200
        player.graph[5][12, 10] = 1
        self.food_coord(game, player)
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if (self.x_food, self.y_food) not in player.position and (self.x_food, self.y_food) not in player.obstacle:
            player.graph[5][int(self.x_food // 20), int(self.y_food // 20)] = 1
            for adp in player.advisedPosition:
                player.graph[6][int(adp[0] // 20), int(adp[1] // 20)] = 0
            player.advisePosition(game.game_width, game.game_height, self.x_food, self.y_food)
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        player.graph[5][int(food.x_food // 20), int(food.y_food // 20)] = 0
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()

@gif.frame
def get_frame():
    pygame.image.save(pygame.display.get_surface(), "img/temp.png")
    frame = cv.imread("img/temp.png")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    plt.imshow(frame)

def initialize_game(player, game, food, agent, batch_size, short_batch):
    state_init1 = agent.get_state(game, player, food)
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size, short_batch)


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev


def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = ResNetAgent(params)
    #light medium heavy agent
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    high_score = [0, 0]
    slot = 0
    state_old_cont = [torch.zeros((params['batch_size_short'], 7, 22, 22), dtype=torch.float32).to(DEVICE), 
                      torch.zeros((params['batch_size_short'], 11), dtype=torch.float32).to(DEVICE)]
    state_new_cont = [torch.zeros((params['batch_size_short'], 7, 22, 22), dtype=torch.float32).to(DEVICE), 
                      torch.zeros((params['batch_size_short'], 11), dtype=torch.float32).to(DEVICE)]
    action_cont = torch.zeros((params['batch_size_short'], 3), dtype=torch.float32).to(DEVICE)
    reward_cont = torch.zeros(params['batch_size_short'], dtype=torch.float32).to(DEVICE)
    done_cont = torch.zeros(params['batch_size_short'], dtype=torch.float32).to(DEVICE)
    while counter_games < params['episodes']:
        start = time.time()
        frames = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(440, 440, params)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params['batch_size'], params['batch_size_short'])

        if params['display'] and (not params['train'] or ((counter_games + 1) % 10 == 0)):
            display(player1, food1, game, record)
            frame = get_frame() #np.fromstring(pygame.display.get_surface().get_buffer().raw, dtype='b').reshape(440, 500, -1).transpose(1, 0, 2)
            frames.append(frame)
        
        steps = 0       # steps since the last positive reward
        if not params['train']:
            agent.epsilon = 0
        else:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

        while (not game.crash) and (steps < 100):
            # get old state
            state_old = agent.get_state(game, player1, food1)
            state_old_cont[0][slot] = torch.tensor(state_old[0], dtype=torch.float32).to(DEVICE)
            state_old_cont[1][slot] = torch.tensor(state_old[1], dtype=torch.float32).to(DEVICE)
            #if (counter_games + 1) % 10 == 0:
                #print("graph:\n", state_old[0])
            if params['reward'] == 1:
                adv_old = len(set(player1.position).intersection(set(player1.advisedPosition)))
            if params['reward'] == 2:
                adv_old = len(set(player1.position).intersection(set(player1.advisedPosition)))
                state_diff_old = np.array([player1.x - food1.x_food, player1.y - food1.y_food])
                #obs_pot_old = player1.obsPotential[int(player1.x // 20), int(player1.y // 20)]
                #body_pot_old = player1.getBodyPotential()

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0,2)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_graph_tensor = torch.tensor(state_old[0].reshape(1, state_old[0].shape[0], state_old[0].shape[1], state_old[0].shape[2]), dtype=torch.float32).to(DEVICE)
                    state_old_vector_tensor = torch.tensor(state_old[1].reshape(1, 11), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_graph_tensor, state_old_vector_tensor)
                    #mask
                    if not (state_old[1][0] and state_old[1][1] and state_old[1][2]):
                        if state_old[1][0]:
                            prediction[0, 0] = 0
                        if state_old[1][1]:
                            prediction[0, 1] = 0
                        if state_old[1][2]:
                            prediction[0, 2] = 0
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
            action_cont[slot] = torch.tensor(final_move, dtype=torch.float32).to(DEVICE)
                
            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
                
            state_new = agent.get_state(game, player1, food1)
            state_new_cont[0][slot] = torch.tensor(state_new[0], dtype=torch.float32).to(DEVICE)
            state_new_cont[1][slot] = torch.tensor(state_new[1], dtype=torch.float32).to(DEVICE)
            if params['reward'] == 1:
                adv_new = len(set(player1.position).intersection(set(player1.advisedPosition)))
                adv_diff = (adv_new - adv_old)
            if params['reward'] == 2:
                state_diff_new = np.array([player1.x - food1.x_food, player1.y - food1.y_food])
                #obs_pot_new = player1.obsPotential[int(player1.x // 20), int(player1.y // 20)]
                #body_pot_new = player1.getBodyPotential()
                adv_new = len(set(player1.position).intersection(set(player1.advisedPosition)))
                potential_diff = ((np.sum(np.abs(state_diff_old)) - np.sum(np.abs(state_diff_new))) / (4 * 20) / 2 + 
                              #(obs_pot_new - obs_pot_old) / 1 + 
                              #(body_pot_old - body_pot_new) / 2 + 
                              (1 if adv_new > adv_old else 0) / 2
                              ) * 5

            # set reward for the new state
            if params['reward'] == 0:
                reward = agent.set_reward(player1, game.crash)
            elif params['reward'] == 1:
                reward = agent.set_advise_reward(player1, adv_diff, game.crash)
            elif params['reward'] == 2:
                reward = agent.set_potential_reward(player1, potential_diff, game.crash)
            reward_cont[slot] = reward

            if params['debugDisplayTime'] and (steps + 1) % 10 == 0:
                lastTime = time.time()
            
            # if food is eaten, steps is set to 0
            if reward >= 10:
                steps = 0

            done_cont[slot] = 0 if game.crash else 1
                
            if params['train'] and (slot + 1) % params['batch_size_short'] == 0:
                # train short memory base on the new action and state
                loss = agent.train_short_memory(state_old_cont, action_cont, reward_cont, state_new_cont, done_cont)
                #if (steps + 1) % 10 == 0:
                    #print("loss:", loss)
                #train short memory, interval
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            slot = (slot + 1) % params['batch_size_short']
            if params['debugDisplayTime'] and (steps + 1) % 10 == 0:
                print(f"\ttrain: {time.time() - lastTime}s", end="")
                lastTime = time.time()

            record = get_record(game.score, record)
            if params['display'] and (not params['train'] or ((counter_games + 1) % 10 == 0)):
                display(player1, food1, game, record)
                frame = get_frame()
                frames.append(frame)
                pygame.time.wait(params['speed'])

            if params['debugDisplayTime'] and (steps + 1) % 10 == 0:
                print(f"\tdisplay: {time.time() - lastTime}s")
                lastTime = time.time()
                
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'], params['batch_size_short'])
        counter_games += 1
        total_score += game.score
        if params['display'] and (not params['train'] or ((counter_games) % 10 == 0)):
            gif.save(frames, params['gif_path'] + ("train/" if params['train'] else "test/") + f'{counter_games}.gif', duration=5)
        line = f'Game {counter_games}      Score: {game.score}      Time: {time.time() - start}s'
        print(line)
        with open(params['gif_path'] + "log.txt", 'a') as f:
            f.writelines(line + '\n')
        if game.score > high_score[1]:
            high_score = [counter_games, game.score]
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    with open(params['gif_path'] + "log.txt", 'a') as f:
        f.writelines(f"{('Train: ' if params['train'] else 'Test: ')}score {high_score[1]} in game {high_score[0]}\n")
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    #if params['plot_score']:
        #plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--displayTime", action='store_true')
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--reward", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    params['train'] = args.train
    params['display'] = args.display
    params['speed'] = args.speed
    params['obs'] = args.obs
    params['episodes'] = args.epochs
    params['reward'] = args.reward
    params['debugDisplayTime'] = args.displayTime
    params['batch_size_short'] = args.batch_size
    print("Args", args)
    if args.bayesianopt:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_RL()
    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['episodes'] = 100
        params['load_weights'] = True
        run(params)