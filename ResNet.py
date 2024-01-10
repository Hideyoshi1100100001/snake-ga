import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetAgent(nn.Module):

    def __init__(self, params):
        super(ResNetAgent, self).__init__()

        num_classes = 3
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 11)
        self.fc2 = nn.Linear(22, 11)

        self.f1 = nn.Linear(11, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        # input size: 22 * 22
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        x = F.relu(self.fc2(torch.cat((x, y), 1)))
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))

        x = F.softmax(x, dim=-1)
        return x
    
    def get_state(self, game, player, food):
        """
        Return the state graph and state vector.
        The state is a graph, representing:
            0: ground
            1: food
            -1: obstacle
            -2: body
            -3: the body before head
            -4: head
        The state vector is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side  
        """

        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.obstacle) or 
                                                                 (list(map(add, player.position[-1], [20, 0])) in player.position) or
                                                                 player.position[-1][0] + 20 >= (game.game_width - 20))) or
            (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.obstacle) or
                                                                  (list(map(add, player.position[-1], [-20, 0])) in player.position) or
                                                                  player.position[-1][0] - 20 < 20)) or
            (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.obstacle) or
                                                                  (list(map(add, player.position[-1], [0, -20])) in player.position) or
                                                                  player.position[-1][-1] - 20 < 20)) or
            (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.obstacle) or
                                                                 (list(map(add, player.position[-1], [0, 20])) in player.position) or
                                                                 player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.obstacle) or
                                                                  (list(map(add,player.position[-1],[20, 0])) in player.position) or
                                                                  player.position[ -1][0] + 20 > (game.game_width-20))) or
            (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1], [-20,0])) in player.obstacle) or
                                                                 (list(map(add,player.position[-1], [-20,0])) in player.position) or
                                                                 player.position[-1][0] - 20 < 20)) or
            (player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,-20])) in player.obstacle) or
                                                                  (list(map(add,player.position[-1],[0,-20])) in player.position) or
                                                                  player.position[-1][-1] - 20 < 20)) or
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.obstacle) or
                                                                 (list(map(add,player.position[-1],[0,20])) in player.position) or
                                                                 player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger right

            (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.obstacle) or
                                                                 (list(map(add,player.position[-1],[20,0])) in player.position) or
                                                                  player.position[-1][0] + 20 > (game.game_width-20))) or
            (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1],[-20,0])) in player.obstacle) or
                                                                  (list(map(add,player.position[-1],[20,0])) in player.position) or
                                                                  player.position[-1][0] - 20 < 20)) or
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,-20])) in player.obstacle) or
                                                                 (list(map(add,player.position[-1],[20,0])) in player.position) or
                                                                 player.position[-1][-1] - 20 < 20)) or
            (player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.obstacle) or
                                                                  (list(map(add,player.position[-1],[20,0])) in player.position) or
                                                                  player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left


            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return [player.graph, np.asarray(state)]

    def set_reward(self, player, crash):
        """
        Return the reward.
        The reward is:
            -10 when Snake crashes. 
            +10 when Snake eats food
            0 otherwise
        """
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def set_potential_reward(self, player, dis, crash):
        """
        Return the potential reward.
        The reward is:
            -10 when Snake crashes. 
            +10 when Snake eats food
            potential otherwise
        """
        self.reward = dis
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_graph_tensor = torch.tensor(next_state[0].reshape(1, 1, next_state[0].shape[0], next_state[0].shape[1]), dtype=torch.float32).to(DEVICE)
            next_state_vector_tensor = torch.tensor(next_state[1].reshape(1, 11), dtype=torch.float32).to(DEVICE)
            state_graph_tensor = torch.tensor(state[0].reshape(1, 1, state[0].shape[0], state[0].shape[1]), dtype=torch.float32, requires_grad=True).to(DEVICE)
            state_vector_tensor = torch.tensor(state[1].reshape(1, 11), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_graph_tensor, next_state_vector_tensor)[0])
            output = self.forward(state_graph_tensor, state_vector_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()            

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_graph_tensor = torch.tensor(next_state[0].reshape(1, 1, next_state[0].shape[0], next_state[0].shape[1]), dtype=torch.float32).to(DEVICE)
        next_state_vector_tensor = torch.tensor(next_state[1].reshape(1, 11), dtype=torch.float32).to(DEVICE)
        state_graph_tensor = torch.tensor(state[0].reshape(1, 1, state[0].shape[0], state[0].shape[1]), dtype=torch.float32, requires_grad=True).to(DEVICE)
        state_vector_tensor = torch.tensor(state[1].reshape(1, 11), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_graph_tensor, next_state_vector_tensor)[0])
        output = self.forward(state_graph_tensor, state_vector_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()