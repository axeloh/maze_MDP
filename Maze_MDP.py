#!/usr/bin/env python
# coding: utf-8

# In[12]:


class Route:
    def __init__(self, x, y, blocked, reward):
        self.x = x
        self.y = y
        self.blocked = blocked
        self.reward = reward
        self.containsAgent = False
        
    def setReward(self, reward):
        self.reward = reward
        
    def getReward(self):
        return self.reward
    
    def setContainsAgentFalse(self):
        self.containsAgent = False
        
    def setContainsAgentTrue(self):
        self.containsAgent = True
    
    def containsAgent(self):
        return self.containsAgent
    
    def block(self):
        self.blocked = True
        
    def unBlock(self):
        self.blocked = False
        
    def isBlocked(self):
        return self.blocked
        
    def canMoveLeft(self):
        return not self.blocked and self.y != 0 and not (self.x == 1 and self.y == 2)
    
    def canMoveRight(self):
        return not self.blocked and self.y != 3 and not (self.x == 1 and self.y == 0)
    
    def canMoveUp(self):
        return not self.blocked and self.x != 0 and not (self.x == 2 and self.y == 1)
    
    def canMoveDown(self):
        return not self.blocked and self.x != 2 and not (self.x == 0 and self.y == 1)
    
    def __str__(self):
        return str(self.reward if not self.blocked else '/')
    
    def __repr__(self):
        return self.__str__()
    
    
r = Route(1,1,False,-1)
r.setReward(100)
r.getReward()


# In[39]:


import random

class Maze:
    def __init__(self, generalReward):
        self.maze = [[0]*4 for x in range(3)]
        self.finished = False
        self.agentLoc = -1, -1
        self.generalReward = generalReward
        
    def makeMaze(self):
        for i in range(3):
            for j in range(4):
                self.maze[i][j] = Route(i, j, False, self.generalReward)
        self.maze[0][3].setReward(1.00)
        self.maze[1][3].setReward(-1.00)
        self.maze[0][3].block()
        self.maze[1][3].block()
        self.maze[1][1].block()
    
    def getRoute(self, x, y):
        return self.maze[x][y]
    
    def setAgentLoc(self, x, y):
        self.maze[self.agentLoc[0]][self.agentLoc[1]].setContainsAgentFalse()
        self.maze[x][y].setContainsAgentTrue()
        self.agentLoc = x, y
        
    def getAgentLoc(self):
        return self.agentLoc
    
    def updateFinished(self):
        if self.agentLoc == (0, 3):
            self.finished = True
    
    def move(self, x0, y0, action):
        if action == 0:
            return (x0,y0-1) if self.maze[x0][y0].canMoveLeft() else (x0,y0)
        if action == 1:
             return (x0+1,y0) if self.maze[x0][y0].canMoveDown() else (x0,y0)
        if action == 2:
             return (x0,y0+1) if self.maze[x0][y0].canMoveRight() else (x0,y0)
        if action == 3:
             return (x0-1,y0) if self.maze[x0][y0].canMoveUp() else (x0,y0)
    

    def actualMove(self, x0, y0, action, trans_model):
        intended = trans_model[0]
        left = intended + trans_model[1]
        right = left + trans_model[2]
        r = random.uniform(0,1)
        if r < intended:
            return self.move(x0, y0, action)
        elif r < left:
            return self.move(x0, y0, (action+1)%4)
        else:
            return self.move(x0, y0, (action+3)%4)
    
    


# In[ ]:





# In[40]:


# Training
# Using the value iteration method



class Train:
    def __init__(self, maze):
        self.maze = maze
        self.utilities = [[0]*4 for x in range(3)] # Initial guess
    
    def train(self):
        gamma = 1 # Learning rate
        n = 1000
        trans_model = [0.8, 0.1, 0.1]
        maze = self.maze
        
        # 0: left
        # 1: down
        # 2: right
        # 3 up
        
        print("Running the value iteration method:\nn = {} iterations, transition model = {}, gamma={}.".format(n, trans_model, gamma))
        for i in range(n):
            copy = [[0]*4 for r in range(3)]
            for x in range(3):
                for y in range(4):
                    sums = []

                    for action in range(4):
                        if (x == 0 and y == 3) or (x == 1 and y == 3):
                            continue
                        if action == 0:
                            sums.append(0.8*self.getUtility(self.maze.move(x, y, 0)) + 0.1*self.getUtility(self.maze.move(x, y, 1))
                                      + 0.1*self.getUtility(self.maze.move(x,y,3)))
                        if action == 1:
                            sums.append(0.8*self.getUtility(self.maze.move(x, y, 1)) + 0.1*self.getUtility(self.maze.move(x, y, 0))
                                      + 0.1*self.getUtility(self.maze.move(x,y,3)))
                        if action == 2:
                            sums.append(0.8*self.getUtility(self.maze.move(x, y, 2)) + 0.1*self.getUtility(self.maze.move(x, y, 1))
                                      + 0.1*self.getUtility(self.maze.move(x,y,3)))
                        if action == 3:
                            sums.append(0.8*self.getUtility(self.maze.move(x, y, 3)) + 0.1*self.getUtility(self.maze.move(x, y, 0))
                                      + 0.1*self.getUtility(self.maze.move(x,y,2)))

                    bestNeighbourUtility = 0 if not sums else gamma*max(sums)
                    utility = self.maze.maze[x][y].getReward() + bestNeighbourUtility
                    copy[x][y] = utility
            self.utilities = copy[:]


    def getUtility(self, coord):
        return self.utilities[coord[0]][coord[1]]
    

    def printUtilities(self):
        print("\nFinal utilities: \n")
        for i in range(len(self.utilities)):
            for j in range(len(self.utilities[0])):
                self.utilities[i][j] = round(self.utilities[i][j], 3)
            print(self.utilities[i])


# In[41]:


def find_best_action(maze, utilities, x, y):
    bestAction = -1
    bestUtility = -10000000
    for i in range(4):
        newPos = maze.move(x, y, i)
        newX, newY = newPos[0], newPos[1]
        utility = utilities[newX][newY]
        if utility >= bestUtility:
            bestUtility = utility
            bestAction = i
    return bestAction


# In[42]:


#print(find_best_action(maze, utilities, 1, 1))


# In[43]:


def printMaze(maze):
    # maze = [[' ']*4 for x in range(3)]
    out = ""
    for x in range(3):
        out += '|'
        for y in range(4):
            route = maze.getRoute(x, y)
            if route.containsAgent:
                out += 'o'
            elif route.isBlocked():
                if route.getReward() == 1:
                    out += '+1'
                elif route.getReward() == -1:
                    out += '-1'
                else:
                    out += '/'
            else:
                out += ' '
            out += '|'
        out += '\n'
        
    return out


# In[45]:


from IPython.display import clear_output
from time import sleep

# Setting up the maze
maze = Maze(-0.04)
maze.makeMaze()
maze.setAgentLoc(2,0)

# Training (finding utilities)
train = Train(maze)
train.train()
utilities = train.utilities

while not maze.finished:
    clear_output(wait=True)
    out = printMaze(maze)
    print(out)
    maze.updateFinished()
    agentLoc = maze.getAgentLoc()
    print(agentLoc)
    agent_x, agent_y = agentLoc[0], agentLoc[1]
    best_action = find_best_action(maze, utilities, agent_x, agent_y)
    newAgentLoc = maze.actualMove(agent_x, agent_y, best_action, [0.8, 0.1, 0.1])
    maze.setAgentLoc(newAgentLoc[0], newAgentLoc[1])
    sleep(.5)


# In[ ]:





# In[ ]:




