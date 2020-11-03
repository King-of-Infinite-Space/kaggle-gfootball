# start executing cells from here to rewrite submission.py

from kaggle_environments.envs.football.helpers import *
import math
import random
import numpy as np

# dictionary of sticky actions
sticky_actions = {
    "left": Action.Left,
    "top_left": Action.TopLeft,
    "top": Action.Top,
    "top_right": Action.TopRight,
    "right": Action.Right,
    "bottom_right": Action.BottomRight,
    "bottom": Action.Bottom,
    "bottom_left": Action.BottomLeft,
    "sprint": Action.Sprint,
    "dribble": Action.Dribble
}
'''
    Try to check sticky_actions before you do slide, pass, and shot, epecially the direction part. It is simulating your direction button holding action when you use controller to paly the game.
    The ball ground level height is around 0.10 - 0.15 and player can pick a high pass at height around 0.5-1.0, while goalkeeper can catch the ball at around 0.5-2.0. Gravity is around 0.098, and drag plays a role during ball flying.
    The sprint speed is around 0.015 per step ---- 7.875, ~8m/s
    When PalyerRole is involved, there will be errors somehow, which is blocking to locate enemy goalkeeper accurately. My assumption here is that the latest PR merged in the helper is not available in the scoring environment. Will try to use it later.
    The space between boal controller and ball is around 0.012 during sprint which is close to the sprint speed. We may assume this is equal to one-step running length.
    https://www.kaggle.com/sx2154/gfootball-rules-from-environment-exploration
'''

'''
    Bottom left/right corner of the field is located at [-1, 0.42] and [1, 0.42], respectively.
    Top left/right corner of the field is located at [-1, -0.42] and [1, -0.42], respectively.
    Left/right goal is located at -1 and 1 X coordinate, respectively. They span between -0.044 and 0.044 in Y coordinates.
    Speed vectors represent a change in the position of the object within a single step.
'''
def coor2meter(coor):
    c = coor.copy()
    c[:,0] = c[:,0] * 52.5
    c[:,1] = - c[:,1] * 34 / 0.42
    # upper right corner: [52.5, 34]
    return c

goal_coors = coor2meter(np.array([[1, -0.044],[1, 0.044]]))

def toPolar(coors, center):
    rel = coors - center
    rho = np.hypot(rel[:,0], rel[:,1])
    theta = np.arctan2(rel[:,1], rel[:,0])
    angle = theta * 180 / np.pi
    return np.column_stack((rho, angle))

def goalAngle(goal_coors, center):
    theta = toPolar(goal_coors, center)[:1]
    return theta[1] - theta[0]

class GameState():
    def __init__(self, obs):
        self._obs = obs
        self._position = coor2meter(np.array(obs['left_team']))
        self._velocity = coor2meter(np.array(obs['left_team_direction']))
        self._opponent_position = coor2meter(np.array(obs['right_team']))
        self._opponent_velocity = coor2meter(np.array(obs['right_team_direction']))
        self._directions = [5, 4, 3, 2, 1, 8, 7, 6] #from right, couterclockwise
        self._player_id = obs['active']
        self._player_position = self._position[self._player_id,:]  

    def _get_active_sticky_action(self, exceptions=[]):
        """ get release action of the first active sticky action, except those in exceptions list """
        release_action = None
        for k in sticky_actions:
            if k not in exceptions and sticky_actions[k] in self._obs["sticky_actions"]:
                if k == "sprint":
                    release_action = Action.ReleaseSprint
                elif k == "dribble":
                    release_action = Action.ReleaseDribble
                else:
                    release_action = Action.ReleaseDirection
                break
        return release_action     

    def _checkOffside(self):
        offsideLine = np.sort(self._opponent_position[:,0])[-2]
        self._offside = self._position[:,0] > offsideLine
    
    def _moveTowards(self, angle): #assuming xy coordinate
        angle = angle if angle > 0 else 360 + angle # [0, 2pi)
        choice = int(np.floor((angle + 22.5) / 45))
        return self._directions[choice]
    
    ####################################

    ####################################
    # set pieces
    def _throw_in(self):
        if self._obs['game_mode'] == GameMode.ThrowIn:
            action_to_release = self._get_active_sticky_action(["right"])
            if action_to_release != None:
                return action_to_release
            if Action.Right not in self._obs["sticky_actions"]:
                return Action.Right
            return Action.ShortPass

    def _penalty(self):
        if self._obs['game_mode'] == GameMode.Penalty:
            action_to_release = self._get_active_sticky_action(["top_right", "bottom_right"])
            if action_to_release is not None:
                return action_to_release
            # randomly choose direction
            if Action.TopRight not in self._obs["sticky_actions"] and Action.BottomRight not in self._obs["sticky_actions"]:
                if random.random() < 0.5:
                    return Action.TopRight
                else:
                    return Action.BottomRight
            return Action.Shot

    def _corner(self):
        if self._obs['game_mode'] == GameMode.Corner:
            action_to_release = self._get_active_sticky_action(["top", "bottom"])
            if action_to_release != None:
                return action_to_release
            if Action.Top not in self._obs["sticky_actions"] and Action.Bottom not in self._obs["sticky_actions"]:
                if self._player_position[1] < 0:
                    return Action.Top
                else:
                    return Action.Bottom
            return Action.HighPass
        
    def _free_kick(self):
        if self._obs['game_mode'] == GameMode.FreeKick:
            action_to_release = self._get_active_sticky_action(["right"])
            if action_to_release != None:
                return action_to_release
            if Action.Right not in self._obs["sticky_actions"]:
                return Action.Right
            return Action.HighPass   

    def _goal_kick(self):
        """ perform a short pass in goal kick game mode """
        if self._obs['game_mode'] == GameMode.GoalKick:
            action_to_release = self._get_active_sticky_action(["top_right", "bottom_right"])
            if action_to_release != None:
                return action_to_release
            # randomly choose direction
            if Action.TopRight not in self._obs["sticky_actions"] and Action.BottomRight not in self._obs["sticky_actions"]:
                if random.random() < 0.5:
                    return Action.TopRight
                else:
                    return Action.BottomRight
            return Action.ShortPass
            
    def _kick_off(self):
        if self._obs['game_mode'] == GameMode.KickOff:
            action_to_release = self._get_active_sticky_action(["top", "bottom"])
            if action_to_release != None:
                return action_to_release
            if Action.Top not in self._obs["sticky_actions"] and Action.Bottom not in self._obs["sticky_actions"]:
                if self._player_position[1] < 0:
                    return Action.Top
                else:
                    return Action.Bottom
            return Action.ShortPass  

    #######################################
    def getAction(self):
        return Action.Right

def go_through_opponents(self):
    """ avoid closest opponents by going around them """
    def environment_fits(self):
        """ environment fits constraints """
        # right direction is safest
        biggest_distance, final_opponents_amount = get_average_distance_to_opponents(self._obs, player_x + 0.01, player_y)
        self._obs["memory_patterns"]["go_around_opponent"] = Action.Right
        # if top right direction is safest
        top_right, opponents_amount = get_average_distance_to_opponents(self._obs, player_x + 0.01, player_y - 0.01)
        if (top_right > biggest_distance and player_y > -0.15) or (top_right == 2 and player_y > 0.07):
            biggest_distance = top_right
            final_opponents_amount = opponents_amount
            self._obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
        # if bottom right direction is safest
        bottom_right, opponents_amount = get_average_distance_to_opponents(self._obs, player_x + 0.01, player_y + 0.01)
        if (bottom_right > biggest_distance and player_y < 0.15) or (bottom_right == 2 and player_y < -0.07):
            biggest_distance = bottom_right
            final_opponents_amount = opponents_amount
            self._obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
        # is player is surrounded?
        if opponents_amount >= 3:
            self._obs["memory_patterns"]["go_around_opponent_surrounded"] = True
        else:
            self._obs["memory_patterns"]["go_around_opponent_surrounded"] = False
        return True
        
    def get_action(self):
        """ get action of this memory pattern """
        # if player is surrounded
        if self._obs["memory_patterns"]["go_around_opponent_surrounded"]:
            return Action.HighPass
        if self._obs["memory_patterns"]["go_around_opponent"] not in self._obs["sticky_actions"]:
            action_to_release = get_active_sticky_action(self._obs, ["sprint"])
            if action_to_release != None:
                return action_to_release
            return self._obs["memory_patterns"]["go_around_opponent"]
        if Action.Sprint not in self._obs["sticky_actions"]:
            return Action.Sprint
        return self._obs["memory_patterns"]["go_around_opponent"]
    



# "%%writefile -a submission.py" will append the code below to submission.py,

# @human_readable_agent wrapper modifies raw observations provided by the environment:
# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations
# Following modifications are applied:
# - Action, PlayerRole and GameMode enums are introduced.
# - 'sticky_actions' are turned into a set of active actions (Action enum)
#    see usage example below.
# - 'game_mode' is turned into GameMode enum.
# - 'designated' field is removed, as it always equals to 'active'
#    when a single player is controlled on the team.
# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.
# - Action enum is to be returned by the agent function.
def game_agent(obs):
    """ Ole ole ole ole """
    # shift positions of opponent team players ## pos. at next timestep?  
    # for i in range(len(obs["right_team"])):
    #     obs["right_team"][i][0] += obs["right_team_direction"][i][0]
    #     obs["right_team"][i][1] += obs["right_team_direction"][i][1]

    # # coordinates of the ball in the next step
    # obs["memory_patterns"]["ball_next_coords"] = {
    #     "x": obs["ball"][0] + obs["ball_direction"][0] * 10,
    #     "y": obs["ball"][1] + obs["ball_direction"][1] * 2
    # }
    
    s = GameState(obs)
    return s.getAction()

@human_readable_agent
def agent(obs):
    return game_agent(obs)