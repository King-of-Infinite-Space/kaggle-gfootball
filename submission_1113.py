# start executing cells from here to rewrite submission.py

from kaggle_environments.envs.football.helpers import human_readable_agent, sticky_index_to_action, Action, PlayerRole, GameMode
import math
import random
import numpy as np
from sympy import Interval

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
    # another mesurement: run 0.0087, sprint 0.0125
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
    if isinstance(coor, list):
        c = np.array(coor)
    else:
        c = coor.copy()
    if c.ndim == 1:
        c = c[None,:]
    c[:,0] = c[:,0] * 52.5
    c[:,1] = - c[:,1] * 34 / 0.42
    # upper right corner: [52.5, 34]
    return c.squeeze()

def toPolar(coors, center):
    if coors.ndim == 1:
        coors = coors[None,:]
    rel = coors - center
    rho = np.hypot(rel[:,0], rel[:,1])
    theta = np.arctan2(rel[:,1], rel[:,0])
    angle = theta * 180 / np.pi
    out = np.column_stack((rho, angle))
    if out.shape[0] == 1:
        out = out.squeeze()
    return out
    
def convertAngle(angle, max=360):
    # assert np.all(np.abs(angle)<=180)
    angle = (angle + 360) % 360. # [0, 360)
    if max == 180:
        if isinstance(angle, np.ndarray):
            angle[angle>180] -= 360
        else:
            if angle > 180:
                angle -= 360
    return angle

def angleInterval(a, b):
    # from a to b
    if a > 0 and b < 0:
        return Interval(a, 180) + Interval(-180, b)
    else:
        return Interval(a, b)

def angleBetween(start, end):
    # a, b are theta in polar coordiate in degrees
    d = end - start
    d = convertAngle(d, max=180)
    return d # (-180,180)

# facing goal: 50 m ~ 8 deg, 40-10, 30-14, 20-21

class GameState():
    def __init__(self, obs):
        self._obs = obs

        self._position = coor2meter(np.array(obs['left_team']))
        self._velocity = coor2meter(np.array(obs['left_team_direction']))
        self._position_next = self._position + self._velocity
        self._player_id = obs['active']
        self._player_position = self._position[self._player_id,:]
        self._player_velocity = self._velocity[self._player_id,:]
        self._player_heading = np.rad2deg(np.arctan2(self._player_velocity[1], self._player_velocity[0]))
        
        self._opponent_position = coor2meter(np.array(obs['right_team']))
        self._opponent_velocity = coor2meter(np.array(obs['right_team_direction']))
        self._opponent_position_next = self._opponent_position + self._opponent_velocity
    
        self._ball_position = coor2meter(obs['ball'][:2])
        # only x, y components
        self._ball_velocity = coor2meter(obs['ball_direction'][:2])

        self._position_polar = toPolar(self._position, self._player_position) 
        self._opponents_polar = toPolar(self._opponent_position, self._player_position)
        self._ball_polar = toPolar(self._ball_position, self._player_position)
        self._position_polar_next = toPolar(self._position_next, self._player_position) 
        self._opponents_polar_next = toPolar(self._opponent_position_next, self._player_position)
        
        self._goal_coors = coor2meter(np.array([[1, 0.044],[1, -0.044]]))
        self._own_goal_coors = coor2meter(np.array([[-1, -0.044],[-1, 0.044]]))
        self._directions = [Action.Right, Action.TopRight, Action.Top, Action.TopLeft, Action.Left, Action.BottomLeft, Action.Bottom, Action.BottomRight] #from right, couterclockwise
        self._offsideLine = np.sort(self._opponent_position_next[:,0])[-2]
   
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

    def _goalAngle(self, center):
        theta = toPolar(self._goal_coors, center)[:,1]
        # [lower post, upper post]
        return theta

    # slow
    def _openAngle(self, obstacles, target=None,lower=None, upper=None):
        # target is in polar
        # target = None is meant for goal angles
        # assuming obstacles in polar coordinates
        if target is not None and lower is None and upper is None:
            ha = (180 / np.pi) * 0.5 / target[0]
            lower = target[1] - ha
            upper = target[1] + ha
        angle = angleInterval(lower, upper)
        for i, pos in enumerate(obstacles):
                # exclude target and passer
                if i != self._player_id and (target is None or pos[0] < target[0]):
                    # only counts if closer to target
                    ha = (180 / np.pi) * 0.5 / pos[0]
                    ha = np.maximum(ha, 2)
                    # blow up if too close
                    angle -= angleInterval(pos[1]-ha, pos[1]+ha)
        openAngle = 0
        for i, a in enumerate(angle.boundary):
            if i % 2 == 0:
                openAngle -= float(a)
            else: 
                openAngle += float(a)
        return openAngle

    def _pathClear(self, target, obstacles):
        clear = True
        for i, pos in enumerate(obstacles):
                # exclude target and passer
                if i != self._player_id and pos[0] < target[0]:
                    # only counts if closer to target
                    ha = (180 / np.pi) * 0.5 / pos[0]
                    ha = np.maximum(ha, 2) # 1 deg ~ 25 m
                    if abs(angleBetween(target[1],pos[1])) < ha:
                        clear = False
                        break
        return clear
                    
    def _angle2direction(self, angle):
        angle = angle if angle >= 0 else 360 + angle # [0, 2pi)
        choice = int(np.floor((angle + 22.5) / 45)) % 8
        return choice
   
    def _moveTowards(self, angle, sprint=True): 
        #assuming polar coordinate
        action = self._directions[self._angle2direction(angle)]
        if action not in self._obs["sticky_actions"]:
            exception = ["sprint"] if sprint else []
            # need to release direction before change direction?
            action_to_release = self._get_active_sticky_action(exceptions=exception)
            if action_to_release != None:
                return action_to_release
        if sprint and Action.Sprint not in self._obs["sticky_actions"]:
            return Action.Sprint
        return action

    def _meetBall(self, predict=False):
        # vector calculation
        ball2 = self._ball_position + 2*self._ball_velocity
        ball2Polar = toPolar(ball2, self._player_position)
        if (not predict) or self._player_position[0] < -25 or self._ball_polar[0] < 5:
            return self._moveTowards(ball2Polar[1])
        else:
            v_ball = self._ball_velocity.squeeze()
            v_run = 0.788
            diff = []
            v_ball_norm = np.linalg.norm(v_ball)
            if v_ball_norm < 0.01:
                return self._moveTowards(ball2Polar[1])
            else:
                for i in range(8):
                    theta = 45*i
                    v_rel = v_run*np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]) - v_ball
                    dp = np.dot(v_rel, v_ball) / v_ball_norm / v_run
                    diff.append(dp)
                direction = np.argmax(diff)
                return self._moveTowards(45*direction)

    def _evaluatePosition(self, pos):
        def weightFunc(angle):
            # risk weighted by opponent direction 
            # in front (towards goal): higher risk (max 1 x multiplier)
            #  behind: lower risk (min 0.17)
            # "risk" ~ if prox. proportional to 1/r -- equiv. to "angle" ~ 1/r * 180/3.14 ~ 50 -- but choose 20 ~ 30?
           
            # w = np.cos(np.abs(np.deg2rad(angle))/2)
            # w = (0.1 + (180-abs(angle))/180)/1.1 # linear
            # w = (np.cos(np.deg2rad(angle))+1)/2 # cos
            # shouldn't be symmetric?
            offset = 0.4
            w = np.exp(-angle**2/90**2)
            # w[0] = w[0] * 0.5 # keeper
            w = (w + offset) / (1 + offset) # max = 1
            return w
        def proximityFunc(dis):
            # 1/r ? 1/r^2 ?
            inv = 1 / dis
            inv = inv * (dis < 10) # ignore if far
            return inv
        def wProxFunc(pos):
            # 30 if pos[0] > 0 else 40
            w = -10*pos[0]/52.5 + 22 # 30 - 10
            # in own half, more risk, stay away from defenders
            return w
        def dis2Edge(pos):
            dl = 52.5-abs(pos[0])
            dh = 34-abs(pos[1])
            dl = np.maximum(dl, 0.3)
            dh = np.maximum(dh, 0.3)
            return np.minimum(dl, dh)

        angles = self._goalAngle(pos)
        goal_angle = angleBetween(angles[0],angles[1])
        og_angles = toPolar(self._own_goal_coors, pos)[:,1]
        own_goal_angle = angleBetween(og_angles[0], og_angles[1])
        theta_mid = np.mean(angles)
        opponents = toPolar(self._opponent_position_next, pos)
        proximity = proximityFunc(opponents[:,0])
        # this value won't get too high

        rel_angle = angleBetween(theta_mid, opponents[:,1])
        weightedProximity = weightFunc(rel_angle) * proximity
        weightedProximity = np.sort(weightedProximity)[-3:]
        # only consider top 3 risks
        wProx = wProxFunc(pos)
        cProximity = wProx * np.sum(weightedProximity)
        # cumulated proximity

        advance = goal_angle - own_goal_angle
        value = advance - cProximity
        # print(round(goal_angle,1), round(risk,1), round(value,1))
        dEdge = dis2Edge(pos)

        value -= 20*(1/dEdge)*(dEdge < 6)
        # don't go out of play
        return value, cProximity

    def _checkPass(self, target, target_polar):
        # target in x,y
        # centered at ball-carrying player
        rho = target_polar[0]
        theta = target_polar[1]
        targetDirection = self._angle2direction(theta)
        dAngle = abs(angleBetween(self._player_heading, theta))
        # pAngle = (180 / np.pi) / rho # 1 m control radius
        obstacles = np.vstack((self._position_polar_next, self._opponents_polar_next))
        pathClear = self._pathClear(target_polar, obstacles)
        
        passType = -1 
        if target[0] > self._offsideLine:
            return -1  # offside, ignore it
        else:
            if pathClear:
                # 50m ~ 1 deg
                # ground pass
                # release?
                if dAngle < 30:
                    if rho < 25:
                        passType = Action.ShortPass
                    elif rho < 40:
                        passType = Action.LongPass
                    else:
                        passType = Action.HighPass
                else:
                    return targetDirection
            else: # no angle
                if dAngle < 30:
                    goLong = True
                    for tm in self._position_polar:
                        if tm[0] < 10 and abs(angleBetween(tm[1],self._player_heading)) < 15:
                            # ambiguous teammate options
                            goLong = False
                            break
                    if goLong:
                        for op in self._opponents_polar:
                            if op[0] < 8 and abs(angleBetween(theta, op[1])) < np.rad2deg(0.3 / op[0]):
                                # may be blocked by opponent
                                goLong = False
                                break
                    if goLong:
                        passType = Action.HighPass
                    else:
                        return targetDirection
        return passType
        # return values:
        # -1 don't pass, 0~7 move to, Short, Long, High
    
    def _evaluateOptions(self):
        # if opponents far, run towards goal
        if np.max(self._opponents_polar[:,0]) < 6:
            return self._moveTowards(toPolar(np.array([52.5,0]), self._player_position)[1])
        # probe L meter away in D directions
        else:
            if self._player_position[0] < -45 and abs(self._player_position[1]) < 10:
                if abs(self._player_heading) < 90:
                    return Action.HighPass 
            # in own small box, clear

            probeL = 1
            probeD = 8
            angles = np.arange(probeD)*2*np.pi/probeD
            moveOptions = probeL*np.stack((np.cos(angles),np.sin(angles)), axis=-1)
            posOptions = self._player_position + moveOptions
            valueRisk = np.array([self._evaluatePosition(p) for p in posOptions])
            # valueRiskF = np.array([self._evaluatePosition(2*p) for p in posOptions + moveOptions])
            moveValues = valueRisk[:,0] # advance - proximity
            risk = valueRisk[:,1]

            for i in range(probeD):
                 a = np.rad2deg(angles[i])
                 for op in self._opponents_polar_next:
                    if op[0] < probeL and abs(angleBetween(a, op[1])) < 15:
                        risk[i] += 30
                        moveValues[i] -= 30
                # don't run into opponent

            mVmax = np.max(moveValues)
            passOptions = []
            passValues = []
            gainValues = np.zeros((8,))
            for i, pos in enumerate(self._position_next):
                if i != self._player_id:
                    pV, _pRisk = self._evaluatePosition(pos)
                    if pV > mVmax:
                        gainV = pV - mVmax
                        # weigh pass and dribble
                        pos_polar = toPolar(pos, self._player_position)
                        passType = self._checkPass(pos, pos_polar)
                        if passType == Action.LongPass:
                            # 25 ~ 40 m
                            wl = 25 / pos_polar[0]
                            gainV = wl * gainV
                        if passType == Action.HighPass:
                            passBack = pos[0] < - 20 or (abs(self._player_heading) > 120 and self._player_position[0] < 20)
                            close2G = self._player_position[0] > 30 and abs(self._player_position[1]) < 21
                            if passBack or close2G:
                                # don't high pass if backwards or close to oppo goal
                                wh = -1 
                            else:
                                wh = 1 - 0.6 * self._player_position[0] / 52.5
                            # own half, more likely; vice versa
                            gainV = wh * gainV
                        # discount difficult passes
                        if passType in [Action.ShortPass, Action.LongPass, Action.HighPass]:                            
                            passOptions.append(passType)
                            passValues.append(gainV)
                        else:
                            if passType >= 0 and passType <= 7:
                                # move to directions for pass
                                gainValues[passType] += gainV
            
            totalMoveValues = moveValues - mVmax + gainValues
            # good moves are positive
            if passOptions == [] or np.max(passValues) <= 0:
                totalMoveValues[risk > 40] -= 100 # ignore risky ones
                moveDir = np.argmax(totalMoveValues)
                if gainValues[moveDir] > 0:
                    # move for pass
                    return self._moveTowards(moveDir*360/probeD, sprint=False)
                else:
                    return self._moveTowards(moveDir*360/probeD)
            else:
                return passOptions[np.argmax(passValues)]

    def _shoot(self):
        if self._player_position[0] > 20:
            lower, upper = self._goalAngle(self._player_position)
            mid = 0.5*(lower+upper)
            obstacles = np.vstack((self._position_polar_next, self._opponents_polar_next))
            openAngle = self._openAngle(obstacles, lower=lower, upper=upper)
            gkAngle = self._opponents_polar_next[0,1]
            gkDis = self._opponents_polar_next[0,0]
            dAngle = abs(angleBetween(mid, self._player_heading))

            keeperOut = gkDis < 10 and np.linalg.norm(self._opponent_position_next[0,:]-np.array([52.5, 0])) > 5
            
            if openAngle > 15 or (gkDis < 10 and openAngle > 10) or keeperOut:
                if dAngle > 45 and not (openAngle > 40 or keeperOut):
                    return self._moveTowards(mid, sprint=False)
                else:
                    if Action.Sprint in self._obs["sticky_actions"]:
                        return Action.ReleaseSprint
                    if abs(angleBetween(self._player_heading, gkAngle)) < 5 and keeperOut:
                        d = round(gkAngle / 45)
                        sign = ((gkAngle > 0)-0.5)*2
                        a = d * 45 + sign * 45
                        return self._moveTowards(a,sprint=False)
                    return Action.Shot
    
    ####################################

    ####################################
    # set pieces
    def _throw_in(self):
        if self._obs['game_mode'] == GameMode.ThrowIn:
            action_to_release = self._get_active_sticky_action(["right"])
            if action_to_release != None:
                return action_to_release
            action = Action.ShortPass
            passValues = []
            passOptions = []
            for i, pos in enumerate(self._position_next):
                if i != self._player_id:
                    gainV, _pRisk = self._evaluatePosition(pos)
                        # weigh pass and dribble
                    pos_polar = toPolar(pos, self._player_position)
                    passType = self._checkPass(pos, pos_polar)
                    if passType == Action.LongPass:
                        # 25 ~ 40 m
                        wl = 25 / pos_polar[0]
                        gainV = wl * gainV
                    if passType == Action.HighPass:
                        if pos[0] < - 20 or (abs(self._player_heading) > 120 and self._player_position[0] < 20):
                            wh = -1 # don't high pass back
                        else:
                            wh = 1 - 0.6 * self._player_position[0] / 52.5
                        # own half, more likely; vice versa
                        gainV = wh * gainV
                    # discount difficult passes
                    passOptions.append(passType)
                    passValues.append(gainV)
                    passType = passOptions[np.argmax(passValues)]
                    if passType in [Action.ShortPass, Action.LongPass, Action.HighPass]:                            
                        return passType
                    elif passType >= 0 and passType <= 7:
                        action = self._moveTowards(45*passType)
                        if action in self._obs['sticky_actions']:
                            return Action.ShortPass
                    else:
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
    
    def _takeSetPiece(self):
        for sp in [self._throw_in, self._penalty, self._corner, self._free_kick, self._goal_kick, self._kick_off]:
            action = sp()
            if action is not None:
                return action
        return Action.ShortPass # useless. just in case

    #######################################
    def takeAction(self):
        if self._obs['ball_owned_team'] == 1:
            # defending
            if abs(self._ball_position[1]) > 34: # throw in
                return Action.Idle
            return self._meetBall()
        elif self._obs['ball_owned_team'] == -1:
            # no one has the ball. chase it
            return self._meetBall(predict=True)
        else:
            # in possession
            if self._obs['game_mode'] != GameMode.Normal:
                return self._takeSetPiece()
            # charge!!!
            elif self._obs["ball_owned_player"] == self._obs["active"]:
                for kick in [self._shoot, self._evaluateOptions]:
                    action = kick()
                    # logging.debug(time()-self.startTime)
                    # print(action)
                    if action is not None:
                        return action 
                return Action.Shot # just in case
            else:
                return Action.Shot

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
    s = GameState(obs)
    action = s.takeAction()
    # if obs['ball_owned_team'] == 0:
    #     logging.debug(action)
    #     logging.debug(time()-s.startTime)
    #     logging.debug('---------------')
    return action

@human_readable_agent
def agent(obs):
    return game_agent(obs)