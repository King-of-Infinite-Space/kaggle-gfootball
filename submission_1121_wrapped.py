# start executing cells from here to rewrite submission.py

from kaggle_environments.envs.football.helpers import human_readable_agent, sticky_index_to_action, Action, PlayerRole, GameMode
import math
import random
import numpy as np

# logging.basicConfig(filename='test.log', level=logging.DEBUG)

class GameState():
    # dictionary of sticky actions
    
    def coor2meter(self, coor):
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

    def toPolar(self, coors, center):
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
        
    def convertAngle(self, angle, max=360):
        # assert np.all(np.abs(angle)<=180)
        angle = (angle + 360) % 360. # [0, 360)
        if max == 180:
            if isinstance(angle, np.ndarray):
                angle[angle>180] -= 360
            else:
                if angle > 180:
                    angle -= 360
        return angle

    def angleBetween(self, start, end):
        # a, b are theta in polar coordiate in degrees
        d = end - start
        d = self.convertAngle(d, max=180)
        return d # (-180,180)

    def mergeIntervals(self, arr): 
        arr.sort(key = lambda x: x[0])  
        m = [] 
        s = -10000
        max = -100000
        for i in range(len(arr)): 
            a = arr[i] 
            if a[0] > max: 
                if i != 0: 
                    m.append([s,max]) 
                max = a[1] 
                s = a[0] 
            else: 
                if a[1] >= max: 
                    max = a[1] 
        #'max' value gives the last point of  
        # that particular interval 
        # 's' gives the starting point of that interval 
        # 'm' array contains the list of all merged intervals 
        if max != -100000 and [s, max] not in m: 
            m.append([s, max])
        return m

    def angleOpen(self, lower, upper, intervals):
        a = upper - lower
        for intv in intervals:
            if intv[0] < upper and intv[1] > lower:
                da = np.minimum(intv[1], upper) - np.maximum(intv[0], lower)
                a -= da
        return np.maximum(a, 0)

    # facing goal: 50 m ~ 8 deg, 40-10, 30-14, 20-21

    def inBox(self, pos, dir=1,extend_x=0, extend_y=0):
        x_box = 52.5 - 16.5 + extend_x
        y_box = 20.15 + extend_y
        if dir:
            inside = pos[0]>x_box and abs(pos[1])<y_box
        else:
            inside = pos[0]<-x_box and abs(pos[1])<y_box
        return inside

    def inSmallBox(self, pos, dir=1,extend_x=0, extend_y=0):
        x_box = 52.5 - 5.5 + extend_x
        y_box = 9.15 + extend_y
        if dir:
            inside = pos[0]>x_box and abs(pos[1])<y_box
        else:
            inside = pos[0]<-x_box and abs(pos[1])<y_box
        return inside

    def __init__(self, obs):
        # self.startTime = time()
        
        self._obs = obs
        self._npred = 1 # timesteps to predict for position_next
        self._position = self.coor2meter(np.array(obs['left_team']))
        self._velocity = self.coor2meter(np.array(obs['left_team_direction']))
        self._position_next = self._position + self._velocity * self._npred
        self._player_id = obs['active']
        self._player_position = self._position[self._player_id,:]
        self._player_velocity = self._velocity[self._player_id,:]
        self._player_heading = np.rad2deg(np.arctan2(self._player_velocity[1], self._player_velocity[0]))
        
        self._opponent_position = self.coor2meter(np.array(obs['right_team']))
        self._opponent_velocity = self.coor2meter(np.array(obs['right_team_direction']))
        self._opponent_position_next = self._opponent_position + self._opponent_velocity * self._npred
    
        self._ball_position = self.coor2meter(obs['ball'][:2])
        # only x, y components
        self._ball_velocity = self.coor2meter(obs['ball_direction'][:2])
        self._ball_speed = np.linalg.norm(self._ball_velocity)
        self._position_polar = self.toPolar(self._position, self._player_position) 
        self._opponents_polar = self.toPolar(self._opponent_position, self._player_position)
        self._ball_polar = self.toPolar(self._ball_position, self._player_position)
        self._position_polar_next = self.toPolar(self._position_next, self._player_position) 
        self._opponents_polar_next = self.toPolar(self._opponent_position_next, self._player_position)
        
        self._goal_coors = self.coor2meter(np.array([[1, 0.044],[1, -0.044]]))
        self._goal_center_angle = self.toPolar(np.array([52.5,0]), self._player_position)[1]
        self._own_goal_coors = self.coor2meter(np.array([[-1, -0.044],[-1, 0.044]]))
        self._directions = [Action.Right, Action.TopRight, Action.Top, Action.TopLeft, Action.Left, Action.BottomLeft, Action.Bottom, Action.BottomRight] #from right, couterclockwise
        self._offsideLine = np.sort(self._opponent_position[:,0])[-2] - 0.4
        self.sticky_actions = {
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
    def _get_active_sticky_action(self, exceptions=[]):
        """ get release action of the first active sticky action, except those in exceptions list """
        release_action = None
        for k in self.sticky_actions:
            if k not in exceptions and self.sticky_actions[k] in self._obs["sticky_actions"]:
                if k == "sprint":
                    release_action = Action.ReleaseSprint
                elif k == "dribble":
                    release_action = Action.ReleaseDribble
                else:
                    release_action = Action.ReleaseDirection
                break
        return release_action

    def _goalAngle(self, center):
        theta = self.toPolar(self._goal_coors, center)[:,1]
        # [lower post, upper post]
        return theta

    # slow
    # @timing
    def _openAngle(self, obstacles, target=None,lower=None, upper=None):
        # target is in polar
        # target = None is meant for goal angles
        # assuming obstacles in polar coordinates
        if target is not None and lower is None and upper is None:
            ha = (180 / np.pi) * 0.5 / target[0]
            lower = target[1] - ha
            upper = target[1] + ha
        intervals = []
        for i, pos in enumerate(obstacles):
                # exclude target and passer
                if i != self._player_id and (target is None or pos[0] < target[0]):
                    # only counts if closer to target
                    ha = (180 / np.pi) * 0.5 / pos[0]
                    ha = np.maximum(ha, 2)
                    # blow up if too close
                    intervals.append([pos[1]-ha, pos[1]+ha])
        intervals = self.mergeIntervals(intervals)
        openAngle = self.angleOpen(lower, upper, intervals)
        return openAngle

    def _pathClear(self, target, obstacles):
        clear = True
        a_min = 6
        if target[0] > 10:
            a_min += 0.6 * (target[0] - 10)**1.2
        for i, pos in enumerate(obstacles):
                # exclude target and passer
                if i != self._player_id and pos[0] < target[0]:
                    # only counts if closer to target
                    ha = (180 / np.pi) * 0.5 / pos[0]
                    ha = np.maximum(ha, a_min) # 1 deg ~ 25 m
                    if abs( self.angleBetween(target[1], pos[1])) < ha:
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

    def _meetBall(self, n_pred=6):
        # vector calculation
        # v_oppo = np.array([0,0])
        # if self._obs['ball_owned_team'] == 1: # opponent has ball
        #     v_oppo = self._opponent_velocity[self._obs['ball_owned_player']]
        # v_oppo_norm = np.linalg.norm(v_oppo)
        # v_ball_norm = np.linalg.norm(v_ball)
        v = self._ball_velocity #if v_ball_norm > v_oppo_norm else v_oppo
        ballFuture = self._ball_position + n_pred * v # n_pred steps ahead
        ballFuturePolar = self.toPolar(ballFuture, self._player_position)
        return self._moveTowards(ballFuturePolar[1])
    
    def _meetBallThere(self):
        if self._ball_speed < 0.01:
            return self._meetBall()
        else:
            v_run = 0.788
            diff = []
            for i in range(8):
                theta = 45*i
                v_rel = v_run*np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]) - self._ball_velocity
                dp = np.dot(v_rel, self._ball_velocity) / self._ball_speed / v_run
                diff.append(dp)
        direction = np.argmax(diff)
        return self._moveTowards(45*direction)
    
    def _goToMidpoint(self):
        mid = 0.5 * (self._ball_position + np.array([-52.5, 0]))
        _rho, theta = self.toPolar(mid, self._player_position)
        return self._moveTowards(theta)

    def _evaluatePosition(self, pos):
        def riskFunc(dis, rel_angle):
            # risk weighted by opponent direction 
            # in front (towards goal): higher risk (max 1 x multiplier)
            #  behind: lower risk (min 0.17)
            # "risk" ~ if prox. proportional to 1/r -- equiv. to "angle" ~ 1/r * 180/3.14 ~ 50 -- but choose 20 ~ 30?
           
            # w = np.cos(np.abs(np.deg2rad(angle))/2)
            # w = (0.1 + (180-abs(angle))/180)/1.1 # linear
            # w = (np.cos(np.deg2rad(angle))+1)/2 # cos
            # shouldn't be symmetric?
            offset = 0.3
            w = np.exp(-rel_angle**2/90**2)
            w = (w + offset) / (1 + offset) # max = 1
            w[0] = w[0] * 0.5 # keeper
            dis = np.maximum(dis, 0.2)
            inv = 1 / dis
            # w[dis < 1] = 1
            r = inv * w
            return r

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
            if abs(-52.5-pos[0]) < 2 and abs(pos[1]) < 3.7:
                return 0.05 # beware of own goal
            return np.minimum(dl, dh)

        angles = self._goalAngle(pos)
        goal_angle =  self.angleBetween(angles[0],angles[1])
        og_angles = self.toPolar(self._own_goal_coors, pos)[:,1]
        own_goal_angle =  self.angleBetween(og_angles[0], og_angles[1])
        theta_mid = np.mean(angles)
        opponents = self.toPolar(self._opponent_position_next, pos)
        # this value won't get too high

        rel_angle =  self.angleBetween(theta_mid, opponents[:,1])
        risk = riskFunc(opponents[:,0], rel_angle)
        # only consider top 3 risks
        wProx = wProxFunc(pos)
        cProximity = wProx * np.sum(np.sort(risk)[-3:])
        # cumulated proximity

        advance = goal_angle - own_goal_angle

        dEdge = dis2Edge(pos)

        cProximity += 30*(1/dEdge)*(dEdge < 6) # don't go out of play

        value = advance - cProximity
        # print(round(goal_angle,1), round(risk,1), round(value,1))       
        return value, cProximity

    def _checkPass(self, target, target_polar, gainV):
        # target in x,y
        # centered at ball-carrying player
        rho = target_polar[0]
        theta = target_polar[1]
        targetDirection = self._angle2direction(theta)
        dAngle = abs( self.angleBetween(self._player_heading, theta))
        # pAngle = (180 / np.pi) / rho # 1 m control radius
        obstacles = np.vstack((self._position_polar_next, self._opponents_polar_next))
        pathClear = self._pathClear(target_polar, obstacles)
        
        passType = -1 
        if target[0] > self._offsideLine or (rho > 22 and abs(self._player_heading)>100) or rho < 4:
            return -1  # offside, ignore it
        else:
            if pathClear:
                # ground pass
                # release?
                if rho < 25 and dAngle < 30:
                    passType = Action.ShortPass
                elif 25 < rho < 33 and dAngle < 20:
                    passType = Action.LongPass
                elif rho > 30 and dAngle < 30:
                    passType = Action.HighPass
                else:
                    return targetDirection
            else: # no angle
                if dAngle < 30 and rho > 25:
                    goLong = True
                    for tm in self._position_polar:
                        if tm[0] < 15 and abs( self.angleBetween(tm[1],self._player_heading)) < 15:
                            # ambiguous teammate options
                            goLong = False
                            break
                    if goLong:
                        for op in self._opponents_polar:
                            if op[0] < 8 and abs( self.angleBetween(theta, op[1])) < np.rad2deg(0.3 / op[0]):
                                # may be blocked by opponent
                                goLong = False
                                break
                    if goLong:
                        passType = Action.HighPass
                    else:
                        if gainV > 2:
                            return targetDirection
                        return -1
        return passType
        # return values:
        # -1 don't pass, 0~7 move to, Short, Long, High

    def _runClear(self, dis=10):
        clear = True
        if self._player_position[0] < -10: 
            dis += 4
        for oppo in self._opponents_polar_next:
            if oppo[0] < dis:
                if abs( self.angleBetween(oppo[1], self._goal_center_angle)) < 40:
                    clear = False
                    break
        return clear

    def _clearance(self, moveValues):
        if abs(self._player_heading) < 100:
            return Action.HighPass
        else:
            directions = 45*np.arange(8)
            angles_pi = directions * (directions <= 180) + (directions - 360) * (directions > 180)
            allowed_mask = abs(angles_pi) <= 90
            allowed_directions = directions[allowed_mask]
            mVs = moveValues[allowed_mask]
            return self._moveTowards(allowed_directions[np.argmax(mVs)],sprint=False)

    def _moveOptions(self):
        # probe L meter away in D directions
        probeL = 0.6
        probeD = 8
        angles = np.arange(probeD)*2*np.pi/probeD
        moveOptions = probeL*np.stack((np.cos(angles),np.sin(angles)), axis=-1)
        
        posOptions = self._player_position + moveOptions

        valueRisk = 0.5 * np.array([self._evaluatePosition(p) for p in posOptions]) + 0.5 * np.array([self._evaluatePosition(p) for p in self._player_position + 2*moveOptions])
        # average of short and long distance

        moveValues = valueRisk[:,0] # advance - proximity
        risk = valueRisk[:,1]
        diffi = np.zeros((probeD,))
        for i in range(probeD):
            for op in self._opponents_polar_next:
                if op[0] < 1.5:
                    cos = np.cos((angles[i]-np.deg2rad(op[1]))/2)
                    diffi[i] = 30 * cos
        # don't run into opponent
        return moveOptions, moveValues, risk, diffi

    def _passOptions(self, mVmax):
        passOptions = []
        passValues = []
        gainValues = np.zeros((8,))
        for i, pos in enumerate(self._position_next):
            if i != self._player_id:
                pV, _pRisk = self._evaluatePosition(pos)
                gainV = pV - mVmax
                if gainV > 1:
                    pos_polar = self.toPolar(pos, self._player_position)
                    passType = self._checkPass(pos, pos_polar, gainV)

                    gainV = pV - mVmax
                    gainV = gainV * np.exp(-pos_polar[0]/30)
                    
                    if passType in [Action.HighPass, Action.LongPass]:
                        passBack = pos[0] - self._player_position[0] < - 20 or abs(self._player_heading) > 100
                        if passBack:
                            gainV = 0
                    if passType == Action.HighPass:
                        close2G = self._player_position[0] > 30 and abs(self._player_position[1]) < 28
                        if close2G:
                            # don't high pass if backwards or close to oppo goal
                            wh = 0
                        else:
                            wh = 1 - 0.5 * self._player_position[0] / 52.5
                        # own half, more likely; vice versa
                        gainV = wh * gainV

                    if gainV < 0.5:
                        passType = -1
                    # discount difficult passes
                    if passType in [Action.ShortPass, Action.LongPass, Action.HighPass]:
                        passOptions.append(passType)
                        passValues.append(gainV)
                    else:
                        if passType >= 0 and passType <= 7:
                            # move to directions for pass
                            gainValues[passType] += gainV
        return passOptions, passValues, gainValues
    
    def _evaluateOptions(self):
        # if opponents ahead far, run towards goal
        if self._player_position[0] < 34 and self._runClear():
            return self._moveTowards(self._goal_center_angle)

        currentValue, currentRisk = self._evaluatePosition(self._player_position)                          

        moveOptions, moveValues, risk, diffi = self._moveOptions() 

        if self._player_position[0] < -30 and (currentRisk > 25 or self.inSmallBox(self._player_position,dir=-1, extend_x=2)):
            return self._clearance(moveValues-diffi)

        mVmax = np.max(moveValues)

        passOptions, passValues, gainValues = self._passOptions(mVmax)
        totalMoveValues = moveValues - mVmax + gainValues - diffi
        # good moves are positive
        if passOptions == [] or np.max(passValues) <= 0:
            # totalMoveValues[risk > 40] -= 100 # ignore risky ones
            moveDir = np.argmax(totalMoveValues)
            if gainValues[moveDir] > 0:
                # move for pass
                return self._moveTowards(45*moveDir, sprint=False)
            else:
                return self._moveTowards(45*moveDir)
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
            dAngle = abs( self.angleBetween(mid, self._player_heading))

            keeperOut = gkDis < 20 and np.linalg.norm(self._opponent_position_next[0,:]-np.array([52.5, 0])) > 5
            
            if openAngle > 15 or (gkDis < 10 and openAngle > 10) or keeperOut:
                if dAngle > 45 and not openAngle > 40:
                    return self._moveTowards(mid, sprint=False)
                else:
                    if Action.Sprint in self._obs["sticky_actions"]:
                        return Action.ReleaseSprint
                    if abs( self.angleBetween(self._player_heading, gkAngle)) < 5 and keeperOut:
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
            # action_to_release = self._get_active_sticky_action(["right"])
            # if action_to_release != None:
            #     return action_to_release
            action = Action.ShortPass
            passValues = []
            passOptions = []
            for i, pos in enumerate(self._position_next):
                if i != self._player_id:
                    gainV, _pRisk = self._evaluatePosition(pos)
                        # weigh pass and dribble
                    pos_polar = self.toPolar(pos, self._player_position)
                    passType = self._checkPass(pos, pos_polar, gainV)
                    if passType == Action.LongPass:
                        # 25 ~ 40 m
                        wl = 25 / pos_polar[0]
                        gainV = wl * gainV
                    if passType == Action.HighPass:
                        if pos[0] < - 20 or (abs(self._player_heading) > 120):
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
        #     action_to_release = self._get_active_sticky_action(["top_right", "bottom_right"])
        #     if action_to_release is not None:
        #         return action_to_release
        #     # randomly choose direction
        #     if Action.TopRight not in self._obs["sticky_actions"] and Action.BottomRight not in self._obs["sticky_actions"]:
        #         if random.random() < 0.5:
        #             return Action.TopRight
        #         else:
        #             return Action.BottomRight
        #     return Action.Shot
            if (random.random() < 0.5 and
                    Action.TopRight not in self._obs["sticky_actions"] and
                    Action.BottomRight not in self._obs["sticky_actions"]):
                return Action.TopRight
            else:
                if Action.BottomRight not in self._obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.Shot

    def _corner(self):
        if self._obs['game_mode'] == GameMode.Corner:
        #     action_to_release = self._get_active_sticky_action(["top", "bottom"])
        #     if action_to_release != None:
        #         return action_to_release
        #     if Action.Top not in self._obs["sticky_actions"] and Action.Bottom not in self._obs["sticky_actions"]:
        #         if self._player_position[1] < 0:
        #             return Action.Top
        #         else:
        #             return Action.Bottom
        #     return Action.HighPass
            '''
            if self._player_position[1] < 0:
                if Action.TopRight not in self._obs["sticky_actions"]:
                    return Action.TopRight
            else:
                if Action.BottomRight not in self._obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.Shot
            '''
            if self._player_position[1] < 0:
                if Action.Top not in self._obs["sticky_actions"]:
                    return Action.Top
            else:
                if Action.Bottom not in self._obs["sticky_actions"]:
                    return Action.Bottom
            return Action.HighPass
    
        
    def _free_kick(self):
        if self._obs['game_mode'] == GameMode.FreeKick:
            ga = self._goalAngle(self._player_position)
            ljs = self._moveTowards(ga.mean(),sprint=False)
            if ljs not in self._obs["sticky_actions"]:
                return ljs
            if ga.ptp() > 15:
                return Action.Shot
            else:
                return Action.HighPass

    def _goal_kick(self):
        """ perform a short pass in goal kick game mode """
        if self._obs['game_mode'] == GameMode.GoalKick:
        #     action_to_release = self._get_active_sticky_action(["top_right", "bottom_right"])
        #     if action_to_release != None:
        #         return action_to_release
            # randomly choose direction
            if (random.random() < 0.5 and
                    Action.TopRight not in self._obs["sticky_actions"] and
                    Action.BottomRight not in self._obs["sticky_actions"]):
                return Action.TopRight
            else:
                if Action.BottomRight not in self._obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.ShortPass
            
    def _kick_off(self):
        if self._obs['game_mode'] == GameMode.KickOff:
            # action_to_release = self._get_active_sticky_action(["top", "bottom"])
            # if action_to_release != None:
            #     return action_to_release
            if self._player_position[1] < 0:
                if Action.Top not in self._obs["sticky_actions"]:
                    return Action.Top
            else:
                if Action.Bottom not in self._obs["sticky_actions"]:
                    return Action.Bottom
            return Action.ShortPass
    
    def _takeSetPiece(self):
        for action in [self._throw_in(), self._penalty(), self._corner(), self._free_kick(), self._goal_kick(), self._kick_off()]:
            # action = sp()
            if action is not None:
                return action
        return Action.HighPass # useless. just in case

    def _slide(self):
        op_v = self._opponent_velocity[self._obs['ball_owned_player'], :]
        op_rel = self._opponent_position[self._obs['ball_owned_player'], :] - self._player_position
        lastD = np.sum(self._position[:,0] < self._player_position[0]) <= 5 # n-th last def
        nearOp = self._opponents_polar_next[self._obs['ball_owned_player'],0] < 0.75
        towardsMe = op_v.dot(op_rel) < 0
        rightPlace = self._player_position[0] > -52.5+18
        noYellow = self._obs['left_team_yellow_card'][self._player_id] == 0
        if lastD and nearOp and towardsMe and rightPlace and noYellow:
            return Action.Slide

    def _defend(self):
        op = self._opponent_position_next[self._obs['ball_owned_player']]
        # if self._obs['ball_owned_player'] == 0:
        #     return self._goToMidpoint()
        if self._ball_speed < 0.01:
            return self._meetBall()
        if self._ball_polar[0] < 2 and abs(self._ball_polar[1]) < 70 and self.inBox(self._player_position, dir=-1):
            return self._meetBall(n_pred=4)
        if (self._ball_polar[0] < 7 or self.inBox(op, dir=-1, extend_x=3)) and not self._player_position[0] - op[0] > 1.5:
            slide = self._slide()
            if slide is not None:
                return slide
            # op not at wing
            else:
                return self._meetBall(n_pred=7)
        return self._goToMidpoint()

    #######################################
    def takeAction(self):
        if self._obs['ball_owned_team'] == 1:
            # defending
            if abs(self._ball_position[1]) > 34: # throw in
                return Action.Idle
            return self._defend()
        if self._obs['ball_owned_team'] == -1:
            # no one has the ball. chase it
            # return self._meetBallThere()
            return self._meetBall(n_pred=6)
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

@human_readable_agent
def agent(obs):
    s = GameState(obs)
    action = s.takeAction()
    # if obs['ball_owned_team'] == 0:
    #     logging.debug(action)
    #     # logging.debug(time()-s.startTime)
    #     logging.debug('---------------')
    return action