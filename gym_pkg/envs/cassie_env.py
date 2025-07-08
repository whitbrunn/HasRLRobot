import gym
from gym import spaces
from .cassiemujoco import pd_in_t, CassieSim, CassieVis, state_out_t
import os
import random
import time
import copy
import gym
from gym import spaces
import numpy as np
from math import floor
from .reward.clockreward import create_phase_reward
import math
def rotate_by_quaternion(vector, quaternion):
	q1 = np.copy(quaternion)
	q2 = np.zeros(4)
	q2[1:4] = np.copy(vector)
	q3 = inverse_quaternion(quaternion)
	q = quaternion_product(q2, q3)
	q = quaternion_product(q1, q)
	result = q[1:4]
	return result
def quaternion_product(q1, q2):
	result = np.zeros(4)
	result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
	result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return result
def inverse_quaternion(quaternion):
	result = np.copy(quaternion)
	result[1:4] = -result[1:4]
	return result
def euler2quat(z=0, y=0, x=0):
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result = np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])
    if result[0] < 0:
        result = -result
    return result

class CassieEnv(gym.Env):
    # Metadata for the rendering options, setting it to 'human' mode
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, simrate=50):
        config_path = os.path.join(os.path.dirname(__file__), 'cassiemujoco', 'cassie.xml')
        self.cassim = CassieSim(modelfile=config_path)
        self.vis = None
        self.config = config_path
        self.command_profile = "clock"
        self.input_profile = "full"
        self.clock_based = True

        # 定义观察空间
        self.observation_space, self.clock_inds, self.mirrored_obs = self.set_up_state_space(self.command_profile, self.input_profile)

        self._obs = len(self.observation_space)
        self.history = 0
        obs_dim = self._obs + self._obs * self.history
        self.history = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 定义动作空间，假设动作是 10 维连续的控制输入，范围是 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)


        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]
        self.pdctrl = pd_in_t()

        self.cassie_state = state_out_t()
        self.simrate = simrate
        self.time = 0
        self.phase = 0
        self.counter = 0
        self.phase_add = 1
        self.phaselen = 32

        self.phaselen = 32
        self.phase_add = 1

        self.stance_mode = "zero"
        self.reward_func = "clock"

        self.have_incentive = False if "no_incentive" in self.reward_func else True
        self.strict_relaxer = 0.1
        self.early_reward = False

        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34])
        self.vel_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31])

        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        self.max_orient_change = 0.2

        self.max_simrate = self.simrate + 10
        self.min_simrate = self.simrate - 20

        self.max_speed = 4.0
        self.min_speed = -0.3

        self.max_side_speed = 0.1
        self.min_side_speed = -0.1

        self.neutral_foot_orient = np.array(
            [-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])



        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        self.max_clock_reward = 1.2
        self.reward_scale =2

        self.last_pelvis_pos = self.cassim.qpos()[0:3]



        self.dynamics_randomization = True

        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03

        self.encoder_noise = 0.01

        self.damping_low = 0.3
        self.damping_high = 5.0

        self.mass_low = 0.5
        self.mass_high = 1.5

        self.fric_low = 0.4
        self.fric_high = 1.1

        self.speed = 0
        self.side_speed = 0
        self.orient_add = 0

        self.default_damping = self.cassim.get_dof_damping()
        self.default_mass = self.cassim.get_body_mass()
        self.default_ipos = self.cassim.get_body_ipos()
        self.default_fric = self.cassim.get_geom_friction()
        self.default_rgba = self.cassim.get_geom_rgba()
        self.default_quat = self.cassim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        self.prev_action = None
        self.curr_action = None
        self.prev_torque = None


    def set_up_state_space(self, command_profile, input_profile):

        full_state_est_size = 46
        speed_size = 2  # x speed, y speed
        clock_size = 2  # sin, cos
        phase_size = 5  # swing duration, stance duration, one-hot encoding of stance mode


        base_mir_obs = np.array(
            [0.1, 1, -2, 3, -4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, -16, 17, -18, 19, -20, -26, -27, 28, 29,
             30, -21, -22, 23, 24, 25, 31, -32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42])
        obs_size = full_state_est_size


        # command --> CLOCK_BASED : clock, speed

        append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size + speed_size)])
        #append_obs = np.array([46, 47, 48, 49])
        mirrored_obs = np.concatenate([base_mir_obs, append_obs])
        clock_inds = append_obs[0:clock_size].tolist()# [46, 47]
        obs_size += clock_size + speed_size

        observation_space = np.zeros(obs_size)
        mirrored_obs = mirrored_obs.tolist()

        return observation_space, clock_inds, mirrored_obs



    def rotate_to_orient(self, vec):
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient


    def _step_simulation(self, action):
        # Set the target positions for the PD controller from the action
        target = action + self.offset
        target -= self.motor_encoder_noise

        foot_pos = np.zeros(6)
        self.cassim.foot_pos(foot_pos)
        prev_foot = copy.deepcopy(foot_pos)

        self.pdctrl  = pd_in_t()# Initialize the PD input structure

        for i in range(5):
            self.pdctrl.leftLeg.motorPd.pGain[i] = self.P[i]  # define the Left leg P-gain
            self.pdctrl.rightLeg.motorPd.pGain[i] = self.P[i]  # Right leg P-gain

            self.pdctrl.leftLeg.motorPd.dGain[i] = self.D[i]  # Left leg D-gain
            self.pdctrl.rightLeg.motorPd.dGain[i] = self.D[i]  # Right leg D-gain

            self.pdctrl.leftLeg.motorPd.torque[i] = 0  # Zero feedforward torque
            self.pdctrl.rightLeg.motorPd.torque[i] = 0  # Zero feedforward torque

            self.pdctrl.leftLeg.motorPd.pTarget[i] = target[i]  # Set target position for left leg
            self.pdctrl.rightLeg.motorPd.pTarget[i] = target[i + 5]  # Set target position for right leg

            self.pdctrl.leftLeg.motorPd.dTarget[i] = 0  # Zero velocity target for left leg
            self.pdctrl.rightLeg.motorPd.dTarget[i] = 0  # Zero velocity target for right leg

        # Step the simulation forward by one step with the PD control input
        self.cassie_state = self.cassim.step_pd(self.pdctrl)
        self.cassim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        # Update the time step counter
        # self.time += 1

    def step(self, action, return_omniscient_state=False, f_term=0):

        if self.dynamics_randomization:
            simrate = np.random.uniform(self.max_simrate, self.min_simrate)
        else:
            simrate = self.simrate

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid, expected shape: {self.action_space.shape}")

        self.l_foot_frc = 0
        self.r_foot_frc = 0
        foot_pos = np.zeros(6)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0

        self.time += 1
        pos_pre = np.copy(self.cassim.qpos())[0]

        for _ in range(self.simrate):
            self._step_simulation(action)

            foot_forces = self.cassim.get_foot_forces()

            # print(f"Here is gym, foot force:{foot_forces}")

            self.l_foot_frc += foot_forces[0]
            self.r_foot_frc += foot_forces[1]
            self.cassim.foot_pos(foot_pos)
            self.l_foot_pos += foot_pos[0:3]
            self.r_foot_pos += foot_pos[3:6]
            self.l_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.cassim.xquat("left-foot")) ** 2)
            self.r_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.cassim.xquat("right-foot")) ** 2)


        self.l_foot_frc /= self.simrate
        self.r_foot_frc /= self.simrate
        self.l_foot_pos /= self.simrate
        self.r_foot_pos /= self.simrate
        self.l_foot_orient_cost /= self.simrate
        self.r_foot_orient_cost /= self.simrate
        self.phase += self.phase_add
        height = self.cassim.qpos()[2]
        self.curr_action = action

        if self.phase > self.phaselen:
            self.last_pelvis_pos = self.cassim.qpos()[0:3]
            self.phase = 0
            self.counter += 1

        if height < 0.4 or height > 3.0:
            done = True
        else:
            done = False

        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])

        reward = self._get_reward(action)

        self.prev_action = action
        # update previous torque
        self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])

        obs = self._get_obs()

        return obs, reward, done, {}

    def reset(self):
        # Reset the time step counter
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        total_duration = (0.9 - 0.25 / 3.0 * abs(self.speed)) / 2
        self.swing_duration = (0.30 + ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.stance_duration = (0.70 - ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration,
                                                                               self.stance_duration,
                                                                               self.strict_relaxer, self.stance_mode,
                                                                               self.have_incentive,

                                                                               FREQ=2000 // self.simrate)
        self.phase = random.randint(0, floor(self.phaselen))
        self.time = 0
        self.counter = 0

        self.state_history = [np.zeros(self._obs) for _ in range(self.history + 1)]


        damp = self.default_damping

        pelvis_damp_range = [[damp[0], damp[0]],
                             [damp[1], damp[1]],
                             [damp[2], damp[2]],
                             [damp[3], damp[3]],
                             [damp[4], damp[4]],
                             [damp[5], damp[5]]]  # 0->5

        hip_damp_range = [[damp[6] * self.damping_low, damp[6] * self.damping_high],
                          [damp[7] * self.damping_low, damp[7] * self.damping_high],
                          [damp[8] * self.damping_low, damp[8] * self.damping_high]]  # 6->8 and 19->21

        achilles_damp_range = [[damp[9] * self.damping_low, damp[9] * self.damping_high],
                               [damp[10] * self.damping_low, damp[10] * self.damping_high],
                               [damp[11] * self.damping_low, damp[11] * self.damping_high]]  # 9->11 and 22->24

        knee_damp_range = [[damp[12] * self.damping_low, damp[12] * self.damping_high]]  # 12 and 25
        shin_damp_range = [[damp[13] * self.damping_low, damp[13] * self.damping_high]]  # 13 and 26
        tarsus_damp_range = [[damp[14] * self.damping_low, damp[14] * self.damping_high]]  # 14 and 27

        heel_damp_range     = [[damp[15], damp[15]]]                                      # 15 and 28
        fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
        prod_damp_range     = [[damp[17], damp[17]]]                                      # 17 and 30
        foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

        side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
        damp_range = pelvis_damp_range + side_damp + side_damp
        damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

        m = self.default_mass
        pelvis_mass_range = [[self.mass_low * m[1], self.mass_high * m[1]]]  # 1
        hip_mass_range = [[self.mass_low * m[2], self.mass_high * m[2]],  # 2->4 and 14->16
                          [self.mass_low * m[3], self.mass_high * m[3]],
                          [self.mass_low * m[4], self.mass_high * m[4]]]

        achilles_mass_range    = [[self.mass_low*m[5], self.mass_high*m[5]]]    # 5 and 17
        knee_mass_range        = [[self.mass_low*m[6], self.mass_high*m[6]]]    # 6 and 18
        knee_spring_mass_range = [[self.mass_low*m[7], self.mass_high*m[7]]]    # 7 and 19
        shin_mass_range        = [[self.mass_low*m[8], self.mass_high*m[8]]]    # 8 and 20
        tarsus_mass_range      = [[self.mass_low*m[9], self.mass_high*m[9]]]    # 9 and 21
        heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]]  # 10 and 22
        fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]]  # 11 and 23
        prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]]  # 12 and 24
        foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]]  # 13 and 25

        side_mass = hip_mass_range + achilles_mass_range \
                    + knee_mass_range + knee_spring_mass_range \
                    + shin_mass_range + tarsus_mass_range \
                    + heel_spring_mass_range + fcrank_mass_range \
                    + prod_mass_range + foot_mass_range

        mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
        mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

        delta = 0.0
        com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

        fric_noise = []
        translational = np.random.uniform(self.fric_low, self.fric_high)
        torsional = np.random.uniform(1e-4, 5e-4)
        rolling = np.random.uniform(1e-4, 2e-4)
        for _ in range(int(len(self.default_fric) / 3)):
            fric_noise += [translational, torsional, rolling]

        self.cassim.set_dof_damping(np.clip(damp_noise, 0, None))
        self.cassim.set_body_mass(np.clip(mass_noise, 0, None))
        self.cassim.set_body_ipos(com_noise)
        self.cassim.set_geom_friction(np.clip(fric_noise, 0, None))



        self.cassim.set_geom_quat(self.default_quat)

        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)


        self.cassim.set_const()

        self.last_pelvis_pos = self.cassim.qpos()[0:3]

        self.cassie_state = self.cassim.step_pd(self.pdctrl)

        self.orient_add = 0  # random.randint(-10, 10) * np.pi / 25
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0


        # self.set_slope()  # reset degree every time
        obs= self._get_obs()

        return obs



    def render(self, mode='human'):
        if mode == 'human':
            if self.vis is None:
                self.vis = CassieVis(self.cassim, self.config)  # 假设 CassieVis 是用于可视化模拟的类
            return self.vis.draw(self.cassim)
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")







    def _get_reward(self, action):

        qpos = np.copy(self.cassim.qpos())
        qvel = np.copy(self.cassim.qvel())

        # 标准化脚的力量和速度
        desired_max_foot_frc = 350
        desired_max_foot_vel = 2.0
        orient_targ = np.array([1, 0, 0, 0])

        com_vel = qvel[0]  #只关注X方向速度
        # 对FRC和vel设置一个上限
        normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
        normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
        normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
        normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

        com_orient_error = 0
        foot_orient_error = 0
        com_vel_error = 0

        # com orient error
        com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

        # foot orient error
        foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

        # com vel error
        com_vel_error += np.linalg.norm(com_vel - self.speed)



        straight_diff = np.abs(qpos[1])  # straight difference penalty
        if straight_diff < 0.05:
            straight_diff = 0
        height_diff = np.abs(qpos[2] - 0.9)
        deadzone_size = 0.05 + 0.05 * self.speed
        if height_diff < deadzone_size:
            height_diff = 0

        pelvis_acc = 0.25 * (np.abs(self.cassie_state.pelvis.rotationalVelocity[:]).sum() + np.abs(
            self.cassie_state.pelvis.translationalAcceleration[:]).sum())
        pelvis_motion = straight_diff + height_diff + pelvis_acc


        left_frc_clock = self.max_clock_reward*self.left_clock[0](self.phase)
        right_frc_clock = self.max_clock_reward*self.right_clock[0](self.phase)
        left_vel_clock = self.max_clock_reward*self.left_clock[1](self.phase)
        right_vel_clock = self.max_clock_reward*self.right_clock[1](self.phase)

        left_frc_score = np.tan(np.pi / 4 * left_frc_clock * normed_left_frc)

        left_vel_score = np.tan(np.pi / 4 * left_vel_clock * normed_left_vel)
        right_frc_score = np.tan(np.pi / 4 * right_frc_clock * normed_right_frc)
        right_vel_score = np.tan(np.pi / 4 * right_vel_clock * normed_right_vel)

        foot_frc_score = left_frc_score + right_frc_score
        foot_vel_score = left_vel_score + right_vel_score

        hip_roll_penalty = np.abs(qvel[6]) + np.abs(qvel[13])
        torque = np.asarray(self.cassie_state.motor.torque[:])
        torque_penalty = 0.25 * (sum(np.abs(self.prev_torque - torque)) / len(torque))

        action_penalty = 5 * sum(np.abs(self.prev_action - action)) / len(action)

        reward = 0.200 * foot_frc_score + \
                 0.200 * foot_vel_score + \
                 0.200 * np.exp(-(com_orient_error + foot_orient_error)) + \
                 0.150 * np.exp(-pelvis_motion) + \
                 0.150 * np.exp(-com_vel_error) + \
                 0.050 * np.exp(-hip_roll_penalty) + \
                 0.025 * np.exp(-torque_penalty) + \
                 0.025 * np.exp(-action_penalty)
        return reward*self.reward_scale

    def _get_obs(self):
        clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                 np.cos(2 * np.pi * self.phase / self.phaselen)]
        ext_state = np.concatenate((clock, [self.speed, self.side_speed]))

        new_orient = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        new_translationalVelocity = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
        new_translationalAcceleleration = self.rotate_to_orient(self.cassie_state.pelvis.translationalAcceleration[:])

        motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
        joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise

        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],  # pelvis height
            new_orient,  # pelvis orientation
            motor_pos,  # actuated joint positions
            new_translationalVelocity,  # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],  # pelvis rotational velocity
            self.cassie_state.motor.velocity[:],  # actuated joint velocities
            new_translationalAcceleleration,  # pelvis translational acceleration
            joint_pos,  # unactuated joint positions
            self.cassie_state.joint.velocity[:]  # unactuated joint velocities
        ])

        state = np.concatenate([robot_state, ext_state])
        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history + 1]

        return np.concatenate(self.state_history)




