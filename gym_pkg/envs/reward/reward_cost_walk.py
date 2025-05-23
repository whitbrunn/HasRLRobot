

def _get_reward(self, foot_pos, rw=(0.15, 0.15, 0.3, 0.2, 0.2, 0, 0), multiplier=500):
    qpos = np.copy(self.cassim.qpos())
    qvel = np.copy(self.cassim.qvel())

    left_foot_pos = foot_pos[:3]
    right_foot_pos = foot_pos[3:]

    # midfoot position
    foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

    # A. Task Rewards

    # 1. Pelvis Orientation 
    target_pose = np.array([1, 0, 0, 0])
    pose_error = 1 - np.inner(qpos[3:7], target_pose) ** 2

    r_pose = np.exp(-1e5 * pose_error ** 2)

    # 2. CoM Position Modulation
    # 2a. Horizontal Position Component (target position is the center of the support polygon)
    xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                0.5 * (foot_pos[1] + foot_pos[4])])

    xy_com_pos = np.exp(-np.sum(qpos[:2] - xy_target_pos) ** 2)

    # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
    height_thresh = 0.1  # m = 10 cm
    z_target_pos = self.target_height

    if qpos[2] < z_target_pos - height_thresh:
        z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos - height_thresh)) ** 2)
    elif qpos[2] > z_target_pos + 0.1:
        z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos + height_thresh)) ** 2)
    else:
        z_com_pos = 1.

    r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

    # 3. CoM Velocity Modulation
    target_speed = np.array([self.speed, 0, 0]) # Only care the vel. along x axis
    r_com_vel = np.exp(-np.linalg.norm(target_speed - qvel[:3]) ** 2)

    # 4. Feet Width
    width_thresh = 0.02     # m = 2 cm
    target_width = 0.18     # m = 18 cm seems to be optimal
    feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])

    if feet_width < target_width - width_thresh:
        r_foot_width = np.exp(-multiplier * (feet_width - (target_width - width_thresh)) ** 2)
    elif feet_width > target_width + width_thresh:
        r_foot_width = np.exp(-multiplier * (feet_width - (target_width + width_thresh)) ** 2)
    else:
        r_foot_width = 1.

    # 5. Foot/Pelvis Orientation (may need to be revisited when doing turning)
    _, _, pelvis_yaw = quaternion2euler(qpos[3:7])
    foot_yaw = np.array([qpos[8], qpos[22]])
    left_foot_orient = np.exp(-multiplier * (foot_yaw[0] - pelvis_yaw) ** 2)
    right_foot_orient = np.exp(-multiplier * (foot_yaw[1] - pelvis_yaw) ** 2)

    r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

    # Total Reward
    reward = (rw[0] * r_pose
                + rw[1] * r_com_pos
                + rw[2] * r_com_vel
                + rw[3] * r_foot_width
                + rw[4] * r_fp_orient)

    if self.debug:
        print('Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}]]'.format(r_pose,
                                                                                    r_com_pos,
                                                                                    r_com_vel,
                                                                                    r_foot_width,
                                                                                    r_fp_orient))

    
    return reward


def _get_cost(self, foot_grf, cw=(0, 0.1, 0.5)):
    # 1. Ground Contact (At least 1 foot must be on the ground)
    # TODO: Only valid for walking gaits
    qpos = np.copy(self.cassim.qpos())
    c_contact = 1 if (foot_grf[0] + foot_grf[1]) == 0 else 0

    # 2. Power Consumption
    # Specs taken from RoboDrive datasheet for ILM 115x50

    # in Newton-meters
    max_motor_torques = np.array([4.66, 4.66, 12.7, 12.7, 0.99,
                                    4.66, 4.66, 12.7, 12.7, 0.99])

    # in Watts
    power_loss_at_max_torque = np.array([19.3, 19.3, 20.9, 20.9, 5.4,
                                            19.3, 19.3, 20.9, 20.9, 5.4])

    gear_ratios = np.array([25, 25, 16, 16, 50,
                            25, 25, 16, 16, 50])

    # calculate power loss constants
    power_loss_constants = power_loss_at_max_torque / np.square(max_motor_torques)

    # get output torques and velocities
    output_torques = np.array(self.cassie_state.motor.torque[:10])
    output_velocity = np.array(self.cassie_state.motor.velocity[:10])

    # calculate input torques
    input_torques = output_torques / gear_ratios

    # get power loss of each motor
    power_losses = power_loss_constants * np.square(input_torques)

    # calculate motor power for each motor
    motor_powers = np.amax(np.diag(output_torques).dot(output_velocity.reshape(10, 1)), initial=0, axis=1)

    # estimate power
    power_estimate = np.sum(motor_powers) + np.sum(power_losses)

    c_power = 1. / (1. + np.exp(-(power_estimate - self.power_threshold)))

    # 3. Falling
    c_fall = 1 if qpos[2] < self.fall_height else 0

    # Total Cost
    cost = cw[0] * c_contact + cw[1] * c_power + cw[2] * c_fall

    return cost