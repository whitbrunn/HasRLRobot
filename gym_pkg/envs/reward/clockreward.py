import numpy as np
from scipy.interpolate import PchipInterpolator
def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, have_incentive, FREQ=40,
                        for_viz=False):
    total_duration = 2 * swing_duration + 2 * stance_duration
    phaselength = total_duration * FREQ

    # NOTE: these times are being converted from time in seconds to phaselength
    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, total_duration]) * FREQ

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 0] = r_frc_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0, 1] = r_frc_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0, 0] = r_vel_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0, 1] = r_vel_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    # During right swing we want foot velocities and don't want foot forces
    if not have_incentive:
        l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(
            np.ones(2))  # penalize l vel and r force
        l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.zeros(2)  # don't incentivize l force or r vel
    else:
        l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(
            np.ones(2))  # penalize l vel and r force
        l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.ones(2)  # incentivize l force and r vel

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 2] = r_frc_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 3] = r_frc_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 2] = r_vel_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 3] = r_vel_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        if not have_incentive:
            l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(
                np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(
                2)  # don't incentivize l and r foot velocity
        else:
            l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(
                np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.ones(
                2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(
                2)  # don't incentivize l and r foot force
            l_frc_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(
                np.ones(2))  # penalize l and r foot velocity
        else:
            l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.ones(2)  # incentivize l and r foot force
            l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(
                np.ones(2))  # penalize l and r foot velocity

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 4] = r_frc_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0, 5] = r_frc_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0, 4] = r_vel_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0, 5] = r_vel_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    # During left swing we want foot forces and don't want foot velocities (from perspective of right foot)
    if not have_incentive:
        l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.zeros(2)  # don't incentivize l vel and r force
        l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(
            np.ones(2))  # penalize l force and r vel
    else:
        l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.ones(2)  # incentivize l vel and r force
        l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(
            np.ones(2))  # penalize l force and r vel

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 6] = r_frc_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 7] = r_frc_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 6] = r_vel_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 7] = r_vel_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(
                np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(
                2)  # don't incentivize l and r foot velocity
        else:
            l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(
                np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.ones(2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(
                2)  # don't incentivize l and r foot force
            l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(
                np.ones(2))  # penalize l and r foot velocity
        else:
            l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.ones(2)  # incentivize l and r foot force
            l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(
                np.ones(2))  # penalize l and r foot velocity

    ## extend the data to three cycles : one before and one after : this ensures continuity

    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[
        0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[
        0, -1] - dbl_stance_relax_offset

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[
        0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[
        0, -1] + dbl_stance_relax_offset

    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    ## Create the smoothing function with cubic spline and cutoff at limits -1 and 1
    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    if for_viz:
        repeat_time = np.linspace(r_frc_phase_points_repeated[0, 0], r_frc_phase_points_repeated[0, -1],
                                  num=int(2000 * total_duration))
        r_frc_phase_spline_out = np.vstack([repeat_time, r_frc_phase_spline(repeat_time)])
        r_vel_phase_spline_out = np.vstack([repeat_time, r_vel_phase_spline(repeat_time)])
        l_frc_phase_spline_out = np.vstack([repeat_time, l_frc_phase_spline(repeat_time)])
        l_vel_phase_spline_out = np.vstack([repeat_time, l_vel_phase_spline(repeat_time)])
        right_foot_info = [r_frc_phase_spline, r_vel_phase_spline, r_frc_phase_spline_out, r_vel_phase_spline_out,
                           r_frc_phase_points_repeated, r_vel_phase_points_repeated]
        left_foot_info = [l_frc_phase_spline, l_vel_phase_spline, l_frc_phase_spline_out, l_vel_phase_spline_out,
                          l_frc_phase_points_repeated, l_vel_phase_points_repeated]
        return right_foot_info, left_foot_info, repeat_time

    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline], phaselength