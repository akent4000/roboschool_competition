from __future__ import annotations

from aliengo_competition import COMPETITION_ROOT_DIR


class AliengoFlatCfg:
    seed = 1

    class env:
        num_envs = 512
        num_observations = 58
        num_observation_history = 5
        num_privileged_obs = None
        num_actions = 12
        send_timeouts = True
        episode_length_s = 20.0
        env_spacing = 3.0
        observe_vel = False
        observe_command = True
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = True
        observe_two_prev_actions = False
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        record_video = False

    class terrain:
        mesh_type = "plane"
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands:
        num_commands = 15
        resampling_time = 10.0
        lin_vel_x = [-1.5, 1.5]
        lin_vel_y = [-0.75, 0.75]
        ang_vel_yaw = [-1.0, 1.0]
        body_height = [0.0, 0.0]
        gait_frequency = [3.0, 3.0]
        gait_phase = [0.0, 0.0]
        gait_offset = [0.5, 0.5]
        gait_bound = [0.5, 0.5]
        gait_duration = [0.5, 0.5]
        footswing_height = [0.08, 0.08]
        body_pitch = [-0.2, 0.2]
        body_roll = [0.0, 0.0]
        stance_width = [0.25, 0.25]
        stance_length = [0.45, 0.45]
        aux_reward_coef = [0.0, 0.0]
        default_command = [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5, 0.5, 0.5, 0.08, 0.0, 0.0, 0.25, 0.45, 0.0]
        command_curriculum = True
        curriculum_seed = 100
        num_bins_vel_x = 21
        num_bins_vel_y = 3
        num_bins_vel_yaw = 21
        num_bins_body_pitch = 11
        pacing_offset = False
        binary_phases = True
        gaitwise_curricula = False
        distributional_commands = True
        curriculum_type = "RewardThresholdCurriculum"
        limit_vel_x = [-1.5, 1.5]
        limit_vel_y = [-0.75, 0.75]
        limit_vel_yaw = [-1.0, 1.0]
        limit_body_pitch = [-0.2, 0.2]
        limit_body_height = [0.0, 0.0]
        limit_gait_frequency = [3.0, 3.0]
        limit_gait_phase = [0.0, 0.0]
        limit_gait_offset = [0.5, 0.5]
        limit_gait_bound = [0.5, 0.5]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.08, 0.08]
        limit_body_roll = [0.0, 0.0]
        limit_stance_width = [0.25, 0.25]
        limit_stance_length = [0.45, 0.45]
        limit_aux_reward_coef = [0.0, 0.0]

    class init_state:
        pos = [0.0, 0.0, 0.51]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control:
        control_type = "P"
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.8}
        action_scale = 0.25
        hip_scale_reduction = 0.5
        decimation = 4

    class asset:
        file = str(COMPETITION_ROOT_DIR / "resources" / "robots" / "aliengo" / "urdf" / "aliengo.urdf")
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 0
        flip_visual_attachments = False
        fix_base_link = False
        collapse_fixed_joints = True
        default_dof_drive_mode = 3
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01
        disable_gravity = False
        replace_cylinder_with_capsule = True

    class normalization:
        clip_actions = 10.0
        clip_observations = 100.0
        obs_scales = type(
            "ObsScales",
            (),
            {
                "lin_vel": 2.0,
                "ang_vel": 0.25,
                "dof_pos": 1.0,
                "dof_vel": 0.05,
                "command": 1.0,
                "action": 1.0,
                "body_height_cmd": 1.0,
                "gait_freq_cmd": 1.0,
                "gait_phase_cmd": 1.0,
                "gait_offset_cmd": 1.0,
                "gait_bound_cmd": 1.0,
                "gait_duration_cmd": 1.0,
                "footswing_height_cmd": 1.0,
                "body_pitch_cmd": 1.0,
                "body_roll_cmd": 1.0,
                "stance_width_cmd": 1.0,
                "stance_length_cmd": 1.0,
                "aux_reward_cmd": 1.0,
            },
        )

    class rewards:
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.30
        tracking_sigma = 0.25
        orientation_sigma = 0.25
        max_contact_force = 100.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.0
        gait_vel_sigma = 10.0
        stand_vel_on = 0.25
        stand_vel_full = 0.5

        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            tracking_body_pitch = 0.75
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            feet_clearance_cmd_linear = -30.0
            raibert_heuristic = -10.0
            stand_still = -0.2
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            orientation = -0.25
            base_height = 0.0
            torques = -1e-4
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            collision = -5.0
            dof_pos_limits = -10.0
            feet_air_time = 0.0
            feet_slip = -0.04

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        push_robots = False
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        rand_interval_s = 4.0

    class curriculum_thresholds:
        tracking_lin_vel = 0.8
        tracking_ang_vel = 0.7
        tracking_body_pitch = 0.8

    class viewer:
        pos = [8.0, 8.0, 4.0]
        lookat = [0.0, 0.0, 0.45]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 2

        class physx:
            num_threads = 1
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**20
            default_buffer_size_multiplier = 2
            contact_collection = 2


class AliengoFlatCfgPPO:
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1000
        save_interval = 50
        experiment_name = "aliengo_flat"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
