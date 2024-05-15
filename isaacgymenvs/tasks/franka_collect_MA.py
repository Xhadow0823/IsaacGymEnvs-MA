# franka multi-arm collect task
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask

# this block is for bypassing the type checking in reward function
from torch import Tensor
from typing import Dict, Tuple

from .franka_reach import axisangle2quat


class FrankaCollectMA(VecTask):

    # Override the VecTask
    def allocate_buffers(self):
        '''the self.obs_buf dimension have changed'''
        # self.num_agents = 2 # this is for testing

        # allocate all buffers
        # then override the obs and rew buffer for multi-agent
        super().allocate_buffers()
        self.obs_buf      = torch.zeros((self.num_envs * self.num_agents, self.num_obs), 
                                        device=self.device, dtype=torch.float)
        self.rew_buf      = torch.zeros(self.num_envs * self.num_agents, 
                                        device=self.device, dtype=torch.float)
        self.reset_buf    = torch.ones(self.num_envs * self.num_agents, 
                                        device=self.device, dtype=torch.long)
        self.timeout_buf  = torch.zeros(self.num_envs * self.num_agents, 
                                        device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs * self.num_agents, 
                                        device=self.device, dtype=torch.long)

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # using self.num_agents for get the num_arms_per_env

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]  # default = 3

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type (deprecated)
        self.control_type = self.cfg["env"]["controlType"]
        '''控制方式，已棄用，預設是osc'''
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        self.num_targets = self.cfg["env"].get("numTargets", -1)
        '''一個env中有多少個target(cube 數量)'''
        if self.num_targets <= -1:
            self.num_targets = self.cfg["env"].get("numAgents", -1)
        self.wall_height = None  # TODO: 參數化!
        '''牆的高度'''


        # dimensions
        _all_target_state = self.num_targets * 3  # target_pos_shape = [3]
        _other_agent_state = (self.cfg["env"].get("numAgents", 1) - 1) * 3  # agent_pos_shape = [3]
        _base_obs_state = 3 + 4 + 3 + (7)  # 10 for eef_pos, eef_quat and relative_pos   # + base pose(7)
        _FSM_obs = 1 + 1  # FSM and gFSM
        self.cfg["env"]["numObservations"] = _base_obs_state + _all_target_state + _other_agent_state + _FSM_obs
        self.cfg["env"]["numActions"] = 7        # 7 dof
        assert self.cfg["env"]["numObservations"]==(_base_obs_state+(2*3)+(1*3)+_FSM_obs)
        assert self.cfg["env"]["numActions"] == 7 

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        '''Dict[str, tensor] , 各種 actor, dof 和 rigid body state 的 partial view \ 
        (num_envs x states) or (num_envs x num_agents x states)'''
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        '''[Dict[str, int|list[int]] , rigid body name 對應到其在 env 中的 一個 actor 的 rigid body 的 idx \ 
        用 find_actor_rigid_body_handle() 依據 (env, actor, rigid_body_name) 找到的 idx \ 
        ex:{ \ 
            hand: [4 9], \ 
            cubeA: [8 ...] \ 
        }''' 

        self.num_dofs = None                    # Total number of DOFs per env
        '''int, 一個env有多少dof'''
        self.actions = None                     # Current actions to be deployed
        '''網路輸出的 actions ，這個tensor 會在 pre_physics_step 中被設定，並前處理完再套用到actor上'''

        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        ''' cubeA 的初始 actor state (姿態) \ 
        這是 root_state 的 partial view \ 
        (num_envs x num_targets x 13)
        '''
        self._cubeA_state = None                # Current state of cubeA for the current env
        ''' cubeA 的當前 actor state (姿態) \ 
        (num_envs x num_targets x 13)
        '''
        self._cubeA_ids = []                   # Actor ID corresponding to cubeA for a given env
        '''actor cubeA 的 actor handle list(就是 env 中的 ids), 是 list[num_targets]'''
        
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        '''每個env 的每個 actor state， actor 的 state 就是姿態 \ 
        (n_envs x n_actors_per_env x 13), 13 is position[3], rotation[4], linear velocity and angular velocity
        '''
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        '''每個 env 的每個 dof 的 state， state 就是 position 和 velocity \ 
        (n_envs x n_dofs_per_env x 2), 2 is position and velocity of a dof
        '''
        self._q = None  # Joint positions           (n_envs, n_dof)
        '''這是 _dof_state 的 partial view，它有每個 env 的每個 dof 的 position \ 
        (n_envs x n_dofs_per_env)
        '''
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        '''這是 _dof_state 的 partial view，它有每個 env 的每個 dof 的 velocity \ 
        (n_envs x n_dofs_per_env)
        '''
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        '''每個env的每個 rigid body 的 state （就是姿態，跟actor state 一樣）\ 
        (n_envs x n_rigid_body_per_env x 13) 13 is position[3], rotation[4], linear velocity and angular velocity
        '''
        self._contact_forces = None     # Contact forces in sim
        '''碰撞的 force 資訊，維度是 (num_envs x num_rigids_per_env x 3)'''
        self._hands_contact_forces = None
        '''hand 這個 rigid body 的contact force, 維度是 (num_envs x num_agents x 3)'''
        self._cubes_contact_forces = None
        '''cube 這個 rigid body 的 contact force, 維度是 (num_envs x num_targets x 3)'''

        self._base_state = None
        '''是_rigid_body_state 的 partial view, 手臂的 panda_link0 這個 rigid body 的 state(13) \ 
        (num_envs x num_agents x 13)
        '''
        self._eef_state = None  # end effector state (at grasping point)
        ''' 是 _rigid_body_state 的 partial view,  panda_grip_site 這個 rigid body 的 state \ 
        (num_envs x num_agents x 13)
        '''
        self._eef_lf_state = None  # end effector state (at left fingertip)
        ''' 是 _rigid_body_state 的 partial view \ 
        (num_envs x num_agents x 13)
        '''
        self._eef_rf_state = None  # end effector state (at left fingertip)
        ''' 是 _rigid_body_state 的 partial view \ 
        (num_envs x num_agents x 13)
        '''
        self._j_eef = None  # Jacobian for end effector
        ''' 是 _rigid_body_state 的 partial view \ 
        '''
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        '''是_effort_control的partial view \ 
        而_effort_control是每個env的dofs \ 
        ((num_envs*num_agents) x 7)
        '''
        self._gripper_control = None  # Tensor buffer for controlling gripper
        '''是_pos_control的partial view \ 
        而_pos_control是每個env的dofs \ 
        ((num_envs*num_agents) x 2)
        '''
        self._pos_control = None            # Position actions
        '''被用來設定 actor 的關節位置控制
        (num_envs x num_dofs)
        '''
        self._effort_control = None         # Torque actions
        '''被用來設定 actor 的功率控制
        (num_envs x num_dofs)
        '''
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        '''所有的 env 的 所有 actor 的 indices，可以被用來作為 mask \ 
        維度是 (num_envs x num_actors_per_env), \ 
        ex: [[1 2 3][4 5 6]]
        ''' 
        
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.envs = []  # 原本是在 create_env 時被初始化與設定，現在配合 self.frankas 改到這裡來
        '''所有的 envs handle list \n 在 create_env() 中被進行設定 '''
        self.frankas = []
        '''所有 franka handle (actor 的 handle) \ 
        在 create_env() 中被進行設定 \ 
        (num_envs x num_agent), 第一個維度是env_idx, 第二個維度是該env第幾個arm，拿到的handle就是arm在該env中的actor idx
        '''

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        assert self.num_agents==2, "This task only support 2 agents!!"
        assert self.num_targets==2, "This task only support 2 targets!!"

        # modified 0122  add camera look at
        if self.viewer != None:
            cam_pos    = gymapi.Vec3(5.0, 5.0, 3.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )
        '''franka 手臂的預設 dof pos，維度是(9,)'''

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        # self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.cmd_limit = to_torch([0.6, 0.6, 0.6, 0.6, 0.6, 0.6], device=self.device).unsqueeze(0) if \
                            self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs*self.num_agents, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_stand_opts)

        # Create wall on the table
        # TODO: 建立桌上的中間隔板，高度要參數化 ！！！！！！！！！
        self.wall_height = 0.3
        wall_pos = [0., 0.3, table_pos[-1] + (table_thickness/2 + self.wall_height/2)]
        wall_asset = self.gym.create_box(self.sim, *[1.2, 0.05, self.wall_height], table_opts)
        self.wall_pos = wall_pos  # temp

        self.cubeA_size = 0.050
        # self.cubeB_size = 0.070

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_opts.disable_gravity = False
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        # cubeB_opts = gymapi.AssetOptions()
        # cubeB_opts.disable_gravity = True  # modified 0122  disable cube gravity
        # cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        # cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # print("num franka bodies: ", self.num_franka_bodies)
        # print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # Define start pose for wall
        wall_start_pose = gymapi.Transform()
        wall_start_pose.p = gymapi.Vec3(*wall_pos)
        wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # cubeB_start_pose = gymapi.Transform()
        # cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        # cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        # max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, cubeA, cubeB
        # max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cubeA, cubeB

        self.envs = []
        self.frankas = []

        # _cube_colid_mask = int(2**2 * ((2**self.num_agents)-1/(2-1)))  # this mask makes cube not colid with arms
        _cube_colid_mask = int(2)  # this mask make gripper can hold cube
        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # if self.aggregate_mode >= 3:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            # if self.aggregate_mode == 2:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_ids = []
            for cubeA_idx in range(1, self.num_targets+1):
                _cube_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, f"cubeA{cubeA_idx}", i, _cube_colid_mask, 0)  # modified 0122  change the franka's collision mask
                self._cubeA_ids.append(_cube_id)
                # Set colors
                self.gym.set_rigid_body_color(env_ptr, _cube_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            # if self.aggregate_mode == 1:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # if self.aggregate_mode > 0:
            #     self.gym.end_aggregate(env_ptr)
            
            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(list())  # this is handle list for every env

            # Create franka
            # Potentially randomize start pose
            # TODO: NOT IMPLEMENT YET
            if self.franka_position_noise > 0:
                raise NotImplementedError()
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                raise NotImplementedError()
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)

            # ====================== CREATE ARMS =====================================
            franka_start_pose = gymapi.Transform()
            franka_start_z = 1.0 + table_thickness / 2 + table_stand_height
            _p, _r = self._get_franka_start_poses_rots(r=0.45, offset=[0., -0.4])

            for franka_idx in range(1, self.num_agents+1):
                franka_start_pose.p = gymapi.Vec3(*_p[franka_idx-1], franka_start_z)
                franka_start_pose.r = gymapi.Quat(*_r[franka_idx-1])
                # franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, f"franka{franka_idx}", i, 3, 0)
                franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, f"franka{franka_idx}", i, 2**(franka_idx+1), 0)  # 00100, 01000, 10000 ...
                self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
                self.frankas[i].append(franka_actor)

            # Create wall
            wall_actor = self.gym.create_actor(env_ptr, wall_asset, wall_start_pose, "wall", i, 1, 0)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, self.num_targets, 13, device=self.device) # this reserved buffer for reset cube function
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]

        self.handles = {  # this is for RIGID BODY !!
            # Franka
            "base":              [self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle_i, "panda_link0") for franka_handle_i in self.frankas[0]],
            # TODO: 注意！！這邊的 handle 將不是用引用的，不能直接對應到原本值的變動！！！！ 待確認需求
            "hand":              [self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle_i, "panda_hand") for franka_handle_i in self.frankas[0]],
            "leftfinger_tip":    [self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle_i, "panda_leftfinger_tip") for franka_handle_i in self.frankas[0]],
            "rightfinger_tip":   [self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle_i, "panda_rightfinger_tip") for franka_handle_i in self.frankas[0]],
            "grip_site":         [self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle_i, "panda_grip_site") for franka_handle_i in self.frankas[0]],
            # Cubes
            "cubeA_body_handle": [self.gym.find_actor_rigid_body_handle(env_ptr, cubeA_idx, "box") for cubeA_idx in self._cubeA_ids],
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape: (n_actors of all envs, 13), 13 is position[3], rotation[4], linear velocity and angular velocity
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # shape: (n_dof, 2), 2 is position and velocity of the dof
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape: (n_rigid_body, 13)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)  # change into (n_envs, n_actors in the env, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)  # change into (n_envs, n_dof of the env, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13) # change into (n_envs, n_rigid_body of the env, 13)
        self._q = self._dof_state[..., 0]  # (n_envs, n_dof of the env)'s position
        self._qd = self._dof_state[..., 1]  # (n_envs, n_dof of the env)'s velocity

        # self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]   # NOTE: 這樣雖然可以拿到一開始的資料，但是這樣 不 是 reference 的方式，會讓後面追蹤不到 observation !!!
        _grip_start_idx = self.handles["grip_site"][0]
        _grip_step = self.handles["grip_site"][1] - self.handles["grip_site"][0]
        _grip_end_idx = _grip_start_idx + _grip_step * self.num_agents
        self._eef_state = self._rigid_body_state[:, _grip_start_idx:_grip_end_idx:_grip_step, :]

        _base_start_idx = self.handles["base"][0]
        _base_step = self.handles["base"][1] - self.handles["base"][0]
        _base_end_idx = _base_start_idx + _base_step * self.num_agents
        self._base_state = self._rigid_body_state[:, _base_start_idx:_base_end_idx:_base_step, :]  # 新增一組_base_state來追蹤姿態以供obs系統使用

        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]  # TODO: this not works!!
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :] # TODO: this not works!!
        
        # _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")  # (n_envs, joints, 6, 9)  # NOTE: deprecated, 改用 _get_j_eef()
        # jacobian = gymtorch.wrap_tensor(_jacobian)
        # hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']  # =7
        # self._j_eef = jacobian[:, hand_joint_index, :, :7]   # (num_envs x 6 x 7)  # 這邊有點問題 因為只有拿到第一agent的
        # _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")  # (num_envs x 9 x 9)  # NOTE: deprecated, 改用 _get_mm()
        # mm = gymtorch.wrap_tensor(_massmatrix)
        # self._mm = mm[:, :7, :7]

        _cubeA_start_idx = self.handles["cubeA_body_handle"][0]
        _cubeA_step = self.handles["cubeA_body_handle"][1] - self.handles["cubeA_body_handle"][0]
        _cubeA_end_idx = _cubeA_start_idx + _cubeA_step * self.num_targets
        self._cubeA_state = self._root_state[:, _cubeA_start_idx:_cubeA_end_idx:_cubeA_step, :]

        # TODO: fix the size of cube A size!!
        # Initialize states
        self.states.update({
            # "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,  # the author use ones_like is for the device
            "cubeA_size": torch.ones(self.num_envs, 1, device=self.device) * self.cubeA_size,              # 2 x 1 
            # "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        # these 2 properties are used to set the actor states in the simulation (in self.pre_physics_step())
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        # 選取0:0+7, 9:9+7 ... ，改成9個一排，選前7個。  維度變成 ((num_envs*num_agents) x 7)
        self._arm_control = self._effort_control.view(-1, 9)[:, :7]
        # 選取8:8+2, 17:17+2 ... ，改成9個一排，選後2個。  維度變成 ((num_envs*num_agents) x 2)
        self._gripper_control = self._pos_control.view(-1, 9)[:, -2:]
        
        # ===== prepare for campute the contact force =====
        _hand_start_idx = self.handles['hand'][0]
        _hand_step = self.handles['hand'][1] - self.handles['hand'][0]
        _hand_end_idx = _hand_start_idx + _hand_step * self.num_agents
        self._contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(*self._rigid_body_state.shape[:2], 3)
        # 原本是 (所有rigid_body數量 x 3)，依據 _rigid_body_state 的維度去調整成 (num_envs x num_rigid_per_env x 3)
        # 抓出手臂接觸力
        self._hands_contact_forces = self._contact_forces[:, _hand_start_idx:_hand_end_idx:_hand_step, :] 
        # 抓出目標接觸力
        self._cube_contact_forces = self._contact_forces[:, _cubeA_start_idx:_cubeA_end_idx:_cubeA_step, :]

        # Initialize indices
        # NOTE: 每次修改rigid或actor時，確保這個 global indice 維度與內容可以正確對應到所有 actors
        _num_actors = 2 + self.num_targets + self.num_agents + 1 # 一個 env 中有多少 actors, 2 is for table and table stand  # the last 1 is for wall
        self._global_indices = torch.arange((self.num_envs*_num_actors), dtype=torch.int32, device=self.device).view(self.num_envs, -1) # 0123 modified  remove cube so n_actors in a env is 4 now

    def compute_FSM(self):
        # only compute local FSM

        md = torch.norm(self.states["cubeA_pos_min_relative"], dim=-1)  # shape: [num_envs x num_agents]
        cubeA_size = self.cubeA_size  # this is not tensor, good
        table_height = 1.05
        
        # state 0 - approching
        state_machine = torch.zeros_like(md, dtype=torch.long)

        # state 1 - holding
        is_close_enough = md <= (cubeA_size * 0.5 * 0.9)
        state_machine = torch.where(is_close_enough, 1, state_machine)
        
        # === only lifting ===
        # # state 2 - lifting
        # is_gripper_closed = self.actions[:, -1].view(-1, self.num_agents) < 0 if  self.actions is not None else torch.zeros_like(md, dtype=torch.bool)
        # state_machine = torch.where(is_close_enough & is_gripper_closed, 2, state_machine)
        # # state 3 - GOAL
        # is_high_enough = (self.states["eef_pos"][..., -1]-table_height) > (self.wall_height + cubeA_size/2)
        # state_machine = torch.where(is_close_enough & is_gripper_closed & is_high_enough, 3, state_machine)

        # # === 5 states method ===
        # # state 2 - moving to target line
        # is_gripper_closed = self.actions[:, -1].view(-1, self.num_agents) < 0 if  self.actions is not None else torch.zeros_like(md, dtype=torch.bool)
        # state_machine = torch.where(is_close_enough & is_gripper_closed, 2, state_machine)
        # # state 3 - releasing
        # target_line = torch.tensor([self.wall_pos[1]+cubeA_size*1.5, self.wall_height+table_height+cubeA_size], device=self.device) # y and z
        # d_target_line = torch.norm(self.states["eef_pos"][..., -2:] - target_line, dim=-1)
        # is_reach_target_area = d_target_line <= cubeA_size
        # state_machine = torch.where(is_close_enough & is_gripper_closed & is_reach_target_area, 3, state_machine)
        # # state 4 - GOAL
        # is_gripper_opened = ~is_gripper_closed
        # is_inside = self.states["eef_pos"][..., 1] >= self.wall_pos[1]
        # state_machine = torch.where(is_gripper_opened & is_inside, 4, state_machine)  # no no no this is so weird.......
        
        holding_cube_pos = self.states["nearest_cubeA_pos"]  # envs x agents x 3

        # === 7 states method ===
        # state 2 - lifting
        is_gripper_closed = self.actions[:, -1].view(-1, self.num_agents) < 0 if  self.actions is not None else torch.zeros_like(md, dtype=torch.bool)
        is_holding = is_close_enough & is_gripper_closed
        state_machine = torch.where(is_holding, 2, state_machine)
        # state 3 - moving
        is_high_enough = (holding_cube_pos[..., -1]-table_height) > (self.wall_height + cubeA_size/2)
        state_machine = torch.where(is_holding & is_high_enough, 3, state_machine)
        # state 4 - descent
        target_y = (self.wall_pos[1]+cubeA_size*1.)  # only Y
        target_x_border = 0.6
        is_in_target_area = (holding_cube_pos[..., 1] > target_y) & (abs(holding_cube_pos[..., 0]) < target_x_border)
        state_machine = torch.where(is_holding & is_in_target_area, 4, state_machine)
        # state 5 - releasing
        is_low_enough = (holding_cube_pos[..., 2]-table_height < self.wall_height/2.)
        state_machine = torch.where(is_holding & is_in_target_area & is_low_enough, 5, state_machine)
        # state 6 - GOAL
        is_gripper_opened = ~is_gripper_closed
        state_machine = torch.where((is_holding & is_in_target_area & is_low_enough) & is_gripper_opened, 6, state_machine)  # no no no this is so weird.......

        return state_machine  # shape: [num_envs x num_agents]
    
    def compute_global_FSM(self):
        FSM = self.states["FSM"]  # shape: [num_envs x num_agents]
        # state 0 - no holding
        gFSM = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # shape: [num_envs,]

        # state 1 - any holding
        gFSM = torch.where(torch.any(FSM, dim=-1), 1, gFSM)

        # state 2 - all holding
        gFSM = torch.where(torch.all(FSM>=1, dim=-1), 2, gFSM)

        # state 3 - only holding and lifting
        gFSM = torch.where(torch.all(FSM>=2, dim=-1), 3, gFSM)

        # state 4 - all lifting
        gFSM = torch.where(torch.all(FSM>=3, dim=-1), 4, gFSM)
        
        # state 5
        gFSM = torch.where(torch.all(FSM>=4, dim=-1), 5, gFSM)

        # state 6
        gFSM = torch.where(torch.all(FSM>=5, dim=-1), 6, gFSM)

        # state 7
        gFSM = torch.where(torch.all(FSM>=6, dim=-1), 7, gFSM)

        return gFSM.repeat_interleave(2)  # shape: [num_envs,] -> [num_envs * num_agents]

    # TODO: 注意這邊有些維度是 ((num_envs*num_agents) x m); 有些是(num_envs x num_agents x m)，要確認是否需要在這邊統一或是自行使用 .view() 修改
    def _update_states(self):
        """update the self.state. Be called in ._refresh()"""
        def _get_min_relative_pos():
            '''回傳: 距離最小目標相對向量, 距離最小目標位置, 距離最小目標id \ 
            shape: [envs x agents x 3], [envs x agents x 3], [envs x agents]
            '''
            # self._cubeA_state[:, :, :3] - self._eef_state[:, :, :3]
            c = self._cubeA_state[:, :, :3]
            a = self._eef_state[:, :, :3]
            relative_pos = (c.unsqueeze(1) - a.unsqueeze(2))  # [env, a, c, 3]
            relative_norm = relative_pos.norm(dim=-1)  # [env, a, c]
            min_distance_ids = torch.argmin(relative_norm, dim=-1)  # [env, a]
            env_ids = torch.arange(min_distance_ids.shape[0]).unsqueeze(-1).expand(min_distance_ids.shape)
            nc = c[env_ids, min_distance_ids]
            return (nc - a), nc, min_distance_ids  # 相對向量, 最近目標之位置, 最近目標之id

        cubeA_pos_min_relative, nearest_cubeA_pos, nearest_cubeA_ids = _get_min_relative_pos()
        self.states.update({
            # dof positions
            "q":         self._q[:, :],    # 所有關節位置(dof.pos)
            "q_gripper": self._q[:, -2:],  # 夾爪關節位置(dof.pos)
            # eef 姿態
            "eef_pos":  self._eef_state[:, :, :3],   # 維度：(num_envs x num_agents x 3)
            "eef_quat": self._eef_state[:, :, 3:7],  # 維度：(num_envs x num_agents x 4)
            "eef_vel":  self._eef_state[:, :, 7:],   # 維度：(num_envs x num_agents x 6)
            "eef_lf_pos": self._eef_lf_state[:, :, :3],  # NOT IMPLEMENT YET
            "eef_rf_pos": self._eef_rf_state[:, :, :3],  # NOT IMPLEMENT YET
            # cube 姿態
            "cubeA_quat": self._cubeA_state[:, :, 3:7],  # 維度：(num_envs x num_targets x 4)
            "cubeA_pos":  self._cubeA_state[:, :, 0:3],  # 維度：(num_envs x num_targets x 3)
            # cube 與 eef 相對位置
            # "cubeA_pos_relative": self._cubeA_state[:, 0, :3].unsqueeze(1).repeat_interleave(self.num_agents, dim=1) - self._eef_state[:, :, :3], # target is only the first cube # TODO: THIS IS ONLY FOR TESTING
            "cubeA_pos_relative": self._cubeA_state[:, :, :3] - self._eef_state[:, :, :3], # each one have one target  # comment this if num_targets > num_agents
            #                              2 x (2 x 3)  -  2 x (2 x 3)
            "cubeA_pos_min_relative": cubeA_pos_min_relative, # min dist of agent to random one target
            "nearest_cubeA_pos": nearest_cubeA_pos,
            "nearest_cubeA_ids": nearest_cubeA_ids, # min dist of agent to random one target
            # 新增一個基座姿態
            "base_pos":  self._base_state[:, :, :3],  # 維度：(num_envs x num_agents x 3)
            "base_quat": self._base_state[:, :, 3:7]  # 維度：(num_envs x num_agents x 3)
        })
        FSM = self.compute_FSM()
        self.states.update({
            "FSM": FSM,
            "FSM_p": torch.pow(2.0, FSM),
        })
        self.states.update({ "gFSM": self.compute_global_FSM() })
        pass
    def _refresh(self):
        """to update the self.state dict for compute_observations(). Be called in .__init__() and .compute_observation() """
        # 更新 sim 中的資訊，以便後面讀取
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        '''計算 reward 並儲存至 rew_buf 與 reset_buf '''
        self.rew_buf[:], self.reset_buf[:], self.reward_components = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self._hands_contact_forces, self.reward_settings, self.max_episode_length
        )

        self.extras.update( self.reward_components )

    # TODO: refector this spaghetti flow
    def compute_observations(self):
        '''整理出要交給 NN 進行推論的 obs_buf'''
        self._refresh()
        # 目前一共分成3大項目 obs_all_target_pos, obs_self, obs_all_other_eef_pos, obs_FSM
        # === prepare obs of all targets ===   
        obs_all_target_pos = self.states["cubeA_pos"].reshape(self.num_envs, -1).repeat_interleave(self.num_agents, dim=0)  # shape: ->((self.num_envs*self.num_agents) x (self.num_targets*3))
        
        # === prepare obs of agent itself ===
        _obs_self_names = ["eef_quat", "eef_pos", "cubeA_pos_min_relative", "base_pos", "base_quat"]   # NOTE: add base pose is important?
        obs_self = torch.cat([self.states[ob].contiguous().view(self.num_envs * self.num_agents, -1) for ob in _obs_self_names], dim=-1)
        
        # === prepare obs of all other agents eef pos ===
        _unshifted = self.states["eef_pos"].reshape(self.num_envs, -1)
        _shifted = []
        for i in range(self.num_agents):
            _shifted.append( torch.cat((_unshifted[..., i*3:], _unshifted[..., :i*3]), dim=1) )
        obs_all_other_eef_pos = torch.stack(_shifted, dim=1)[..., 3:].view(self.num_envs * self.num_agents, -1)

        # === prepareobs of FSM ===
        obs_FSM = torch.cat([
            self.states["FSM"].view(self.num_envs * self.num_agents, -1), 
            self.states["FSM"].view(self.num_envs * self.num_agents, -1)
        ], dim=-1)

        self.obs_buf = torch.cat([obs_all_target_pos, obs_self, obs_all_other_eef_pos, obs_FSM], dim=-1)

        return self.obs_buf

    # NOTE: 因應大多數 buffer 都改成 num_envs * num_agents 的大小，故我們將 reset_idx 接收的參數也改成 agent_ids
    # TODO: 這邊改動很多，目前流程很亂，待整理！！
    def reset_idx(self, agent_ids):
        '''依據所收到的 agent 的 ids 去重製對應的 env \ 
        注意：env_ids 是所需要處理的 env 的 ids, 這邊接收的是與 reset_buf 和 timeout_buf 格式的 agent_ids \ 
        this will be called in .post_physical_step() and init()'''
        # env_ids_int32 = env_ids.to(dtype=torch.int32)  # this line is  deprecated by author
        env_ids = self._agent_ids_to_env_ids(agent_ids, use_AND_filter=True) # 這邊有做 AND 的過濾

        # 以下進入重置任務目標方塊的流程
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        # 以下進入重置 agent 狀態的流程
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)  # 9 是因為 franka arm 有 9 個 dof
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)  # 為何這邊沒有 .unsequeeze() ???

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]  # 不對夾爪加噪音

        pos = torch.cat([pos] * self.num_agents , dim=-1) # 將 初始 franka 姿態做成可以對應到multi-agent的

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos    # NOTE: # (n_envs x (num_agents*9)
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])  # this is velocities of dofs

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)  # actor 的受力重設為0


        # TODO: 這邊是有關 global 問題的，待重新並易並修正
        agent_actor_ids = self.frankas[0]
        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, :][:, agent_actor_ids].flatten()  # 0 是指拿到第一個 actor 也就是 第一個 franka  
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, :][:, self._cubeA_ids].flatten()  # TODO: malfunction in multi-arm version

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),  # 設定 dof_state 是因為剛剛有修改 _q 和 _qd (partial view of dof_state)
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        # multi_env_ids_cubes_int32 = self._global_indices[env_ids, 3].flatten()  # TODO: malfunction in multi-arm version
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),  # 這邊是為了設定 cube(actor) 的 state
                                                     gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), 
                                                     len(multi_env_ids_cubes_int32))
        
        new_agent_ids = self._env_ids_to_agent_ids(env_ids)  # env_ids 是指定要 reset 的 env，這邊轉換成 agent的ids
        self.progress_buf[new_agent_ids] = 0
        self.reset_buf[new_agent_ids] = 0

    def _reset_init_cube_state(self, cube=None, env_ids=None, check_valid=True):
        """注意：尚未檢查 target 是否重疊！！""" # TODO: implement the check_valid function!!
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        num_envs_to_reset = len(env_ids)

        sampled_cube_state = torch.zeros(num_envs_to_reset, self.num_targets, 13, device=self.device)
        # self._init_cubeA_state
        cube_heights = self.states["cubeA_size"] # TODO: 不一定需要這個參數做調整，待確認需求

        rand_z_range = 0.2
        cube_rand_z = torch.rand(sampled_cube_state[:, :, 2].shape, device=self.device) * rand_z_range
        sampled_cube_state[:, :, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2  + cube_rand_z

        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        sampled_cube_state[:, :, 0:2] = centered_cube_xy_state.unsqueeze(0) + 2.0 * self.start_position_noise * (torch.rand(num_envs_to_reset, self.num_targets, 2, device=self.device) - 0.5)

        if self.start_rotation_noise > 0:
            # TODO: implement rotation
            pass

        sampled_cube_state[:, :, 6] = 1.0  # set r.w=1
        self._init_cubeA_state[env_ids, :, :] = sampled_cube_state
        return
        
        ################# OLD VERSION ##################
        check_valid = False  # not need check_valid anymore

        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        # TODO: make sure the size of env_ids is <= 2
        # TODO: 如果將來將 env_ids 改為更大的維度，這邊需要改變讀取 env_ids 對應到 cube 的規則

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            # this_cube_state_all = self._init_cubeB_state
            # other_cube_state = self._init_cubeA_state[env_ids, :]
            # cube_heights = self.states["cubeA_size"]
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        # min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        # min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        # modified 0122  add random z for each cube
        rand_z_range = 0.5
        cube_rand_z = torch.rand(sampled_cube_state[:, 2].shape, device=self.device) * rand_z_range
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2  + cube_rand_z

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            pass
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)
            # sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0)  # NOTE: THIS LINE IS FOR TESTING

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        """transform function, output size is (n, 7) no matter the dpose size"""
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/

        _mm = self._get_mm()
        _j_eef = self._get_j_eef()

        # q, qd = self._q[:, :7], self._qd[:, :7]
        q, qd = self._q.view(self.num_envs*self.num_agents, -1)[:, :7], self._qd.view(self.num_envs*self.num_agents, -1)[:, :7]
        mm_inv = torch.inverse(_mm)
        m_eef_inv = _j_eef @ mm_inv @ torch.transpose(_j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        eef_vel = self.states["eef_vel"].reshape(self.num_envs*self.num_agents, -1)  # reshape from 2 x 2 x -1 to 4 x -1

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(_j_eef, 1, 2) @ m_eef @ (self.kp * dpose - self.kd * eef_vel).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ _j_eef @ mm_inv
        
        u_null = self.kd_null * -qd + self.kp_null * ((self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = _mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(_j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        """do preprocess of the actions from actor and send them into the simulation"""
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        u_arm, u_gripper = self.actions[:, :6], self.actions[:, 6]  # shape: (n_envs, 6) u_arn size is the same

        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)  # NOTE: this is to transform two poses to torques
            pass

        self._arm_control[:, :6] = u_arm[:, :6]

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))     # this is for gripper
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))  # this is for arm (OSC)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"][:, 0, :]  # TODO: 現在先用第一手臂做測試，待修正
            eef_rot = self.states["eef_quat"][:, 0, :]
            cubeA_pos = self.states["cubeA_pos"][:, 0, :]
            cubeA_rot = self.states["cubeA_quat"][:, 0, :]
            # cubeB_pos = self.states["cubeB_pos"]
            # cubeB_rot = self.states["cubeB_quat"]
        
            nc = self.states["nearest_cubeA_ids"]
            env_ids = torch.arange(nc.shape[0]).unsqueeze(-1).expand(nc.shape)
            mask = torch.zeros_like(nc)
            mask[env_ids, nc] = 1
            is_all_target = mask.sum(dim=-1) / self.num_agents

            # Plot visualizations
            for i in range(self.num_envs):
                # for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                for pos, rot in zip((eef_pos, cubeA_pos), (eef_rot, cubeA_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
                for agent_idx in range(self.num_agents):
                    from_p = self.states["eef_pos"][i, agent_idx].cpu().numpy()
                    to_p = from_p + self.states["cubeA_pos_min_relative"][i, agent_idx].cpu().numpy()
                    # self.gym.add_lines(self.viewer, self.envs[i], 1, [*from_p, *to_p], [0.85, 0.1, 0.85])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [*from_p, *to_p], [0.85, 0.1, 0.85])
                
                lp = torch.tensor([0.3+0.05*1.5, 0.3+1.05+0.05]).cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [-0.6, lp[0], lp[1], 0.6, lp[0], lp[1]], [0.85, 0.1, 0.85])
            
            target_y = (0.3+0.05*1.5)  # only y
            target_x_border = (0.6)
            d_target_y = torch.abs(self.states["nearest_cubeA_pos"][..., 1] - target_y)
            moving_reward = torch.exp(-5. * (d_target_y**2))
            
            target_z = 0.3 / 2. + 1.05
            d_target_z = torch.abs(self.states["nearest_cubeA_pos"][..., 2] - target_z)
            descent_reward = torch.exp(-5. * (d_target_z**2))

            if self.viewer: 
                FSM = self.states["FSM"][1]  # only env1
                # print(FSM)
                # print( moving_reward[1] * (FSM==3) )
                print(descent_reward[1])

        
    # deprecated   this is not so useful
    def value_viz_rigid_body_color(self, env, actor, rigid_idx, value=0., min_max=(0., 1.)):
        r_value = (value-min_max[0]) / (min_max[1] - min_max[0]) + min_max[0]
        r_value = 1.0 if r_value > 1.0 else r_value
        color = gymapi.Vec3(0.3+r_value, 0.1, 0.1)
        self.gym.set_rigid_body_color(env, actor, rigid_idx, gymapi.MESH_VISUAL, color)

    def _agent_ids_to_env_ids(self, agent_ids, use_AND_filter=True):
        '''輸入[0 1 2 3] 輸出 [0 1], 輸入 [0 3] 輸出 [], 當 num_agents=2, 且use_AND_filter=True ; \ 
           輸入 [0 3] 輸出 [0 1], 當 num_agents=2, 且use_AND_filter=False
        '''
        env_ids = agent_ids // self.num_agents
        if use_AND_filter:
            env_ids = env_ids.bincount()
            env_ids = torch.arange(env_ids.numel())[env_ids >= self.num_agents]
        else:
            env_ids = env_ids.unique()
        return env_ids
        
    def _env_ids_to_agent_ids(self, env_ids):
        '''輸入[0 2] 輸出 [[0 1], [4 5]], 當num_agents=2'''
        return torch.arange(self.num_envs*self.num_agents).view(self.num_envs, self.num_agents)[env_ids, :].flatten()
    
    def _get_j_eef(self):
        j_eef_all_agents = []
        for i in range(1, self.num_agents+1):
            _jacobian = self.gym.acquire_jacobian_tensor(self.sim, f"franka{i}")
            jacobian = gymtorch.wrap_tensor(_jacobian)
            hand_joint_index = self.gym.get_actor_joint_dict(self.envs[0], self.frankas[0][0])['panda_hand_joint']  # =7
            j_eef = jacobian[:, hand_joint_index, :, :7]
            if j_eef.sum() == 0:
                self.gym.refresh_jacobian_tensors(self.sim)  # 這邊很重要，在剛開始運作時會出現尚未refresh就使用導致inverse()失敗
            j_eef_all_agents.append(j_eef)
        return torch.stack(j_eef_all_agents, dim=1).view(self.num_envs*self.num_agents, 6, 7)

    def _get_mm(self):
        mm_all_agents = []
        for i in range(1, self.num_agents+1):
            _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, f"franka{i}")
            mm = gymtorch.wrap_tensor(_massmatrix)
            if mm.sum() == 0:
                self.gym.refresh_mass_matrix_tensors(self.sim)
            mm_all_agents.append( mm[:, :7, :7] )
        return torch.stack(mm_all_agents, dim=1).view(self.num_envs*self.num_agents, 7, 7)
    def _get_franka_start_poses_rots(self, r, offset=None):
        '''輸入分佈的圓之半徑, 輸出agents的位置和繞z軸旋轉四元素。 offset參數可以控制便偏移量'''
        rads = torch.deg2rad(
            torch.tensor(range(0, 359, 360//self.num_agents), dtype=torch.float)
        )
        if offset is None:
            offset = [0., 0.]

        return (torch.stack([-torch.cos(rads)+offset[0], torch.sin(rads)+offset[1]], dim=-1) * r), \
            (torch.stack([torch.zeros(rads.shape[-1]), torch.zeros(rads.shape[-1]), torch.sin(-rads/2), torch.cos(-rads/2)], dim=-1))
        



#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, hands_contact_forces, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Dict[str, float], float) -> Tuple[Tensor, Tensor, Dict[str,Tensor]]
    reward_components = {}
    md = torch.norm(states["cubeA_pos_min_relative"].view(-1, 3), dim=-1)
    FSM = states["FSM"].flatten()  # shape: [envs x agents] -> [envs*agents]
    gFSM = states["gFSM"]
    rewards = torch.zeros_like(md)

    # state 0 - approaching
    approaching_reward = torch.exp(-5. * (md**2))
    state0_reward = torch.where(FSM == 0, approaching_reward, torch.zeros_like(md))

    # state 1 - gripper closing
    # is_gripper_closed = actions[:, -1] < 0 if  actions is not None else torch.zeros_like(md, dtype=torch.bool)
    gripper_closing_reward = torch.exp(-1. * actions[:, -1])  # + 1  # WHY ??? 
    state1_reward = torch.where(FSM == 1, gripper_closing_reward, torch.zeros_like(md))

    # === only lifting ===
    # # state 2 - lifting
    # eef_height = (states["eef_pos"][..., -1] - 1.05).flatten()
    # lifting_reward = eef_height / 0.27
    # state2_reward = torch.where(FSM == 2, lifting_reward, torch.zeros_like(md))
    # # state 3 - GOAL
    # state3_reward = torch.where(FSM ==3, torch.ones_like(md) * 2., torch.zeros_like(md))

    # # === the new 5 states method ===
    # # state 2 - moving
    # target_line = torch.tensor([0.3+0.05*1.5, 0.3+1.05+0.05], device=md.device) # y and z
    # d_target_line = torch.norm(states["eef_pos"][..., -2:] - target_line, dim=-1).flatten()
    # moving_reward = torch.exp(-5. * (d_target_line**2))
    # state2_reward = torch.where(FSM == 2, moving_reward, torch.zeros_like(md))
    # # state 3 - releasing
    # gripper_opening_reward = torch.exp(4 * actions[:, -1])
    # state3_reward = torch.where(FSM == 3, gripper_opening_reward, torch.zeros_like(md))
    # # state 4 - GOAL
    # state4_reward = torch.where(FSM == 4, torch.ones_like(md) * 2, torch.zeros_like(md))

    # === the new 6 states method ===
    # state 2 - lifting
    eef_height = (states["nearest_cubeA_pos"][..., -1] - 1.05).flatten()
    lifting_reward = eef_height / (0.3+0.025)
    state2_reward = torch.where(FSM == 2, lifting_reward, torch.zeros_like(md))
    # state 3 - moving
    target_y = (0.3+0.05*2.)  # only y
    target_x_border = (0.6)
    d_target_y = torch.abs(states["nearest_cubeA_pos"][..., 1] - target_y).flatten()
    moving_reward = torch.exp(-5. * (d_target_y**2))
    state3_reward = torch.where(FSM == 3, moving_reward, torch.zeros_like(md))
    # state 4 - descent
    target_z = 0.3 / 2. + 1.05
    d_target_z = torch.abs(states["nearest_cubeA_pos"][..., 2] - target_z).flatten()
    descent_reward = torch.exp(-5. * (d_target_z**2))
    state4_reward = torch.where(FSM == 4, descent_reward, torch.zeros_like(md))
    # state 5 - release
    gripper_opening_reward = torch.exp(4 * actions[:, -1])
    state5_reward = torch.where(FSM == 5, gripper_opening_reward, torch.zeros_like(md))
    # state 6 - GOAL
    state6_reward = torch.where(FSM == 6, torch.ones_like(md) * 3, torch.zeros_like(md))


    # BSR
    BSR = FSM.to(torch.float)
    # gBSR = gFSM.to(torch.float)
    gBSR = torch.zeros_like(md)

    # # side rules
    # collision_punishment = (torch.norm(hands_contact_forces.reshape(-1, 3), dim=-1) >= 0.1) * -3.
    # side_rule_reward = collision_punishment
    side_rule_reward = torch.zeros_like(md)

    # sum up 
    rewards = state0_reward + state1_reward + state2_reward + state3_reward + state4_reward + state5_reward + state6_reward +   BSR + gBSR + side_rule_reward

    rewards = torch.clip(rewards, 0., None)

    # for log
    reward_components["r/state0"] = state0_reward.mean()
    reward_components["r/state1"] = state1_reward.mean()
    reward_components["r/state2"] = state2_reward.mean()
    reward_components["r/state3"] = state3_reward.mean()
    reward_components["r/state4"] = state4_reward.mean()
    reward_components["r/state5"] = state5_reward.mean()
    reward_components["r/state6"] = state6_reward.mean()
    reward_components["r/BSR"] = BSR.mean()
    reward_components["r/gBSR"] = gBSR.mean()

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, reward_components