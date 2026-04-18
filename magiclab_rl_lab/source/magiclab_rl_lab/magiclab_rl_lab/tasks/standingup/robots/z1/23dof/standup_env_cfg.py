import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from magiclab_rl_lab.assets.robots.magiclab import MAGICLAB_Z1_23DOF_CFG as ROBOT_CFG
from magiclab_rl_lab.tasks.standingup import mdp

Z1_23DOF_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
]

STANDUP_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=15.0,
    num_rows=3,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class Z1StandupSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STANDUP_FLAT_TERRAIN_CFG,
        max_init_terrain_level=STANDUP_FLAT_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class Z1StandupCommandsCfg:
    """Keep velocity commands as near-zero to preserve manager compatibility."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.2),
            lin_vel_y=(0.0, 0.2),
            ang_vel_z=(0.0, 0.2),
        ),
    )


@configclass
class Z1StandupActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=Z1_23DOF_JOINTS,
        scale=0.3,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class Z1StandupObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Z1_23DOF_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Z1_23DOF_JOINTS, preserve_order=True)},
            scale=0.05,
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        progress = ObsTerm(func=mdp.normalized_episode_progress)

        def __post_init__(self):
            self.history_length = 6
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Z1_23DOF_JOINTS, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Z1_23DOF_JOINTS, preserve_order=True)},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        progress = ObsTerm(func=mdp.normalized_episode_progress)

        def __post_init__(self):
            self.history_length = 6
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Z1StandupRewardsCfg:
    upright = RewTerm(func=mdp.upright_reward, weight=2.5)
    head_height = RewTerm(func=mdp.head_height_reward, weight=2.0)
    is_alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # regularization
    action_rate = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.02)
    joint_power = RewTerm(func=mdp.joint_power_penalty, weight=-2.5e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)


@configclass
class Z1StandupEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (-0.5, 0.5)},
    )
    reset_root = EventTerm(
        func=mdp.reset_root_pose_fallen,
        mode="reset",
        params={"roll_range": (-math.pi, math.pi), "pitch_range": (-math.pi, math.pi), "yaw_range": (-math.pi, math.pi)},
    )


@configclass
class Z1StandupTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    head_low = DoneTerm(func=mdp.head_below_minimum, params={"minimum_height": 0.15})


@configclass
class Z1StandupCurriculumCfg:
    # This term is currently used as a signal for tuning schedule updates.
    standup_success = CurrTerm(func=mdp.standup_success_rate)


@configclass
class Z1StandupGroundEnvCfg(ManagerBasedRLEnvCfg):
    scene: Z1StandupSceneCfg = Z1StandupSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: Z1StandupObservationsCfg = Z1StandupObservationsCfg()
    actions: Z1StandupActionsCfg = Z1StandupActionsCfg()
    commands: Z1StandupCommandsCfg = Z1StandupCommandsCfg()
    rewards: Z1StandupRewardsCfg = Z1StandupRewardsCfg()
    terminations: Z1StandupTerminationsCfg = Z1StandupTerminationsCfg()
    events: Z1StandupEventCfg = Z1StandupEventCfg()
    curriculum: Z1StandupCurriculumCfg = Z1StandupCurriculumCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Z1StandupGroundPlayEnvCfg(Z1StandupGroundEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class Z1StandupProneEnvCfg(Z1StandupGroundEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_root.params["roll_range"] = (0.0, 0.0)
        self.events.reset_root.params["pitch_range"] = (math.pi - 0.15, math.pi + 0.15)
        self.events.reset_root.params["yaw_range"] = (-math.pi, math.pi)


@configclass
class Z1StandupPronePlayEnvCfg(Z1StandupProneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class Z1StandupPlatformEnvCfg(Z1StandupGroundEnvCfg):
    """Placeholder terrain profile for platform stand-up.

    Kept flat in v1 migration to avoid mutating existing terrain pipeline.
    """


@configclass
class Z1StandupPlatformPlayEnvCfg(Z1StandupPlatformEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class Z1StandupSlopeEnvCfg(Z1StandupGroundEnvCfg):
    """Placeholder terrain profile for slope stand-up.

    Kept flat in v1 migration; swap to slope generator in follow-up iteration.
    """


@configclass
class Z1StandupSlopePlayEnvCfg(Z1StandupSlopeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class Z1StandupWallEnvCfg(Z1StandupGroundEnvCfg):
    """Placeholder terrain profile for wall stand-up.

    Kept flat in v1 migration; wall obstacle asset can be added as scene object later.
    """


@configclass
class Z1StandupWallPlayEnvCfg(Z1StandupWallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32

