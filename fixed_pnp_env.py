import argparse
import sys
import os
import numpy as np

# Path setup
ISAAC_GROOT_ROOT = os.path.expanduser("~/workspace/Isaac-GR00T")
ROBOCASA_ROOT = os.path.join(ISAAC_GROOT_ROOT, "external_dependencies/robocasa")

sys.path.insert(0, ISAAC_GROOT_ROOT)
sys.path.insert(0, ROBOCASA_ROOT)

from robocasa.environments.kitchen.kitchen import *


# Environment Definition

class FixedPnP(Kitchen):

    FIXED_LAYOUT = [(5, 1)]  # U_SHAPED_SMALL + SCANDANAVIAN

    def __init__(
        self,
        obj_a_group="banana",
        obj_b_group="bowl",
        obj_a_x=-0.4,
        obj_a_y=0.4,
        obj_b_x=0.4,
        obj_b_y=0.4,
        *args,
        **kwargs,
    ):
        self.obj_a_group = obj_a_group
        self.obj_b_group = obj_b_group
        self.obj_a_x = obj_a_x
        self.obj_a_y = obj_a_y
        self.obj_b_x = obj_b_x
        self.obj_b_y = obj_b_y
        kwargs.setdefault("render_camera", "robot0_robotview")
        super().__init__(*args, **kwargs)

        self.layout_and_style_ids = self.FIXED_LAYOUT # Override layout to enforce [5, 1]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="obj_a")
        container_lang = self.get_obj_lang(obj_name="obj_b")
        ep_meta["lang"] = (
            f"pick the {obj_lang} from the counter and place it in the {container_lang}"
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj_a",
            obj_groups=self.obj_a_group,
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                size=(0.30, 0.40),
                pos=("ref", -1.0),
            ),
        ))
        cfgs.append(dict(
            name="obj_b",
            obj_groups=self.obj_b_group,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                size=(0.30, 0.40),
                pos=("ref", -1.0),
            ),
        ))
        return cfgs

    def _reset_internal(self):
        super()._reset_internal()

        base = self.robots[0].base_pos

        jnt_a = self.sim.model.body_jntadr[self.sim.model.body_name2id("obj_a_main")]
        jnt_b = self.sim.model.body_jntadr[self.sim.model.body_name2id("obj_b_main")]
        adr_a = self.sim.model.jnt_qposadr[jnt_a]
        adr_b = self.sim.model.jnt_qposadr[jnt_b]

        z_a = self.sim.data.qpos[adr_a + 2]
        z_b = self.sim.data.qpos[adr_b + 2]

        self.sim.data.qpos[adr_a:adr_a+3] = [base[0] + self.obj_a_x, base[1] + self.obj_a_y, z_a]
        self.sim.data.qpos[adr_b:adr_b+3] = [base[0] + self.obj_b_x, base[1] + self.obj_b_y, z_b]
        self.sim.forward()

    def _check_success(self):
        obj_in_receptacle = OU.check_obj_in_receptacle(self, "obj_a", "obj_b")
        gripper_far = OU.gripper_obj_far(self, obj_name="obj_a")
        return obj_in_receptacle and gripper_far


# Register Environment

def register_and_get_env_name(
    obj_a_group="banana",
    obj_b_group="bowl",
    obj_a_x=-0.4,
    obj_a_y=0.4,
    obj_b_x=0.4,
    obj_b_y=0.4,
):
    from robosuite.environments.base import REGISTERED_ENVS
    from robocasa.utils.gym_utils.gymnasium_groot import create_grootrobocasa_env_class

    class FixedPnPInstance(FixedPnP):
        def __init__(self, *args, **kwargs):
            super().__init__(
                obj_a_group=obj_a_group,
                obj_b_group=obj_b_group,
                obj_a_x=obj_a_x,
                obj_a_y=obj_a_y,
                obj_b_x=obj_b_x,
                obj_b_y=obj_b_y,
                *args, **kwargs
            )
    FixedPnPInstance.__name__ = "FixedPnPInstance"

    REGISTERED_ENVS["FixedPnP"] = FixedPnPInstance
    create_grootrobocasa_env_class("FixedPnP", "PandaOmron", "panda_omron")

    return "robocasa_panda_omron/FixedPnP_PandaOmron_Env"


# Preview Mode

def run_preview(args):
    from robosuite.devices import Keyboard
    from robocasa.scripts.collect_demos import collect_human_trajectory

    print(f"Launching preview: obj_a={args.obj_a}, obj_b={args.obj_b}, seed={args.seed}")
    print(f"  obj_a: ({args.obj_a_x}, {args.obj_a_y}) relative to robot base")
    print(f"  obj_b: ({args.obj_b_x}, {args.obj_b_y}) relative to robot base")

    env = FixedPnP(
        obj_a_group=args.obj_a,
        obj_b_group=args.obj_b,
        obj_a_x=args.obj_a_x,
        obj_a_y=args.obj_a_y,
        obj_b_x=args.obj_b_x,
        obj_b_y=args.obj_b_y,
        robots="PandaOmron",
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
        seed=args.seed,
    )

    device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    obs = env.reset()

    ep_meta = env.get_ep_meta()
    print(f"\nTask: {ep_meta['lang']}")
    print("\nControls:")
    print("  Mouse drag   — rotate view")
    print("  Arrow keys   — move robot horizontally")
    print("  . / ;        — move robot vertically")
    print("  Spacebar     — toggle gripper")
    print("  Ctrl+Q       — reset / exit")

    collect_human_trajectory(
        env, device, "right", "single-arm-opposed",
        mirror_actions=True, render=False, max_fr=30
    )
    env.close()


# Rollout with GR00T Policy

def run_episodes(args):
    from gr00t.policy.server_client import PolicyClient
    from gr00t.eval.rollout_policy import run_rollout_gymnasium_policy, WrapperConfigs
    from gr00t.eval.rollout_policy import VideoConfig, MultiStepConfig

    print(f"Connecting to GR00T server at {args.host}:{args.port}...")
    policy = PolicyClient(host=args.host, port=args.port)
    if not policy.ping():
        print("ERROR: Could not connect. Is the Colab server running?")
        sys.exit(1)
    print("Connected!")

    dist_a = (args.obj_a_x**2 + args.obj_a_y**2)**0.5
    dist_b = (args.obj_b_x**2 + args.obj_b_y**2)**0.5
    dist_ab = ((args.obj_a_x - args.obj_b_x)**2 + (args.obj_a_y - args.obj_b_y)**2)**0.5

    print(f"\nEnvironment config:")
    print(f"  Object A: {args.obj_a} at ({args.obj_a_x}, {args.obj_a_y}) — dist from robot: {dist_a:.2f}m")
    print(f"  Object B: {args.obj_b} at ({args.obj_b_x}, {args.obj_b_y}) — dist from robot: {dist_b:.2f}m")
    print(f"  Distance A↔B: {dist_ab:.2f}m")
    print(f"  Seed: {args.seed}")

    os.environ["MUJOCO_GL"] = "egl"
    env_name = register_and_get_env_name(
        obj_a_group=args.obj_a,
        obj_b_group=args.obj_b,
        obj_a_x=args.obj_a_x,
        obj_a_y=args.obj_a_y,
        obj_b_x=args.obj_b_x,
        obj_b_y=args.obj_b_y,
    )

    video_dir = None
    if args.save_video:
        video_dir = os.path.expanduser(
            "~/workspace/Isaac-GR00T/grasp_experiments/videos"
        )
        os.makedirs(video_dir, exist_ok=True)
        print(f"  Videos will be saved to: {video_dir}")

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_steps,
        ),
    )

    env_name_out, results, info = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=args.n_episodes,
        n_envs=1,
    )

    success_rate = sum(results) / len(results)
    print(f"\n--- results ---")
    print(f"{sum(results)}/{len(results)} ({success_rate:.0%})")
    print(f"episodes: {results}")
    if args.save_video:
        print(f"videos saved to {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed PnP env for VLA eval")
 
    parser.add_argument("--preview", action="store_true", help="interactive mjviewer mode")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
 
    parser.add_argument("--obj-a", default="banana")
    parser.add_argument("--obj-b", default="bowl")
    parser.add_argument("--obj-a-x", type=float, default=-0.4)
    parser.add_argument("--obj-a-y", type=float, default=0.4)
    parser.add_argument("--obj-b-x", type=float, default=0.4)
    parser.add_argument("--obj-b-y", type=float, default=0.4)
 
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--save-video", action="store_true")
 
    args = parser.parse_args()
 
    if args.preview:
        run_preview(args)
    else:
        run_episodes(args)
