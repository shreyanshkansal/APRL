import os
import typing
from jaxrl5.agents.agent import Agent as JaxRLAgent
from jaxrl5.data import ReplayBuffer
from flax.training import orbax_utils
from natsort import natsorted
import pickle
import shutil
from rail_walker_gym.envs.wrappers.rollout_collect import Rollout
import orbax.checkpoint as ocp

def make_checkpoint_manager(chkpt_dir):
    chkpt_dir = os.path.abspath(chkpt_dir)
    item_names = ('actor', 'critic', 'target_critic', 'temp', 'dynamics_model', 'rng')
    options = ocp.CheckpointManagerOptions(max_to_keep=10, create=True)
    checkpoint_manager = ocp.CheckpointManager(chkpt_dir, options=options, item_names=item_names)
    return checkpoint_manager

def initialize_project_log(project_dir) -> None:
    os.makedirs(project_dir, exist_ok=True)
   
    chkpt_dir = os.path.join(project_dir, 'checkpoints')
    buffer_dir = os.path.join(project_dir, "buffers")
    rollout_dir = os.path.join(project_dir, "rollouts")
    eval_rollout_dir = os.path.join(project_dir, "eval_rollouts")
    
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(eval_rollout_dir, exist_ok=True)

def list_checkpoint_steps(checkpoint_manager) -> typing.List[int]:
    return checkpoint_manager.all_steps()

def list_replay_buffer_steps(project_dir) -> typing.List[int]:
    buffer_dir = os.path.join(project_dir, "buffers")
    buffers = natsorted(os.listdir(buffer_dir))
    return [int(buffer.split('_')[-1].split('.')[0]) for buffer in buffers]

def load_checkpoint_at_step(checkpoint_manager, step : int, agent : JaxRLAgent, if_failed_return_step : int = 0) -> typing.Tuple[int, JaxRLAgent]:
    available_steps = checkpoint_manager.all_steps()
    if step not in available_steps:
        return if_failed_return_step, agent
    try:
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                actor=ocp.args.StandardRestore(agent.actor),
                critic=ocp.args.StandardRestore(agent.critic),
                target_critic=ocp.args.StandardRestore(agent.target_critic),
                temp=ocp.args.StandardRestore(agent.temp),
                dynamics_model=ocp.args.StandardRestore(agent.dynamics_model),
                rng=ocp.args.ArrayRestore(agent.rng),
            )
        )
        loaded_agent = agent.replace(
            actor=restored.actor,
            critic=restored.critic,
            target_critic=restored.target_critic,
            temp=restored.temp,
            dynamics_model=restored.dynamics_model,
            rng=restored.rng,
        )
        return step, loaded_agent
    except Exception as e:
        print(f"Checkpoint load failed with exception: {e}")
        return if_failed_return_step, agent

def load_checkpoint_file(
    filename : str,
    agent : JaxRLAgent
):
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(agent)
    agent = checkpointer.restore(filename, item=agent, restore_kwargs={'restore_args': restore_args})
    return agent

def load_latest_checkpoint(checkpoint_manager, agent : JaxRLAgent, if_failed_return_step: int = 0) -> typing.Tuple[int, JaxRLAgent]:
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return if_failed_return_step, agent
    try:
        restored = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.Composite(
                actor=ocp.args.StandardRestore(agent.actor),
                critic=ocp.args.StandardRestore(agent.critic),
                target_critic=ocp.args.StandardRestore(agent.target_critic),
                temp=ocp.args.StandardRestore(agent.temp),
                dynamics_model=ocp.args.StandardRestore(agent.dynamics_model),
                rng=ocp.args.ArrayRestore(agent.rng),
            )
        )
        loaded_agent = agent.replace(
            actor=restored.actor,
            critic=restored.critic,
            target_critic=restored.target_critic,
            temp=restored.temp,
            dynamics_model=restored.dynamics_model,
            rng=restored.rng,
        )
        return latest_step, loaded_agent
    except Exception as e:
        print(f"Checkpoint load failed with exception: {e}")
        return if_failed_return_step, agent

def load_latest_replay_buffer(project_dir) -> typing.Optional[typing.Tuple[int, ReplayBuffer]]:
    buffer_dir = os.path.join(project_dir, "buffers")
    buffers = natsorted(os.listdir(buffer_dir))
    if len(buffers) == 0:
        return None
    else:
        buffer_int = int(buffers[-1].split('_')[-1].split('.')[0])
        with open(os.path.join(buffer_dir, buffers[-1]),'rb') as f:
            replay_buffer = pickle.load(f)
        return buffer_int, replay_buffer

def load_replay_buffer_at_step(project_dir, step: int) -> typing.Optional[ReplayBuffer]:
    buffer_dir = os.path.join(project_dir, "buffers")
    try:
        with open(os.path.join(buffer_dir, f'buffer_{step}.pkl'),'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer
    except:
        return None

def load_latest_additional_replay_buffer(project_dir) -> typing.Optional[ReplayBuffer]:
    buffer_dir = os.path.join(project_dir, "additional_buffers")
    if not os.path.exists(buffer_dir):
        return None
    buffers = natsorted(os.listdir(buffer_dir))
    if len(buffers) <= 1:
        return None
    else:
        with open(os.path.join(buffer_dir, buffers[-2]),'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer

def load_replay_buffer_file(filename : str) -> ReplayBuffer:
    with open(filename,'rb') as f:
        replay_buffer = pickle.load(f)
    return replay_buffer

def save_checkpoint(checkpoint_manager, step : int, agent : JaxRLAgent) -> None:
    try:
        checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                actor=ocp.args.StandardSave(agent.actor),
                critic=ocp.args.StandardSave(agent.critic),
                target_critic=ocp.args.StandardSave(agent.target_critic),
                temp=ocp.args.StandardSave(agent.temp),
                dynamics_model=ocp.args.StandardSave(agent.dynamics_model),
                rng=ocp.args.ArraySave(agent.rng),
            )
        )
    except Exception as e:
        print('Checkpoint save failed:', e)

def save_replay_buffer(project_dir, step: int, replay_buffer: ReplayBuffer, delete_old_buffers : bool = True) -> None:
    buffer_dir = os.path.join(project_dir, "buffers")
    if delete_old_buffers:
        try:
            shutil.rmtree(buffer_dir)
        except:
            pass
    try:
        os.makedirs(buffer_dir, exist_ok=True)
        with open(os.path.join(buffer_dir, f'buffer_{step}.pkl'),
                    'wb') as f:
            pickle.dump(replay_buffer, f)
    except:
        pass

def save_rollout(project_dir, step: int, is_training : bool, rollout : Rollout):
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        os.makedirs(rollout_dir, exist_ok=True)
        rollout.export_npz(os.path.join(rollout_dir, f'rollout_{step}.npz'))
    except:
        pass

def save_episode_rollout(project_dir, step: int, is_training : bool, rollout : Rollout):
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        os.makedirs(rollout_dir, exist_ok=True)
        save_file = os.path.join(rollout_dir, f'rollout_{step}.npz')
        if rollout.is_current_episode_empty():
            rollout.export_last_episode_npz(save_file)
        else:
            rollout.export_current_episode_npz(save_file)
    except:
        pass

def list_rollout_steps(project_dir, is_training : bool) -> typing.List[int]:
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    rollouts = natsorted(os.listdir(rollout_dir))
    return [int(rollout.split('_')[-1].split('.')[0]) for rollout in rollouts]

def load_rollout_at_step(project_dir, step: int, is_training : bool) -> typing.Optional[Rollout]:
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        rollout = Rollout.import_npz(os.path.join(rollout_dir, f'rollout_{step}.npz'))
        return rollout
    except:
        return None