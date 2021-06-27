import sys
sys.path.append('../')
from deep_rl import *
from deep_rl.agent.DSFPG_agent import DSFPGAgent
from deep_rl.agent.DSFPG_v3 import DSFPGAgent_v3

select_device(0)
torch.backends.cudnn.benchmark = True

def dsfpg_online(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    '''kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)'''
    
    config = Config()
    config.merge(kwargs)

    config.offline = False
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(3e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 10

    config.w_learning = 5e5
    config.psi_learning = 1e6
    config.act_learning = 1e6

    
    config.network_fn = lambda: DSFPGNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        sf_body=FCBody(config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        sf_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: UniformReplay(memory_size=int(1e6), batch_size=256)
    config.discount = 0.99
    '''config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))'''
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    # config.pre_training_steps = int(1e6)
    config.target_network_mix = 5e-3
    # run_steps(DSFPGAgent(config))
    agent = DSFPGAgent_v3(config)
    # config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()

    agent.pre_training = False
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval \
            and not agent.total_steps % config.eval_interval \
            and agent.total_steps > max(config.w_learning, config.psi_learning):
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()
    return agent




dsfpg = dsfpg_online(game='Hopper-v2', dataset_name='hopper-expert-v0')

torch.save(dsfpg.network, '../state_dict/dsfpg_v3_pretraining')