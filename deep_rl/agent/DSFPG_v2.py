#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from logging import setLogRecordFactory
from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

# for offline training
import d4rl
import gym


'''
Difference from DDPG :
1. Q value is calculated by SF network
'''

class DSFPGAgent_v2(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.unsupervised_steps = 0
        self.state = None

        # assert config.train_what in ['SF', 'ALL']
        # self.train_what = config.train_what

        # flag that idicate offline/online setting
        self.offline = config.offline
        self.dataset_name = config.dataset_name
        self.dataset = None

        self.pre_training = False

        try:
            self.w_learning = int(config.w_learning)
            self.psi_learning = int(config.psi_learning)
            self.act_learning = int(config.act_learning)
        except:
            self.w_learning = int(1e6)
            self.psi_learning = int(1e6)
            self.act_learning = int(1e6)
        

        if self.offline:
            self.env = gym.make(self.dataset_name)
            self.dataset = self.env.get_dataset()
            self.offline_replay_init()



    def offline_replay_init(self):
        # dataset must be created
        assert self.dataset is not None
        actions = self.dataset['actions']
        observations = self.dataset['observations']
        terminals = self.dataset['terminals']
        rewards = self.dataset['rewards']

        for i in range(len(rewards) - 1):
            # if the next state is still within the same episode
            if not terminals[i]:
                obs = observations[i].reshape(1, -1)
                action = actions[i].reshape(1, -1)
                next_obs = observations[i].reshape(1, -1)
                reward = rewards[i].reshape(1, -1)
                done = terminals[i+1].reshape(1, -1)
                self.replay.feed(dict(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    mask=1-np.asarray(done, dtype=np.int32),
                ))

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        self.step_online()


    def step_offline(self):
        self.total_steps += 1
        config = self.config
        transitions = self.replay.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)

        
        # get next observation
        phi_next = self.target_network.feature(next_states)
        phi = self.target_network.feature(states)
        # get next action
        a_next = self.target_network.actor(phi_next)
        # get next psi value
        sf_next = self.target_network.sf(phi_next, a_next)
        # get next q value
        q_next = self.target_network.critic(sf_next)

        # this is the q_target
        q_next = config.discount * mask * q_next
        q_next.add_(rewards)
        q_next = q_next.detach()

        # this is the psi_target
        one_dim_mask = mask.reshape(mask.shape[0], -1)
        one_dim_mask = one_dim_mask.expand(one_dim_mask.shape[0], sf_next.shape[-1])
        sf_next = sf_next * one_dim_mask
        sf_next.add_(phi)

        # get current values 
        sf = self.network.sf(phi, actions)
        q = self.network.critic(sf)

        # for SF
        if self.unsupervised_steps == 0:
            # backprop
            # for critic
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            self.network.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.network.critic_opt.step()

            sf_loss = (sf - sf_next).pow(2).mul(0.5).sum(-1).mean()
            self.network.zero_grad()
            sf_loss.backward()
            self.network.sf_opt.step()

        # for actor
        phi = self.network.feature(states)
        action = self.network.actor(phi)
        sf = self.network.sf(phi.detach(), action)
        policy_loss = -self.network.critic(sf).mean()
        
        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)



    def step_online(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        if self.total_steps > max(self.w_learning, self.psi_learning):
            # print('recording')
            self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            # get next observation
            phi_next = self.target_network.feature(next_states)
            phi = self.target_network.feature(states)
            # get next action
            a_next = self.target_network.actor(phi_next)
            # get next psi value
            sf_next = self.target_network.sf(phi_next, a_next)

            one_dim_mask = mask.reshape(mask.shape[0], -1)
            one_dim_mask = one_dim_mask.expand(one_dim_mask.shape[0], sf_next.shape[-1])
            sf_next = sf_next * one_dim_mask
            sf_next.add_(phi)

            # get current values 
            sf = self.network.sf(phi, actions)
            # q = self.network.critic(sf)

            # backprop
            # for critic
            if self.w_learning > self.total_steps:
                phi_dot_w = self.network.critic(phi_next.detach())
                critic_loss = (rewards - phi_dot_w).pow(2).mul(0.5).sum(-1).mean()
                # critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
                self.network.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.network.critic_opt.step()

            if self.psi_learning > self.total_steps:
                # for SF
                sf_loss = (sf - sf_next).pow(2).mul(0.5).sum(-1).mean()
                # sf_loss = sf_loss + critic_loss
                self.network.zero_grad()
                sf_loss.backward()
                self.network.sf_opt.step()

            if self.total_steps >= max(self.psi_learning, self.w_learning):
                if self.total_steps == max(self.psi_learning, self.w_learning):
                    print('=========== End of unsupervised training ============')
                    print('================= RL training start =================')
                # for actor
                phi = self.network.feature(states)
                action = self.network.actor(phi)
                sf = self.network.sf(phi.detach(), action)
                policy_loss = -self.network.critic(sf).mean()
                
                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
