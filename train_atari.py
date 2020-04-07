import gym
import pickle
from QLearner import DQNAgent

def train_model(num_time_steps, agent_params):
    
    # make env get frame info.
    env = gym.make('PongNoFrameskip-v4')
    frame = env.reset()
    height, width, channels = frame.shape
    
    #agent params.
    num_frames = agent_params['num_frames']
    buffer_size = agent_params['buffer_size']
    batch_size = agent_params['batch_size']
    start_learning_steps = agent_params['start_learning_steps']
    update_freq = agent_params['update_freq']
    
    # initalize the agent.
    agent = DQNAgent(env, height, width, num_frames, env.action_space.n,
                     num_timesteps=num_time_steps, buffer_size=buffer_size)  
    # main training loop.
    for step_num in range(num_time_steps):
        
        # step the agent.
        _, _, _, done = agent.step_env()
        
        # if we need to update.
        if (step_num > start_learning_steps) and (step_num % update_freq == 0):
            
            # update the model.
            agent.update(batch_size)
            
        if done:
            print(agent.best_mean_episode_reward)
            
        if step_num % 10 == 0:
            print(step_num)
            
    return agent


def main():
    agent_params = {'num_frames':4, 'buffer_size':100000, 'batch_size':32, 'start_learning_steps':500, 'update_freq':4}

    # Run training
    agent = train_model(2000, agent_params)

    
if __name__ == "__main__":
    main()

