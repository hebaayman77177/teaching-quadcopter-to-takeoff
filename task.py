import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward=0
        edge0_dis=1
        edge0_speed=0.1
        self.crash=False
        self.destance= ( ( abs(self.sim.pose[:3] - self.target_pos) )**2 ).sum()
        speed = ( (self.sim.v )**2 ).sum()
        
#         reward=reward-self.destance*0.001
        reward = 1 - ( self.destance/1000000 ) ** 0.4
        
        #crash
        if(self.sim.pose[3]<0 and self.sim.v[3]>0):
            reward=reward-10000
            self.crash=True

        #near but with high speed
        if (self.destance < edge0_dis and speed>edge0_speed ):
            reward=reward-1000
            
        #big reward for being almost near
        if (self.destance < 0.5):
            reward=reward+100000000
        
        #-1*((self.sim.pose[:3] - self.target_pos).sum())**2
        #1-((self.sim.pose[:3] - self.target_pos).sum())**2
        #1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #         reward = 1 - ( self.destance/10000 ) ** 0.4
        return reward
#         ydes=self.sim.pose[1]-self.target_pos[1]
#         reward=0
#         reward=reward-ydes
#         if(ydes<0.5):
#             reward=reward+1000
#         return reward
           
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        if(self.destance < .5 or self.crash):
            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state