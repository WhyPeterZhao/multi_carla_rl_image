import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *

class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True,cars=3) -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = [None]*cars
        self.current_waypoint_index = [0]*cars
        self.checkpoint_waypoint_index = [0]*cars
        self.fresh_start = [True]*cars
        self.checkpoint_frequency = [checkpoint_frequency]*cars
        self.route_waypoints = [list()]*cars
        self.town = town
        self.timesteps = [0]*cars
        # Objects to be kept alive
        self.camera_obj = [None]*cars
        self.env_camera_obj = [None]*cars
        self.collision_obj = [None]*cars
        self.lane_invasion_obj = [None]*cars


        self.sensor_list = [list()]*cars
        self.actor_list = [list()]*cars

        self.cars = cars # 环境中车的数量

        self.velocity = [float(0.0)]*cars
        self.previous_steer = [float(0.0)]*cars
        self.throttle = [float(0.0)]*cars
        self.total_distance = [780]*cars
        self.image_obs = [None]*cars
        self.collision_history = [None]*cars
        self.rotation = [None] * cars
        self.previous_location = [None] * cars
        self.distance_traveled = [None] * cars
        self.center_lane_deviation = [None] * cars
        self.target_speed = [None] * cars
        self.max_speed = [None] * cars
        self.min_speed = [None] * cars
        self.max_distance_from_center = [None] * cars
        self.distance_from_center = [None] * cars
        self.angle = [None] * cars
        self.distance_covered = [None] * cars

        self.navigation_obs = [None] * cars
        self.episode_start_time = [None] * cars
        self.location = [None] * cars
        self.current_waypoint = [None] * cars
        self.next_waypoint = [None] * cars
        self.front_camera = [None] * cars
        self.wrong_maneuver = [None] * cars

    def reset_all(self):
        """
        Reset the environment for all vehicles.
        """
        observations = [None]*self.cars
        for number in range(self.cars):
            observations[number] = self.reset(number=number)
        return observations

    # A reset function for reseting our environment.
    def reset(self, number):

        # try:
            
        if len(self.actor_list[number]) != 0 or len(self.sensor_list[number]) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list[number]])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list[number]])
            self.sensor_list[number].clear()
            self.actor_list[number].clear()
        self.remove_sensors(number)


        # Blueprint of our main vehicle
        vehicle_bp = self.get_vehicle(CAR_NAME)

        # 随机设置车辆位置
        transform = random.choice(self.map.get_spawn_points())
        self.total_distance[number] = 780

        self.vehicle[number] = self.world.try_spawn_actor(vehicle_bp, transform)
        self.actor_list[number].append(self.vehicle[number])


        # Camera Sensor
        self.camera_obj[number] = CameraSensor(self.vehicle[number])
        while(len(self.camera_obj[number].front_camera) == 0):
            time.sleep(0.0001)
        self.image_obs[number] = self.camera_obj[number].front_camera.pop(-1)
        self.sensor_list[number].append(self.camera_obj[number].sensor)

        # Third person view of our vehicle in the Simulated env
        if self.display_on:
            self.env_camera_obj[number] = CameraSensorEnv(self.vehicle[number])
            self.sensor_list[number].append(self.env_camera_obj[number].sensor)

        # Collision sensor
        self.collision_obj[number] = CollisionSensor(self.vehicle[number])
        self.collision_history[number] = self.collision_obj[number].collision_data
        self.sensor_list[number].append(self.collision_obj[number].sensor)

        
        self.timesteps[number] = 0
        self.rotation[number] = self.vehicle[number].get_transform().rotation.yaw
        self.previous_location[number] = self.vehicle[number].get_location()
        self.distance_traveled[number] = 0.0
        self.center_lane_deviation[number] = 0.0
        self.target_speed[number] = 22 #km/h
        self.max_speed[number] = 25.0
        self.min_speed[number] = 15.0
        self.max_distance_from_center[number] = 3
        self.throttle[number] = float(0.0)
        self.previous_steer[number] = float(0.0)
        self.velocity[number] = float(0.0)
        self.distance_from_center[number] = float(0.0)
        self.angle[number] = float(0.0)
        self.center_lane_deviation[number] = 0.0
        self.distance_covered[number] = 0.0

        if self.fresh_start[number]:
            self.current_waypoint_index[number] = 0
            # Waypoint nearby angle and distance from it
            self.route_waypoints[number] = list()
            waypoint = self.map.get_waypoint(self.vehicle[number].get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
            current_waypoint = waypoint
            self.route_waypoints[number].append(current_waypoint)
            for x in range(self.total_distance[number]):
                # if self.town == "Town07":
                #     if x < 650:
                #         next_waypoint = current_waypoint.next(1.0)[0]
                #     else:
                #         next_waypoint = current_waypoint.next(1.0)[-1]
                # elif self.town == "Town02":
                #     if x < 650:
                #         next_waypoint = current_waypoint.next(1.0)[-1]
                #     else:
                #         next_waypoint = current_waypoint.next(1.0)[0]
                # else:
                #     next_waypoint = current_waypoint.next(1.0)[0]
                if x < 650:
                    next_waypoint = current_waypoint.next(1.0)[-1]
                else:
                    next_waypoint = current_waypoint.next(1.0)[0]
                self.route_waypoints[number].append(next_waypoint)
                current_waypoint = next_waypoint
        else:
            # Teleport vehicle to last checkpoint
            waypoint = self.route_waypoints[number][self.checkpoint_waypoint_index[number] % len(self.route_waypoints[number])]
            transform = waypoint.transform
            self.vehicle[number].set_transform(transform)
            self.current_waypoint_index[number] = self.checkpoint_waypoint_index[number]

        self.navigation_obs[number] = np.array([self.throttle[number], self.velocity[number], self.previous_steer[number], self.distance_from_center[number], self.angle[number]])

                    
        time.sleep(0.5)
        self.collision_history[number].clear()

        self.episode_start_time[number] = time.time()
        observations = [self.image_obs[number], self.navigation_obs[number]]
        return observations

        # except:
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list[number]])
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list[number]])

        #     self.sensor_list[number].clear()
        #     self.actor_list[number].clear()
        #     self.remove_sensors(number)
        #     if self.display_on:
        #         pygame.quit()




    def step(self, action_idx):
        observations = [None]*self.cars
        rewards = [None]*self.cars
        dones = [False]*self.cars
        infos = [None]*self.cars
        for number in range(self.cars):
            if self.vehicle[number] is None:
                observations[number] = self.reset(number=number)
                rewards[number] = 0
                dones[number] = False
                infos[number] = None
                self.timesteps[number]+=1
                continue
            try:
                self.timesteps[number]+=1
                self.fresh_start[number] = False

                velocity = self.vehicle[number].get_velocity()
                self.velocity[number] = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                if self.continous_action_space:
                    steer = float(action_idx[number][0])
                    steer = max(min(steer, 1.0), -1.0)
                    throttle = float((action_idx[number][1] + 1.0)/2)
                    throttle = max(min(throttle, 1.0), 0.0)
                    self.vehicle[number].apply_control(carla.VehicleControl(steer=self.previous_steer[number]*0.9 + steer*0.1, throttle=self.throttle[number]*0.9 + throttle*0.1))
                    self.previous_steer[number] = steer
                    self.throttle[number] = throttle
                else:
                    steer = self.action_space[action_idx[number]]
                    if self.velocity < 20.0:
                        self.vehicle[number].apply_control(carla.VehicleControl(steer=self.previous_steer[number]*0.9 + steer*0.1, throttle=1.0))
                    else:
                        self.vehicle[number].apply_control(carla.VehicleControl(steer=self.previous_steer[number]*0.9 + steer*0.1))
                    self.previous_steer[number] = steer
                    self.throttle[number] = 1.0
                
                # 完成
                # 更新红绿灯，与强化学习无关
                if self.vehicle[number].is_at_traffic_light():
                    traffic_light = self.vehicle[number].get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)

                self.collision_history[number] = self.collision_obj[number].collision_data            

                self.rotation[number] = self.vehicle[number].get_transform().rotation.yaw

                self.location[number] = self.vehicle[number].get_location()

                #transform = self.vehicle.get_transform()
                # Keep track of closest waypoint on the route
                print("self.current_waypoint_index[number]: ", self.current_waypoint_index[number])
                waypoint_index = self.current_waypoint_index[number]
                for _ in range(len(self.route_waypoints[number])):
                    # Check if we passed the next waypoint along the route
                    next_waypoint_index = waypoint_index + 1
                    wp = self.route_waypoints[number][next_waypoint_index % len(self.route_waypoints[number])]
                    dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location[number] - wp.transform.location)[:2])
                    if dot > 0.0:
                        waypoint_index += 1
                    else:
                        break

                self.current_waypoint_index[number] = waypoint_index
                # Calculate deviation from center of the lane
                self.current_waypoint[number] = self.route_waypoints[number][ self.current_waypoint_index[number]    % len(self.route_waypoints[number])]
                self.next_waypoint[number] = self.route_waypoints[number][(self.current_waypoint_index[number]+1) % len(self.route_waypoints[number])]
                self.distance_from_center[number] = self.distance_to_line(self.vector(self.current_waypoint[number].transform.location),self.vector(self.next_waypoint[number].transform.location),self.vector(self.location[number]))
                self.center_lane_deviation[number] += self.distance_from_center[number]

                # Get angle difference between closest waypoint and vehicle forward vector
                fwd    = self.vector(self.vehicle[number].get_velocity())
                wp_fwd = self.vector(self.current_waypoint[number].transform.rotation.get_forward_vector())
                self.angle[number]  = self.angle_diff(fwd, wp_fwd)

                # Update checkpoint for training
                if not self.fresh_start[number]:
                    if self.checkpoint_frequency[number] is not None:
                        self.checkpoint_waypoint_index[number] = (self.current_waypoint_index[number] // self.checkpoint_frequency[number]) * self.checkpoint_frequency[number]

                
                # Rewards are given below!
                done = False
                reward = 0

                if len(self.collision_history[number]) != 0:
                    print("Collision detected!")
                    done = True
                    reward = -10
                elif self.distance_from_center[number] > self.max_distance_from_center[number]:
                    print("Vehicle is out of lane!")
                    done = True
                    reward = -10
                elif self.episode_start_time[number] + 10 < time.time() and self.velocity[number] < 1.0:
                    print("Vehicle is stuck!")
                    reward = -10
                    done = True
                elif self.velocity[number] > self.max_speed[number]:
                    print("Vehicle is going too fast!")
                    reward = -10
                    done = True

                # Interpolated from 1 when centered to 0 when 3 m from center
                centering_factor = max(1.0 - self.distance_from_center[number] / self.max_distance_from_center[number], 0.0)
                # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
                angle_factor = max(1.0 - abs(self.angle[number] / np.deg2rad(20)), 0.0)

                if not done:
                    if self.continous_action_space:
                        if self.velocity[number] < self.min_speed[number]:
                            reward = (self.velocity[number] / self.min_speed[number]) * centering_factor * angle_factor    
                        elif self.velocity[number] > self.target_speed[number]:               
                            reward = (1.0 - (self.velocity[number]-self.target_speed[number]) / (self.max_speed[number]-self.target_speed[number])) * centering_factor * angle_factor  
                        else:                                         
                            reward = 1.0 * centering_factor * angle_factor 
                    else:
                        reward = 1.0 * centering_factor * angle_factor

                if self.timesteps[number] >= 7500:
                    print("Episode finished due to timesteps limit!")
                    done = True
                elif self.current_waypoint_index[number] >= len(self.route_waypoints[number]) - 2:
                    print("self.current_waypoint_index[number]: ", self.current_waypoint_index[number])
                    print("Episode finished due to reaching the end of the route!")
                    done = True
                    self.fresh_start[number] = True
                    if self.checkpoint_frequency[number] is not None:
                        if self.checkpoint_frequency[number] < self.total_distance[number]//2:
                            self.checkpoint_frequency[number] += 2
                        else:
                            self.checkpoint_frequency[number] = None
                            self.checkpoint_waypoint_index[number] = 0

                while(len(self.camera_obj[number].front_camera) == 0):
                    time.sleep(0.0001)

                self.image_obs[number] = self.camera_obj[number].front_camera.pop(-1)
                normalized_velocity = self.velocity[number]/self.target_speed[number]
                normalized_distance_from_center = self.distance_from_center[number] / self.max_distance_from_center[number]
                normalized_angle = abs(self.angle[number] / np.deg2rad(20))
                self.navigation_obs[number] = np.array([self.throttle[number], self.velocity[number], normalized_velocity, normalized_distance_from_center, normalized_angle])
                
                # Remove everything that has been spawned in the env
                if done:
                    self.center_lane_deviation[number] = self.center_lane_deviation[number] / self.timesteps[number]
                    self.distance_covered[number] = abs(self.current_waypoint_index[number] - self.checkpoint_waypoint_index[number])
                    
                    for sensor in self.sensor_list[number]:
                        sensor.destroy()
                    
                    self.remove_sensors(number)
                    
                    for actor in self.actor_list[number]:
                        actor.destroy()
                    self.vehicle[number] = None
                

                observations[number]=[self.image_obs[number], self.navigation_obs[number]]
                rewards[number] = reward
                dones[number] = done
                infos[number] = [self.distance_covered[number], self.center_lane_deviation[number]]
            

            except:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
                self.remove_sensors()
                if self.display_on:
                    pygame.quit()

        return observations, rewards, dones, infos

# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint




    # Clean up method
    def remove_sensors(self, number):
        self.camera_obj[number] = None
        self.collision_obj[number] = None
        self.lane_invasion_obj[number] = None
        self.env_camera_obj[number] = None
        self.front_camera[number] = None
        self.collision_history[number] = None
        self.wrong_maneuver[number] = None

