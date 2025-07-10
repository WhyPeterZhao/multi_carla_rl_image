import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *


class CarlaEnvironment():

    def __init__(self, client, world, town, num_vehicles=1, checkpoint_frequency=100, continuous_action=True) -> None:
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.num_vehicles = num_vehicles  # Number of controllable vehicles
        self.vehicles = []  # List to store all vehicles
        self.settings = None
        self.current_waypoint_indices = [0] * num_vehicles
        self.checkpoint_waypoint_indices = [0] * num_vehicles
        self.fresh_start = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive (now lists for each vehicle)
        self.camera_objs = []
        self.env_camera_objs = []
        self.collision_objs = []
        self.lane_invasion_objs = []

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()

    def reset(self):
        try:
            # Clear existing actors and sensors
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()

            # Clear vehicle-related lists
            self.vehicles = []
            self.camera_objs = []
            self.env_camera_objs = []
            self.collision_objs = []
            self.current_waypoint_indices = [0] * self.num_vehicles
            self.checkpoint_waypoint_indices = [0] * self.num_vehicles

            # Spawn all vehicles
            for i in range(self.num_vehicles):
                # Blueprint of our main vehicle
                vehicle_bp = self.get_vehicle(CAR_NAME)

                if self.town == "Town07":
                    transform = self.map.get_spawn_points()[38 + i]  # Offset spawn points for each vehicle
                    self.total_distance = 750
                elif self.town == "Town02":
                    transform = self.map.get_spawn_points()[1 + i]  # Offset spawn points for each vehicle
                    self.total_distance = 780
                else:
                    transform = random.choice(self.map.get_spawn_points())
                    self.total_distance = 250

                vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
                if vehicle is not None:
                    self.vehicles.append(vehicle)
                    self.actor_list.append(vehicle)

                    # Camera Sensor for this vehicle
                    camera_obj = CameraSensor(vehicle)
                    while(len(camera_obj.front_camera) == 0):
                        time.sleep(0.0001)
                    self.camera_objs.append(camera_obj)
                    self.sensor_list.append(camera_obj.sensor)

                    # Third person view of our vehicle in the Simulated env
                    if self.display_on:
                        env_camera_obj = CameraSensorEnv(vehicle)
                        self.env_camera_objs.append(env_camera_obj)
                        self.sensor_list.append(env_camera_obj.sensor)

                    # Collision sensor for this vehicle
                    collision_obj = CollisionSensor(vehicle)
                    self.collision_objs.append(collision_obj)
                    self.sensor_list.append(collision_obj.sensor)

            # Initialize route waypoints (shared among all vehicles)
            if self.fresh_start:
                self.route_waypoints = list()
                waypoint = self.map.get_waypoint(self.vehicles[0].get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint

            # Initialize observations for each vehicle
            image_observations = []
            navigation_observations = []
            
            for i, vehicle in enumerate(self.vehicles):
                # Initialize vehicle-specific variables
                self.current_waypoint_indices[i] = 0
                if not self.fresh_start:
                    # Teleport vehicle to last checkpoint
                    waypoint = self.route_waypoints[self.checkpoint_waypoint_indices[i] % len(self.route_waypoints)]
                    transform = waypoint.transform
                    vehicle.set_transform(transform)
                    self.current_waypoint_indices[i] = self.checkpoint_waypoint_indices[i]

                # Get initial observations
                while(len(self.camera_objs[i].front_camera) == 0):
                    time.sleep(0.0001)
                image_obs = self.camera_objs[i].front_camera.pop(-1)
                image_observations.append(image_obs)
                
                # Initialize navigation observations
                nav_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # throttle, velocity, previous_steer, distance_from_center, angle
                navigation_observations.append(nav_obs)

            time.sleep(0.5)
            for collision_obj in self.collision_objs:
                collision_obj.collision_data.clear()

            self.episode_start_time = time.time()
            return [image_observations, navigation_observations]

        except Exception as e:
            print(f"Error in reset: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def step(self, actions):
        try:
            # Initialize lists for observations and rewards
            image_observations = []
            navigation_observations = []
            rewards = []
            dones = []
            infos = []
            
            # Process each vehicle
            for i, vehicle in enumerate(self.vehicles):
                # Get action for this vehicle
                action_idx = actions[i]
                
                # Velocity of the vehicle
                velocity = vehicle.get_velocity()
                current_velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                # Apply control based on action
                if self.continous_action_space:
                    steer = float(action_idx[0])
                    steer = max(min(steer, 1.0), -1.0)
                    throttle = float((action_idx[1] + 1.0)/2)
                    throttle = max(min(throttle, 1.0), 0.0)
                    vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle))
                else:
                    steer = self.action_space[action_idx]
                    if current_velocity < 20.0:
                        vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=1.0))
                    else:
                        vehicle.apply_control(carla.VehicleControl(steer=steer))
                
                # Traffic Light state
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)

                # Get collision history for this vehicle
                collision_history = self.collision_objs[i].collision_data
                
                # Location of the car
                location = vehicle.get_location()
                
                # Keep track of closest waypoint on the route
                waypoint_index = self.current_waypoint_indices[i]
                for _ in range(len(self.route_waypoints)):
                    # Check if we passed the next waypoint along the route
                    next_waypoint_index = waypoint_index + 1
                    wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                    dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2], 
                                self.vector(location - wp.transform.location)[:2])
                    if dot > 0.0:
                        waypoint_index += 1
                    else:
                        break

                self.current_waypoint_indices[i] = waypoint_index
                
                # Calculate deviation from center of the lane
                current_waypoint = self.route_waypoints[self.current_waypoint_indices[i] % len(self.route_waypoints)]
                next_waypoint = self.route_waypoints[(self.current_waypoint_indices[i]+1) % len(self.route_waypoints)]
                distance_from_center = self.distance_to_line(self.vector(current_waypoint.transform.location),
                                                           self.vector(next_waypoint.transform.location),
                                                           self.vector(location))
                
                # Get angle difference between closest waypoint and vehicle forward vector
                fwd = self.vector(vehicle.get_velocity())
                wp_fwd = self.vector(current_waypoint.transform.rotation.get_forward_vector())
                angle = self.angle_diff(fwd, wp_fwd)

                # Update checkpoint for training
                if not self.fresh_start and self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_indices[i] = (self.current_waypoint_indices[i] // self.checkpoint_frequency) * self.checkpoint_frequency

                # Determine if episode is done for this vehicle
                done = False
                reward = 0

                if len(collision_history) != 0:
                    done = True
                    reward = -10
                elif distance_from_center > self.max_distance_from_center:
                    done = True
                    reward = -10
                elif self.episode_start_time + 10 < time.time() and current_velocity < 1.0:
                    reward = -10
                    done = True
                elif current_velocity > self.max_speed:
                    reward = -10
                    done = True

                # Calculate reward if not done
                if not done:
                    centering_factor = max(1.0 - distance_from_center / self.max_distance_from_center, 0.0)
                    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

                    if self.continous_action_space:
                        if current_velocity < self.min_speed:
                            reward = (current_velocity / self.min_speed) * centering_factor * angle_factor    
                        elif current_velocity > self.target_speed:               
                            reward = (1.0 - (current_velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                        else:                                         
                            reward = 1.0 * centering_factor * angle_factor 
                    else:
                        reward = 1.0 * centering_factor * angle_factor

                # Check other termination conditions
                if self.current_waypoint_indices[i] >= len(self.route_waypoints) - 2:
                    done = True
                    self.fresh_start = True
                    if self.checkpoint_frequency is not None:
                        if self.checkpoint_frequency < self.total_distance//2:
                            self.checkpoint_frequency += 2
                        else:
                            self.checkpoint_frequency = None
                            self.checkpoint_waypoint_indices[i] = 0

                # Get image observation
                while(len(self.camera_objs[i].front_camera) == 0):
                    time.sleep(0.0001)
                image_obs = self.camera_objs[i].front_camera.pop(-1)
                image_observations.append(image_obs)
                
                # Create navigation observation
                normalized_velocity = current_velocity/self.target_speed
                normalized_distance_from_center = distance_from_center / self.max_distance_from_center
                normalized_angle = abs(angle / np.deg2rad(20))
                nav_obs = np.array([throttle if self.continous_action_space else 1.0,
                                   current_velocity,
                                   normalized_velocity,
                                   normalized_distance_from_center,
                                   normalized_angle])
                navigation_observations.append(nav_obs)
                
                rewards.append(reward)
                dones.append(done)
                infos.append({
                    'distance_covered': abs(self.current_waypoint_indices[i] - (self.checkpoint_waypoint_indices[i] if not self.fresh_start else 0)),
                    'center_lane_deviation': distance_from_center
                })

            # Check if all vehicles are done
            all_done = all(dones)
            
            # Clean up if episode is done for all vehicles
            if all_done:
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()

            return [image_observations, navigation_observations], rewards, dones, infos

        except Exception as e:
            print(f"Error in step: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    # The following methods remain unchanged as they are utility functions:
    # create_pedestrians, set_other_vehicles, change_town, get_world, 
    # get_blueprint_library, angle_diff, distance_to_line, vector, 
    # get_discrete_action_space, get_vehicle, set_vehicle, remove_sensors



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


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


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

