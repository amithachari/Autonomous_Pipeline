import pickle
import numpy as np

class VehicleDecision():
    def __init__(self):
        self.lane_width = 4.4

        # vehicle state variable
        # 0: lane keeping
        # 1: lane changing - turning stage
        # 2: lane changing - stabilizing stage
        # 3: emergency stop
        self.vehicle_state = 0

        self.state_verbose = ["lane keeping", "start lane changing", "stabilizing the car", "emergency stop"]

        # current lane
        self.current_lane = 'left'

    def get_ref_state(self, currState, front_dist, lateral_error, lane_theta):
        """
            Get the reference state for the vehicle according to the current state and result from perception module
            Inputs:
                currState: ModelState, the current state of vehicle
                front_dist: float, the current distance between vehicle and obstacle in front
                lateral_error: float, the current lateral tracking error from the center line
                lane_theta: the current lane heading with respect to the vehicle
            Outputs: reference velocity, lateral tracking error, and lane heading
        """

        # TODO: Implement decision module

        ref_v = 15
        lateral_error = lateral_error
        lane_theta = lane_theta


        #### some thoughts
      
        # lane_width_meters = 4.4
        # lane_width_pixels = 265.634
        # self.meter_per_pixel = lane_width_meters / lane_width_pixels
        # abs(right_fitx_pixel - left_fitx_pixel) *

        
        if front_dist < 5:
            self.vehicle_state = 1  ## change lane: mode 2
            lane_width_meters = 4.4
            lateral_error = abs(lane_width_meters - lateral_error)
            if lateral_error < lane_width_meters/2:
                self.vehicle_state = 2  ## stabilize: mode 2
                ref_v = 15
                lateral_error = lateral_error
                lane_theta = lane_theta


        print("current state: ", self.state_verbose[self.vehicle_state])

        return ref_v, lateral_error, lane_theta
