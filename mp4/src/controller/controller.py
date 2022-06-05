import rospy
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from util.util import euler_to_quaternion, quaternion_to_euler

class VehicleController():

    def __init__(self, model_name='gem'):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/" + model_name + "/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.model_name = model_name

    def execute(self, target_v, lateral_error, lane_theta):
        """
            This function takes the current state of the vehicle and
            the target state to compute low-level control input to the vehicle
            Inputs:
                target_v: The desired velocity of the vehicle
                lateral_error: The lateral tracking error from the center line of the current lane
                lane_theta: The current lane heading
        """

        # TODO: compute errors and send computed control input to vehicle

        ky = 2
        k_theta = 2
        velocity = target_v
        steering_angle = ky*lateral_error + k_theta*lane_theta
        
 

        # Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = velocity
        newAckermannCmd.steering_angle = steering_angle

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)