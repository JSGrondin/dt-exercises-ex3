#!/usr/bin/env python3
import numpy as np
import rospy
#import debugpy
import os
import json

#import debugpy
#debugpy.listen(("localhost", 5678))

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import PurePursuitLaneController
from scipy.optimize import curve_fit

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1

def get_xy(points):
    x = []
    y = []
    for point in points:
        x.append(point[0][0])
        x.append(point[1][0])
        y.append(point[0][1])
        y.append(point[1][1])
    return (x,y)

def fit_and_show(x,y, c="b"):
    a,b = fit(x,y)
    show_line(a,b,c)
    return a,b


def fit(x,y, c="b"):
    if len(x)>3:
        popt, pcov = curve_fit(line_func, x, y)
        return popt
    else:
        raise ValueError("Not enough data points")

        
def line_func(x, a, b):
    #return a * np.exp(-b * x) + c
    #return a * x**2 + b*x + c
    return a*x + b

def show_line(a,b,c):
    model_x = np.arange(0,1,0.01)
    model_y = line_func(model_x, a,b)
    plt.plot(model_x, model_y, c)

def get_aim_point(a,b, dist, offset, white_line=False):
    x, y = dist,dist*a+b-offset
    #patch for the dreaded "s" curve:
    if b > 0 and white_line:
        #Looks like we are about to cross the line,
        #let's turn right!
        x, y = dist, rospy.get_param("patch_right_turn",-0.15)
    return (x,y)


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.breakpoints_enabled=False
        # Line Segments: 
        #line_segment_node = "/agent/ground_projection_node/lineseglist_out"
        agent = node_name.split("/")[0]
        print(node_name.split("/"))
        #line_segment_node = f"/{agent}/lane_filter_node/seglist_filtered"
        line_segment_node = "object_detection_node/seglist_filtered"
        self.log(f"Filtered segment topic : {line_segment_node}")
        
        #line_segment_node = "/agent/lane_filter_node/seglist_filtered"
        self.sub_ground_projected_lanes = rospy.Subscriber(line_segment_node,
                                                            SegmentList,
                                                            self.cbGroundProjectedLineSegments,
                                                            queue_size=1)

        #debugpy.listen(5678)
        self.log("Waiting for debugger attach")

        self.right_offset = 0.21
        #self.lookup_distance = 0.2
        self.lookup_depth = 0.2
        #self.white_lookup_distance=0.4
        self.max_speed = 0.1
        self.K = 1
        self.last_omega=0
        self.last_alpha=0
        self.last_v = self.max_speed
        self.last_datalog = None

        self.last_aim_point = (0.2, 0)

        rospy.set_param("relative_name", 10.0)

        if not os.path.exists("/code/datalog"):
            try:
                os.mkdir("/code/datalog")
            except:
                self.log("Unable to open the datalog dir. Probably in evaluation, so it does not matter.")
        #debugpy.wait_for_client()
        self.log("Initialized")
        #self.log('break on this line')

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        car_control_msg.v = 0.5
        car_control_msg.omega = 0

        

        #if self.breakpoints_enabled:
        #    debugpy.breakpoint()
        #    self.log('break on this line')

        #self.publishCmd(car_control_msg)

    def cbGroundProjectedLineSegments(self, segments_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """

        # self.log("Alive and well")

        self.segments_msg = segments_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.segments_msg.header

        relative_name = rospy.get_param("relative_name")

        lookup_distance = rospy.get_param("lookup_distance",0.35)
        offset = rospy.get_param("offset",0.25)


        if self.breakpoints_enabled:
            #debugpy.breakpoint()
            self.log('break on this line')

        yellow_lines = []
        white_lines = []
        bezier_lines = []

        lines_dict = {}

        for segment in segments_msg.segments:
            assert len(segment.points)==2
            # x is the distance from the front of the duckie.
            # y is the left-right distance. 
            start = (segment.points[0].x,segment.points[0].y)
            end = (segment.points[1].x,segment.points[1].y)
            if segment.color==1:
                #Yellow line.
                yellow_lines.append((start,end))

            elif segment.color==0:
                #White line.
                white_lines.append((start,end))

            elif segment.color==2:
                #Bezier lines are registered as red for now, as only three different colors are available and we are not
                #using red for now
                bezier_lines.append((start,end))

        lines = {}
        lines["white"] = white_lines
        lines["yellow"] = yellow_lines
        lines["bezier"] = bezier_lines

        print(lines['bezier'])

        datalog = json.dumps(lines)
        if rospy.get_param("datalog",False) and (self.last_datalog!=datalog):
            with open(f"/code/datalog/segments_{segments_msg.header.seq}.json", "w") as f:
                f.write(datalog)
            self.last_datalog = datalog

        if self.breakpoints_enabled:
            #debugpy.breakpoint()
            self.log('break on this line')

        #lookup_distance =self.lookup_distance

        aim_y = 0
        aim_x = lookup_distance
        match=False

        car_control_msg.omega = self.last_omega

        yellow_aim_point = None
        white_aim_point = None
        bezier_aim_point = None
        #New code here
        x, y = get_xy(lines["white"])
        a_w = b_w = -1
        try:
            a_w,b_w = fit(x,y)
            # white_aim_point = get_aim_point(a_w,b_w,lookup_distance,-offset, white_line=True)
        except ValueError:
            pass

        x, y = get_xy(lines["yellow"])
        a_y=b_y=-1
        try:
            a_y,b_y = fit(x,y)
            yellow_aim_point = get_aim_point(a_y,b_y,lookup_distance,offset)
        except ValueError:
            pass

        x, y = get_xy(lines["bezier"])
        a_bez=b_bez=-1
        try:
            a_bez,b_bez = fit(x,y)
            bezier_aim_point = get_aim_point(a_bez,b_bez,lookup_distance,offset=0)
        except ValueError:
            pass

        aim_point=None
        if bezier_aim_point:
            aim_point = bezier_aim_point
        elif yellow_aim_point:
            aim_point = yellow_aim_point
            if white_aim_point:
                aim_point = (
                                ((yellow_aim_point[0] + white_aim_point[0]) / 2),
                                ((yellow_aim_point[1] + white_aim_point[1]) / 2)
                )
        else:
            aim_point = white_aim_point

        if aim_point is None:
            aim_point = self.last_aim_point
        else:
            self.last_aim_point=aim_point

        #aim_point = (aim_point[0], aim_point[1]+rospy.get_param("lane_offset", -0.03))

        if abs(aim_point[1]) < rospy.get_param("hyst",0.03):
            aim_point = (aim_point[0], 0)

        self.log('bezier aim point   yellow aim point   white aim point')
        self.log(f"{bezier_aim_point}, {yellow_aim_point}, {white_aim_point}")
        #
        alpha = np.arctan(aim_point[1]/aim_point[0])
        d_alpha = alpha-self.last_alpha
        car_control_msg.omega = np.sin(alpha) * rospy.get_param("K",8)
        car_control_msg.omega += np.sin(d_alpha) * rospy.get_param("D",100)

        self.last_alpha = alpha

        #norm = max(0, 1 - abs(car_control_msg.omega))
        #norm_speed = max(rospy.get_param("turn_speed",0.7), norm * rospy.get_param("speed",1.0))
        #car_control_msg.v= norm_speed
        #if 
        car_control_msg.v = rospy.get_param("speed",0.5)
        if abs(car_control_msg.omega) > rospy.get_param("turn_th",2):
            car_control_msg.v = rospy.get_param("turn_speed",0.2)

        self.log(f"v={car_control_msg.v}, alpha = {alpha:.2f} omega = {car_control_msg.omega:.2f}. Aim: {aim_point[0]:.2f},{aim_point[1]:.2f}")

        #self.log(f"Aim point:"{aim_point})

        self.last_omega = car_control_msg.omega
        self.last_v = car_control_msg.v

        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()