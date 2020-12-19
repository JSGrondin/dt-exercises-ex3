#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg
import debugpy
import os
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage #Image
from geometry_msgs.msg import Point as PointMsg
from duckietown_msgs.msg import SegmentList, Segment, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds
from image_processing.anti_instagram import AntiInstagram
from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge

debugpy.listen(("localhost", 5678))


class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb_det,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )
        

        self.pub_seglist_filtered = rospy.Publisher("~seglist_filtered",
                                                    SegmentList,
                                                    queue_size=1,
                                                    dt_topic_type=TopicType.DEBUG)
                                                    
        self.pub_segmented_img = rospy.Publisher("~debug/segmented_image/compressed",
                                              CompressedImage,
                                              queue_size=1,
                                              dt_topic_type=TopicType.DEBUG)

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file','.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute)
        self.homography = self.load_extrinsics()
        homography = np.array(self.homography).reshape((3, 3))
        self.bridge = CvBridge()
        self.gpg = GroundProjectionGeometry(160,120, homography)
        # self.gpg = GroundProjectionGeometry(320, 240, homography)
        self.initialized = True
        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb_det(self, image_msg):
        if not self.initialized:
            return

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )

        # image = cv2.resize(image, (224,224))
        # img_small = cv2.resize(image, (160,120))
        # self.model_wrapper.segment_cv2_image(img_small)
        # img_small = cv2.resize(image, (160, 120))
        img_reg = cv2.resize(image, (320, 240))
        self.model_wrapper.segment_cv2_image(img_reg)
        seg_img = self.model_wrapper.get_seg()
        yellow_segments_px = self.model_wrapper.get_yellow_segments_px()
        white_segments_px = self.model_wrapper.get_white_segments_px()
        duckie_segments_px = self.model_wrapper.get_duckie_segments_px()
        right_bezier_segments_px = self.model_wrapper.get_right_bezier_px()
        # left_bezier_segments_px = self.model_wrapper.get_left_bezier_px()

        # ground project segments
        yellow_segments = self.ground_project_segments_px(yellow_segments_px)
        white_segments = self.ground_project_segments_px(white_segments_px, right_only=True)
        duckie_segments = self.ground_project_segments_px(duckie_segments_px)
        bezier_segments = self.ground_project_segments_px(right_bezier_segments_px)

        seg_msg = SegmentList()
        seg_msg.header = image_msg.header
        self.add_segments(yellow_segments, seg_msg, Segment.YELLOW)
        self.add_segments(white_segments, seg_msg, Segment.WHITE)

        # no other color besides yellow, white and red, so using red for now, as it is not being used for the moment
        self.add_segments(bezier_segments, seg_msg, Segment.RED)

        self.pub_seglist_filtered.publish(seg_msg)

        rgb = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))

        rgb[(seg_img == 0)] = np.array([0, 0, 0]).astype(int)
        rgb[(seg_img == 1)] = np.array([255, 255, 255]).astype(int)
        rgb[(seg_img == 2)] = np.array([255, 255, 0]).astype(int)
        rgb[(seg_img == 3)] = np.array([255, 0, 0]).astype(int)
        rgb[(seg_img == 4)] = np.array([0, 0, 255]).astype(int)
        rgb[(seg_img == 5)] = np.array([0, 255, 0]).astype(int)

        # segmented_img_cv = cv2.applyColorMap(self.model_wrapper.seg*64, cv2.COLORMAP_JET)

        segmented_img = self.bridge.cv2_to_compressed_imgmsg(rgb)
        segmented_img.header.stamp = image_msg.header.stamp
        self.pub_segmented_img.publish(segmented_img)

        print(f"Found {len(duckie_segments_px)} duckie segments")

        msg = BoolStamped()
        msg.header = image_msg.header
        if len(duckie_segments) == 0:
            # No duckie detection at all!
            msg.data = False
        else:
            msg.data = self.det2bool(duckie_segments, min_num_seg=3, x_lim=0.2, y_lim=0.05)
            if msg.data:
                print("A duckie is facing the bot, let's stop and wait for it to cross")
        self.pub_obj_dets.publish(msg)

    def image_cb_seg(self, image_msg):
        if not self.initialized:
            return

        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images. Experiment to find a good number of skipped
        # images.

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )
        
        #image = cv2.resize(image, (224,224))
        # img_small = cv2.resize(image, (160,120))
        # self.model_wrapper.segment_cv2_image(img_small)
        # img_small = cv2.resize(image, (160, 120))
        img_reg = cv2.resize(image, (320,240))
        self.model_wrapper.segment_cv2_image(img_reg)
        seg_img = self.model_wrapper.get_seg()
        yellow_segments_px = self.model_wrapper.get_yellow_segments_px()
        white_segments_px = self.model_wrapper.get_white_segments_px()
        duckie_segments_px = self.model_wrapper.get_duckie_segments_px()
        right_bezier_segments_px = self.model_wrapper.get_right_bezier_px()
        # left_bezier_segments_px = self.model_wrapper.get_left_bezier_px()

        #ground project segments
        yellow_segments = self.ground_project_segments_px(yellow_segments_px)
        white_segments = self.ground_project_segments_px(white_segments_px, right_only=True)
        duckie_segments = self.ground_project_segments_px(duckie_segments_px)
        bezier_segments = self.ground_project_segments_px(right_bezier_segments_px)

        seg_msg = SegmentList()
        seg_msg.header = image_msg.header
        self.add_segments(yellow_segments, seg_msg, Segment.YELLOW)
        self.add_segments(white_segments, seg_msg, Segment.WHITE)

        # no other color besides yellow, white and red, so using red for now, as it is not being used for the moment
        self.add_segments(bezier_segments, seg_msg, Segment.RED)

        self.pub_seglist_filtered.publish(seg_msg)

        rgb = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))

        rgb[(seg_img == 0)] = np.array([0, 0, 0]).astype(int)
        rgb[(seg_img == 1)] = np.array([255, 255, 255]).astype(int)
        rgb[(seg_img == 2)] = np.array([255, 255, 0]).astype(int)
        rgb[(seg_img == 3)] = np.array([255, 0, 0]).astype(int)
        rgb[(seg_img == 4)] = np.array([0, 0, 255]).astype(int)
        rgb[(seg_img == 5)] = np.array([0, 255, 0]).astype(int)

        # segmented_img_cv = cv2.applyColorMap(self.model_wrapper.seg*64, cv2.COLORMAP_JET)

        segmented_img = self.bridge.cv2_to_compressed_imgmsg(rgb)
        segmented_img.header.stamp = image_msg.header.stamp
        self.pub_segmented_img.publish(segmented_img)

        print(f"Found {len(duckie_segments_px)} duckie segments")

        msg = BoolStamped()
        msg.header = image_msg.header
        if len(duckie_segments)==0:
            #No duckie detection at all!
            msg.data = False
        else:
            msg.data = self.det2bool(duckie_segments, min_num_seg=3, x_lim=0.2, y_lim=0.05)
            if msg.data:
                print("A duckie is facing the bot, let's stop and wait for it to cross")
        self.pub_obj_dets.publish(msg)

    def add_segments(self, yellow_segments, seg_msg, color):
        for yellow_segment in yellow_segments:
            new_segment = Segment()
            ground_pt_msg_1 = PointMsg()
            ground_pt_msg_1.z=0
            ground_pt_msg_1.x=yellow_segment[0][0]
            ground_pt_msg_1.y=yellow_segment[0][1]
            ground_pt_msg_2 = PointMsg()
            ground_pt_msg_2.z=0
            ground_pt_msg_2.x=yellow_segment[1][0]
            ground_pt_msg_2.y=yellow_segment[1][1]
            new_segment.points[0] = ground_pt_msg_1
            new_segment.points[1] = ground_pt_msg_2
            new_segment.color = color
            seg_msg.segments.append(new_segment)
    
        
    def ground_project_segments_px(self, segments_px, right_only=False, xmin=0.1, xmax=0.6):
        x=[]
        y=[]
        segments=[]
        for segment_px in segments_px:
            pixel1 = Point(segment_px[0][0]*2,segment_px[0][1]*2) #Conversion. Points are converted in 640x480 for the homography to work
            pixel2 = Point(segment_px[1][0]*2,segment_px[1][1]*2) #Conversion. Points are converted in 640x480 for the homography to work
            ground_projected_point1 = self.gpg.pixel2ground(pixel1)
            ground_projected_point2 = self.gpg.pixel2ground(pixel2)
            pt1 = (ground_projected_point1.x, ground_projected_point1.y)
            pt2 = (ground_projected_point2.x, ground_projected_point2.y)
            segment = (pt1,pt2)
            if right_only: #For the white line, we assume it is right of the duckie.
                if pt1[1] > 0 or pt2[1] > 0: 
                    continue
            if pt1[0] < xmin or pt2[0] < xmin: #Not to close to the duckiebot.
                continue
            if pt1[0] > xmax or pt2[0] > xmax: #Neither too far!
                continue
            segments.append(segment)
        return segments

    def det2bool(self, duckie_segments, min_num_seg=2, x_lim=0.2, y_lim=0.05):

        count=0
        for duckie_segment in duckie_segments:
            x = 0.5 * (duckie_segment[0][0] + duckie_segment[1][0])
            y = 0.5 * (duckie_segment[0][1] + duckie_segment[1][1])
            if (x <= x_lim and abs(y) <= y_lim):
                count+=1

        if count >= min_num_seg:
            return True
        else:
            return False

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file,'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
