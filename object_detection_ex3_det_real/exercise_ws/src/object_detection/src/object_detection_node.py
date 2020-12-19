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


def add_boxes(obs, boxes, classes, scores):
    color_dict = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (150, 255, 150)}
    class_dict = {1: 'duckie', 2: 'cone', 3: 'truck', 4: 'bus'}
    tl = round(0.002 * (224+224) / 2) + 1

# t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
    for i, val in enumerate(boxes):
        label = '%s %.2f' % (class_dict[classes[i]], scores[i])
        cv2.rectangle(obs, (val[0], val[1]), (val[2], val[3]), color_dict[classes[i]], 2)
        cv2.putText(obs, label, (val[0] + 3, val[1] - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)
    return obs


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
        

        # self.pub_seglist_filtered = rospy.Publisher("~seglist_filtered",
        #                                             SegmentList,
        #                                             queue_size=1,
        #                                             dt_topic_type=TopicType.DEBUG)
                                                    
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

        img_reg = cv2.resize(image, (224, 224))
        img_rgb = cv2.cvtColor(img_reg, cv2.COLOR_BGR2RGB)
        boxes, classes, scores = self.model_wrapper.predict(img_rgb)
        boxes, classes, scores = boxes[0], classes[0], scores[0]
        if type(boxes) != type(None):
            img_w_boxes = add_boxes(img_reg, boxes, classes, scores)
        else:
            img_w_boxes = img_reg
        detection_img = self.bridge.cv2_to_compressed_imgmsg(img_w_boxes)
        detection_img.header.stamp = image_msg.header.stamp
        self.pub_segmented_img.publish(detection_img)

        msg = BoolStamped()
        msg.header = image_msg.header
        msg.data = self.det2bool(boxes, classes)  # [0] because our batch size given to the wrapper is 1

        self.pub_obj_dets.publish(msg)

        #
        # msg = BoolStamped()
        # msg.header = image_msg.header
        # if len(duckie_segments) == 0:
        #     # No duckie detection at all!
        #     msg.data = False
        # else:
        #     msg.data = self.det2bool(duckie_segments, min_num_seg=3, x_lim=0.2, y_lim=0.05)
        #     if msg.data:
        #         print("A duckie is facing the bot, let's stop and wait for it to cross")
        # self.pub_obj_dets.publish(msg)


    def det2bool(self, boxes, classes):
        if type(boxes) != type(None):
            for i in range(len(boxes)):
                if classes[i] != 1: #everything except duckie is not important for now
                    continue
                else:
                    x1, y1, x2, y2 = boxes[i]
                    centroid_x = 0.5 * (x1+x2)
                    centroid_y = 0.5 * (y1+y2)
                    if 224 >= centroid_x >= 0.5*224: # in the bottow 50% of the image
                        if 0.75 * 224 >= centroid_y >= 0.25*224: #in the middle third of the image (horizontal)
                            if abs((x2-x1) * (y2-y1)) >= 700:
                                print("duckie detected")
                                return True

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
