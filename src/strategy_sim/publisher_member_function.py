import os
import sys

import rclpy
from booster_interface.msg import LowState, MotorState, Odometer
from localization_msgs.msg import LocalizationResult
from rclpy.node import Node

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
from setting import CONVERT_SCALE, ObjectLabel, convert_objectlabel_to_str
from vision_interface.msg import DetectedObject, Detections


class WorldStatePublisher(Node):
    def __init__(self, is_enemy=False):
        super().__init__("world_state_publisher")
        self.odom_publisher_1 = self.create_publisher(Odometer, "/odometer_state_1", 1)
        self.odom_publisher_2 = self.create_publisher(Odometer, "/odometer_state_2", 1)
        self.odom_publisher_3 = self.create_publisher(Odometer, "/odometer_state_3", 1)
        self.lowstate_publisher_1 = self.create_publisher(LowState, "/low_state_1", 1)
        self.lowstate_publisher_2 = self.create_publisher(LowState, "/low_state_2", 1)
        self.lowstate_publisher_3 = self.create_publisher(LowState, "/low_state_3", 1)
        self.selfpos_publisher_1 = self.create_publisher(
            LocalizationResult, "/self_position_1", 1
        )
        self.selfpos_publisher_2 = self.create_publisher(
            LocalizationResult, "/self_position_2", 1
        )
        self.selfpos_publisher_3 = self.create_publisher(
            LocalizationResult, "/self_position_3", 1
        )
        self.true_selfpos_publisher_1 = self.create_publisher(
            LocalizationResult, "/true_state/self_position_1", 1
        )
        self.true_selfpos_publisher_2 = self.create_publisher(
            LocalizationResult, "/true_state/self_position_2", 1
        )
        self.true_selfpos_publisher_3 = self.create_publisher(
            LocalizationResult, "/true_state/self_position_3", 1
        )
        self.object_pos_publisher_1 = self.create_publisher(
            Detections, "/booster_vision/detection_1", 1
        )
        self.object_pos_publisher_2 = self.create_publisher(
            Detections, "/booster_vision/detection_2", 1
        )
        self.object_pos_publisher_3 = self.create_publisher(
            Detections, "/booster_vision/detection_3", 1
        )
        self.true_object_pos_publisher_1 = self.create_publisher(
            Detections, "/true_state/booster_vision/detection_1", 1
        )
        self.true_object_pos_publisher_2 = self.create_publisher(
            Detections, "/true_state/booster_vision/detection_2", 1
        )
        self.true_object_pos_publisher_3 = self.create_publisher(
            Detections, "/true_state/booster_vision/detection_3", 1
        )
        self.odom_publisher_4 = self.create_publisher(Odometer, "/odometer_state_4", 1)
        self.odom_publisher_5 = self.create_publisher(Odometer, "/odometer_state_5", 1)
        self.odom_publisher_6 = self.create_publisher(Odometer, "/odometer_state_6", 1)
        self.lowstate_publisher_4 = self.create_publisher(LowState, "/low_state_4", 1)
        self.lowstate_publisher_5 = self.create_publisher(LowState, "/low_state_5", 1)
        self.lowstate_publisher_6 = self.create_publisher(LowState, "/low_state_6", 1)
        self.selfpos_publisher_4 = self.create_publisher(
            LocalizationResult, "/self_position_4", 1
        )
        self.selfpos_publisher_5 = self.create_publisher(
            LocalizationResult, "/self_position_5", 1
        )
        self.selfpos_publisher_6 = self.create_publisher(
            LocalizationResult, "/self_position_6", 1
        )
        self.object_pos_publisher_4 = self.create_publisher(
            Detections, "/booster_vision/detection_4", 1
        )
        self.object_pos_publisher_5 = self.create_publisher(
            Detections, "/booster_vision/detection_5", 1
        )
        self.object_pos_publisher_6 = self.create_publisher(
            Detections, "/booster_vision/detection_6", 1
        )

        self.current_object_poses = [
            Detections() for _ in range(6)
        ]  # Initialize for 6 robots

        self.true_current_object_poses = [
            Detections() for _ in range(6)
        ]  # Initialize for 6 robots

    def pub_odometer(self, x, y, theta, label=1):
        odom = Odometer()
        odom.x = x / 100.0  # odomはm単位なので修正
        odom.y = y / 100.0  # odomはm単位なので修正
        odom.theta = theta
        if label == 1:
            self.odom_publisher_1.publish(odom)
        elif label == 2:
            self.odom_publisher_2.publish(odom)
        elif label == 3:
            self.odom_publisher_3.publish(odom)
        elif label == 4:
            self.odom_publisher_4.publish(odom)
        elif label == 5:
            self.odom_publisher_5.publish(odom)
        elif label == 6:
            self.odom_publisher_6.publish(odom)

    def pub_neck_yaw(self, neck_yaw, neck_pitch, label=1):
        state = LowState()
        state.imu_state.rpy[0] = 0.0  # r
        state.imu_state.rpy[1] = 0.0  # p
        state.imu_state.rpy[2] = 0.0  # y
        motor = MotorState()
        motor.q = neck_yaw
        state.motor_state_serial.append(motor)
        motor.q = neck_pitch
        state.motor_state_serial.append(motor)
        if label == 1:
            self.lowstate_publisher_1.publish(state)
        elif label == 2:
            self.lowstate_publisher_2.publish(state)
        elif label == 3:
            self.lowstate_publisher_3.publish(state)
        elif label == 4:
            self.lowstate_publisher_4.publish(state)
        elif label == 5:
            self.lowstate_publisher_5.publish(state)
        elif label == 6:
            self.lowstate_publisher_6.publish(state)

    def pub_selfpos(self, x, y, theta, target_robot=1):
        """
        Publish the self position of the robot.
        """
        selfpos = LocalizationResult()
        selfpos.pose.x = x * CONVERT_SCALE
        selfpos.pose.y = y * CONVERT_SCALE
        selfpos.pose.theta = theta
        selfpos.confidence = 1.0
        selfpos.convergence = True  # シミュレーションなので、常に収束していると仮定
        selfpos.error_code = 0  # シミュレーションなので、常にエラーコードは0
        if target_robot == 1:
            self.selfpos_publisher_1.publish(selfpos)
        elif target_robot == 2:
            self.selfpos_publisher_2.publish(selfpos)
        elif target_robot == 3:
            self.selfpos_publisher_3.publish(selfpos)
        elif target_robot == 4:
            self.selfpos_publisher_4.publish(selfpos)
        elif target_robot == 5:
            self.selfpos_publisher_5.publish(selfpos)
        elif target_robot == 6:
            self.selfpos_publisher_6.publish(selfpos)

    def pub_true_selfpos(self, x, y, theta, target_robot=1):
        """
        自己位置の真値をパブリッシュする
        """
        selfpos = LocalizationResult()
        selfpos.pose.x = x * CONVERT_SCALE
        selfpos.pose.y = y * CONVERT_SCALE
        selfpos.pose.theta = theta
        selfpos.confidence = 1.0
        selfpos.convergence = True  # シミュレーションなので、常に収束していると仮定
        selfpos.error_code = 0  # シミュレーションなので、常にエラーコードは0
        if target_robot == 1:
            self.true_selfpos_publisher_1.publish(selfpos)
        elif target_robot == 2:
            self.true_selfpos_publisher_2.publish(selfpos)
        elif target_robot == 3:
            self.true_selfpos_publisher_3.publish(selfpos)

    def add_object_pos(
        self, label: ObjectLabel, x: float, y: float, target_robot: int = 1
    ):
        obj = DetectedObject()
        obj.label = convert_objectlabel_to_str(label)
        if label == ObjectLabel.LABEL_ROBOT:
            obj.color = "blue"  # 味方ロボットとして登録
        elif label == ObjectLabel.LABEL_OPPENENT:
            obj.color = "red"  # 敵ロボットとして登録
        else:
            obj.color = "None"
        if label in [
            ObjectLabel.LABEL_TCROSS,
            ObjectLabel.LABEL_XCROSS,
            ObjectLabel.LABEL_LCROSS,
            ObjectLabel.LABEL_PENALTYPOINT,
        ]:
            obj.position.append(0.0)  # なんかよく知らんがこれは3つ無いといけないらしい
            obj.position.append(0.0)  # なんかよく知らんがこれは3つ無いといけないらしい
            obj.position.append(0.0)  # なんかよく知らんがこれは3つ無いといけないらしい
        obj.position_projection.append(x * CONVERT_SCALE)
        obj.position_projection.append(y * CONVERT_SCALE)
        obj.confidence = 1.0  # シミュレーションなので、常に確信度は1.0
        self.current_object_poses[target_robot - 1].detected_objects.append(obj)

    def add_true_object_pos(
        self, label: ObjectLabel, x: float, y: float, target_robot: int = 1
    ):
        obj = DetectedObject()
        obj.label = convert_objectlabel_to_str(label)
        obj.position_projection.append(x * CONVERT_SCALE)
        obj.position_projection.append(y * CONVERT_SCALE)
        obj.confidence = 1.0  # シミュレーションなので、常に確信度は1.0
        self.true_current_object_poses[target_robot - 1].detected_objects.append(obj)

    def add_DetectedObject(self, detected_object, target_robot: int = 1):
        """
        Add a DetectedObject to the current detections.
        """
        if isinstance(detected_object, DetectedObject):
            self.current_object_poses[target_robot - 1].detected_objects.append(
                detected_object
            )

    def publish_object_pos(self):
        for i, detections in enumerate(self.current_object_poses):
            if detections is not None:
                if i == 0:
                    self.object_pos_publisher_1.publish(detections)
                    self.current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
                elif i == 1:
                    self.object_pos_publisher_2.publish(detections)
                    self.current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
                elif i == 2:
                    self.object_pos_publisher_3.publish(detections)
                    self.current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
                elif i == 3:
                    self.object_pos_publisher_4.publish(detections)
                    self.current_object_poses[i] = Detections()
                    # Reset after publishing
                elif i == 4:
                    self.object_pos_publisher_5.publish(detections)
                    self.current_object_poses[i] = Detections()
                    # Reset after publishing
                elif i == 5:
                    self.object_pos_publisher_6.publish(detections)
                    self.current_object_poses[i] = Detections()
                    # Reset after publishing

    def publish_true_object_pos(self):
        for i, detections in enumerate(self.true_current_object_poses):
            if detections is not None:
                if i == 0:
                    self.true_object_pos_publisher_1.publish(detections)
                    self.true_current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
                elif i == 1:
                    self.true_object_pos_publisher_2.publish(detections)
                    self.true_current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
                elif i == 2:
                    self.true_object_pos_publisher_3.publish(detections)
                    self.true_current_object_poses[i] = (
                        Detections()
                    )  # Reset after publishing
