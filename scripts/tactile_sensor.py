#!/usr/bin/env python

import argparse
import sys
import rospy
import actionlib
import baxter_interface
import baxter_tools
import tf
from baxter_interface import CHECK_VERSION
from arduino_magnetic_sensor.msg import xServerMsg, xSensorData
from baxter_imp.msg import SensorDisp
import numpy as np
from std_msgs.msg import Bool
import math

CalMeasurements = 50
Treshold = 60.0

halfx = 0.5
halfy = 0.375
TaxelPositions = np.array([[-halfx,halfy],[-halfx,-halfy],[halfx,-halfy],[halfx,halfy]])

class TactileSensor(object):
    def __init__(self, sensor_id, sensor_name):
        self.sensor_id = int(sensor_id)
        self.contactPub = rospy.Publisher(sensor_name+"_response", Bool, queue_size=10)
        self.dispPub = rospy.Publisher(sensor_name+"_displacement", SensorDisp, queue_size=10)
        self.Reset()
        rospy.sleep(5)
    def DetectingMode(self,msg): # Switch detection mode True/False
        self.DetectionMode = msg.data
        print("Sensor",self.sensor_id,": Switched detecting mode to ", msg.data)
    def ResetRequest(self,msg):
        if msg.data:
            print("Sensor",self.sensor_id,"reset requested")
            self.Reset()
    def Reset(self): # Start in calibration mode
        self.Thresh = np.ones((1,4)) * Treshold
        self.MeanValues = None
        self.CalBuffer = None
        self.Ncal = 0
        self.TouchDetected = False
        self.DetectionMode = False
        self.TouchProcessed = False
        self.disp = SensorDisp()
    def Measure(self,msg):
        #print("Measured something... ")
        Measurement = np.empty((4,3)) # 4 sensors 3 axes
        if(msg.sensorid != self.sensor_id):
            return
        #print("Sensor",self.sensor_id,"received data")
        for i in range(4):
            Measurement[i] = np.array([msg.points[i].point.x, msg.points[i].point.y, msg.points[i].point.z]).reshape(1,-1) # Overwrite row
        #print(Measurement)
        
        if self.MeanValues is None:
            self.MeanValues = Measurement # If initially nothing, I puse the received value as "mean" for now until cal is finished
        OffsetMeasurement = Measurement - self.MeanValues
        #print("Measuring",OffsetMeasurement)
        self.Operate(OffsetMeasurement)
        if not self.TouchDetected: #If there's no touch currently, I will help with re-calibration
            #print("No event detected, calibrating")
            if self.Ncal == 0:
                self.CalBuffer = np.zeros((4,3))
            self.CalBuffer += Measurement
            self.Ncal+=1
            if self.Ncal >= CalMeasurements:
                self.MeanValues = self.CalBuffer/self.Ncal
                self.Ncal = 0
                print("Sensor", self.sensor_id,"recalibrated")
                #print("Finished New Calibrationfor sensor N°", self.sensor_id, ", average is ",self.MeanValues)
    def Operate(self,meas):
        LocalContactPoints = np.copy(TaxelPositions)
        ForceMagnitudes = meas[:,2].ravel() # For this experiment (magnitudes) i just keep values of Z force
        if self.DetectionMode:
            print("Force Measurements:", ForceMagnitudes)
        for i in range(4):
            if ForceMagnitudes[i] < Treshold:
                ForceMagnitudes[i] = 0.0
            else:
                ForceMagnitudes[i] = 1.0 # Set the forces that surpassed treshold to 1 (non weighed). if a weiged average position is preferred just remove the "else" statement
        #print("Magnitudes:", ForceMagnitudes)
        
        #ThreshPassed = np.any(np.greater(ForceMagnitudes,self.Thresh))
        ThreshPassed = np.any(ForceMagnitudes>0.0) #Actvates if any taxel surpass treshold
        if ThreshPassed:
            #print("MAGNITUDE SURPASSED THRESH")
            #print("CONTACTS:",LocalContactPoints)
            self.TouchDetected = True
            disp_array = np.dot(ForceMagnitudes.reshape(1,-1), LocalContactPoints) / np.sum(ForceMagnitudes) # Total contact point (weighed by magnitude)
            disp_array = disp_array.ravel()
            
            self.disp.x_disp = disp_array[0]
            self.disp.y_disp = disp_array[1]
            if not self.TouchProcessed and self.DetectionMode:
                #Touch detected for first time
                self.contactPub.publish(True)
                print("Touch Detected in sensor", self.sensor_id,":")
                print("XDISP=",self.disp.x_disp,"YDISP=",self.disp.y_disp)
            self.TouchProcessed = True
        else:
            if self.TouchProcessed: # Means touch stopped happening
                self.Ncal = 0
            self.TouchProcessed = False
            self.TouchDetected = False
        self.dispPub.publish(self.disp)

def main(sensor_id, sensor_location, finger, gripper):
    print("Initializing node...")
    sensor_name = gripper+"_gripper_"+finger+"_finger_"+sensor_location+"_sensor"
    rospy.init_node(sensor_name)
    print("Creating Sensor Class...")
    Sensor = TactileSensor(sensor_id, sensor_name)
    rospy.sleep(2)
    #print("Suscribing... Ctrl-c to quit")
    rospy.Subscriber("xServTopic", xServerMsg, Sensor.Measure)
    rospy.Subscriber(sensor_name+"_activate", Bool, Sensor.DetectingMode)
    rospy.Subscriber(sensor_name+"_reset", Bool, Sensor.ResetRequest)
    print("Sensor", sensor_id, "initialized and subscribed to topics.")
    rospy.spin()

if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) != 5:
        print("usage: tactile_sensor.py sensor_id sensor_location finger gripper")
    else:
        main(myargv[1], myargv[2], myargv[3], myargv[4])
