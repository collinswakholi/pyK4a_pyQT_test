# Description: This file contains functions for the app defined in myK4a_app2.ui
# 2021-03-01

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
import sys
import cv2
import time


desc_text = "This app is designed for collecting image data from the Azure Kinect DK camera." \
    "\n"\
    " It supports capturing color, depth, and IR data, and provides live preview for easy visualization."\
    "\n"\
    "The user can save the captured data to a folder of their choice. " \
    "\n\n"\
    "This app is built using Python 3.11.3, PyQt5, and the pyk4a Python Azure Kinect SDK. "\
    "\n\n"\
    "For any questions or issues, check out the code on GitHub at https://github.com/collinswakholi."\

app_version = "K4a Capture App \n\n"\
    "Version = 0.1.0"\

pwd = os.getcwd()


"""
------------------------------------------------------------------------------------------------------------------------
Preliminary functions
------------------------------------------------------------------------------------------------------------------------
"""

def set_treeView(window, directory):
    model = QtWidgets.QFileSystemModel()
    model.setRootPath(directory)
    window.treeView.setModel(model)
    window.treeView.setRootIndex(model.index(directory))

def set_status(window, status_text):
    window.Status_TXT.setText(status_text)
    
def colorize_depth(depth_image):
    depth_image = cv2.convertScaleAbs(depth_image, alpha=0.05)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_TURBO)# cv2.COLORMAP_JET, cv2.COLORMAP_TURBO
    return depth_image

def get_frame(myK4a, view):
    capture = myK4a.get_capture()
    if capture.color is not None and capture.depth is not None:
        
        if view == 'Color':
            color_image = capture.color
            im = color_image
        elif view == 'Depth':
            depth = capture.transformed_depth
            depth = colorize_depth(depth)
            im = depth
        elif view == 'IR':
            ir_image = capture.transformed_ir
            ir_image = cv2.convertScaleAbs(ir_image, alpha=0.10)
            im = ir_image
        else:
            im = color_image
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
 
def get_recent_frame(myK4a, dummy):
    capture = myK4a.get_capture()
    if capture.color is not None and capture.depth is not None:
            dummy['rgb']=capture.color
            dummy['depth']=cv2.convertScaleAbs(capture.transformed_depth, alpha=0.05)
            dummy['ir']=cv2.convertScaleAbs(capture.transformed_ir, alpha=0.10)
            
    return dummy

def save_data(data, dir, name, fmt):
    
    cv2.imwrite(os.path.join(dir, 'rgb', name + fmt), data['rgb'])
    cv2.imwrite(os.path.join(dir, 'ir', name + fmt), data['ir'])
    cv2.imwrite(os.path.join(dir, 'depth', name + fmt), data['depth'])
    
"""
------------------------------------------------------------------------------------------------------------------------
Main functions
------------------------------------------------------------------------------------------------------------------------
"""

def init_window(window):
    window.actionMore_Information.triggered.connect(lambda: QtWidgets.QMessageBox.about(window, "About", desc_text)) # show app description when clicked
    window.actionVersion.triggered.connect(lambda: QtWidgets.QMessageBox.about(window, "Version", app_version)) # show app version when clicked
    
    # disable buttons
    window.menuSettings.setEnabled(False)
    window.actionDisconnect.setEnabled(False)
    window.SelectView_Combo.setEnabled(False)
    window.Preview_PB.setEnabled(False)
    window.groupBox.setEnabled(False)
    window.Capture_PB.setEnabled(False)
    
    # set progressBar to 0
    window.progressBar.setValue(0)

    
    # change background color
    window.Status_TXT.setStyleSheet("background-color: #fdf5da")
    window.Delay_Edit.setStyleSheet("background-color: #e5f4f4")
    window.Save_Folder_Edit.setStyleSheet("background-color: #e5f4f4")
    window.Exit_PB.setStyleSheet("background-color: #e9bcbb")
    
    # set default values
    window.Delay_Edit.setAlignment(QtCore.Qt.AlignRight) # align text to the right
    window.SelectView_Combo.addItem("Color")
    window.SelectView_Combo.addItem("Depth")
    window.SelectView_Combo.addItem("IR")
    
    return window
    
       
def close_app():
    # variables to delete, processess to kill, etc.
    sys.exit()
    
def select_folder(window):
    folder_path = QFileDialog.getExistingDirectory(window, "Select Directory")
    set_treeView(window, folder_path)
    set_status(window, "Saving directory: " + folder_path)
    
    return folder_path
