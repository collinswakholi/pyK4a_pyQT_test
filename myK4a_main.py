import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from app_functions import *
import os
import time
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import gc
import datetime
import argparse
import queue
import shutil

gc.enable()

parser = argparse.ArgumentParser()
parser.add_argument('--working_dir', type=str, default=os.getcwd(), help='Directory to save data to')
parser.add_argument('--folder_name', type=str, default='test_data', help='Name to save data as')
parser.add_argument('--view', type=str, default='Color', help='Default view to show')
parser.add_argument('--ui_file', type=str, default='myK4a_app.ui', help='Name of the UI file to use')
parser.add_argument('--delay', type=int, default=100, help='Delay in milliseconds between frames during data collection')
parser.add_argument('--fmt', type=str, default='.png', help='File format to save images as')
parser.add_argument('--selected_res', type=int, default=pyk4a.ColorResolution.RES_720P, help='Image Resolution from camera')
parser.add_argument('--selected_fps', type=int, default=pyk4a.FPS.FPS_30, help='Frame rate from camera')
parser.add_argument('--selected_color_format', type=int, default=pyk4a.ImageFormat.COLOR_BGRA32, help='Color format from camera')
parser.add_argument('--selected_depth_mode', type=int, default=pyk4a.DepthMode.NFOV_UNBINNED, help='Depth mode from camera')
parser.add_argument('--app_title', type=str, default='K4a Capture App', help='Title of the app')
parser.add_argument('--max_qsize', type=int, default=100, help='Maximum size of the queue for the capture thread')

args = parser.parse_args()

"""
------------------------------------------------------------------------------------------------------------------------
Threads
------------------------------------------------------------------------------------------------------------------------
"""

# create a thread to get frames from the kinect camera, and update the preview window
class streamThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.stopped = False
        self.device = None
        self.window = None
    
    def run(self):
        while not self.stopped:
            self.device = self.window.device
            im = get_frame(self.device, self.window.view)
            self.image_signal.emit(im)
            
            set_status(self.window, "Frames Elapsed: " + str(self.window.counter) + " in the streaming thread")
            
            self.window.counter += 1
            time.sleep(0.0001)
            
    def stop(self):
        self.stopped = True
        self.wait()
        return self
   
# create a thread to get frames from the kinect camera
class captureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.running = True
        self.device = None
        self.window = None
        
    def run(self):
        self.device = self.window.device
        # start high resolution timer
        tic = time.perf_counter()
        while self.running:
            self.window.Dummy = get_recent_frame(self.device, self.window.Dummy)
            self.window.Image_Queue.put(self.window.Dummy) # put data into queue
            # set_status(self.window, "Frames Captured: " + str(self.window.counter) + ", in the capture thread")
            
            # update the preview window every 200ms
            toc = time.perf_counter()
            if (toc - tic) > 0.2:
                self.image_signal.emit(cv2.cvtColor(self.window.Dummy['rgb'], cv2.COLOR_BGR2RGB))
                tic = toc
            
            self.window.counter += 1
            time.sleep(self.window.delay/1000)
            
    
    def stop(self):
        self.running = False
        self.wait()

        return self 

# create a thread to save frames from the kinect camera
class savingThread(QThread):
    def __init__(self):
        super().__init__()
        self.stopped = False
        self.window = None
        self.counter = 0
        
    def run(self):
        max_qsize = self.window.Image_Queue.maxsize # get the max size of the queue
        self.window.progressBar.setRange(0, max_qsize) # set the range of the progress bar
        while not self.stopped:
            try:
                qsize = self.window.Image_Queue.qsize() # check how many elements in queue
                
                if qsize == max_qsize:
                    self.stopped = True
                    self.window.capture_thread.stop()
                    # self.window.Capture_PB.text.setText("Start Capture")
                    set_status(self.window,"Buffer full, capture thread stopped")
                    self.wait()
                    
                    break
                    
                    
                # get the first element in queue
                frame = self.window.Image_Queue.get()

                save_data(data=frame,
                            dir=self.window.saving_dir,
                            name='Img_'+'{:04d}'.format(self.counter),
                            fmt=self.window.fmt)
                
                try:
                    set_status(self.window, "Frames Saved: " + str(self.counter) + ", in the saving thread")
                except:
                    pass
                # update progressBar with qsize
                self.window.progressBar.setValue(qsize) # update the progress bar
                self.counter += 1
                
            except:
                set_status(self.window, "Frame saving error")
            
    def stop(self):
        self.stopped = True
        self.wait()
        return self
    
"""
------------------------------------------------------------------------------------------------------------------------
Main window class
------------------------------------------------------------------------------------------------------------------------
"""

class MyWindow(QtWidgets.QMainWindow):
    
    def __init__(self, args):
        super().__init__()
        
        self.counter = 0
        self.stream_thread = None
        self.capture_thread = None
        self.saving_thread = None
        self.cam_connected = False
        self.preview_on = False
        self.device = None
        self.ui_file = args.ui_file
        self.device_serial_number = ""
        self.view = args.view
        self.folder_name = args.folder_name
        self.delay = args.delay
        self.working_dir = args.working_dir
        self.saving_dir = ""
        self.saving_defaults = {
            "working_dir": self.working_dir,
            "delay": self.delay,
            "folder_name": self.folder_name
        }
        self.fmt = args.fmt
        self.im_W = 0
        self.im_H = 0
        self.Dummy = None
        self.Image_Queue = queue.Queue(maxsize=args.max_qsize)
        self.capturing = False
        self.saving = False

        self.selected_res = args.selected_res
        self.selected_depth = args.selected_depth_mode
        self.selected_color_format = args.selected_color_format
        self.selected_fps = args.selected_fps

        self.config = Config(
            color_resolution=self.selected_res,
            depth_mode=self.selected_depth,
            color_format=self.selected_color_format,
            camera_fps=self.selected_fps)

        self.resolutions = [i.name for i in pyk4a.ColorResolution]
        self.resolutions.remove("OFF")
        self.depth_modes = [i.name for i in pyk4a.DepthMode]
        self.depth_modes.remove("OFF")
        self.depth_modes.remove("PASSIVE_IR")
        self.color_formats = [i.name for i in pyk4a.ImageFormat]
        self.color_formats = [i for i in self.color_formats if "COLOR" in i]
        self.fps_modes = [i.name for i in pyk4a.FPS]


        # load the UI file
        uic.loadUi(self.ui_file, self)
        self.Save_Folder_Edit.setText(self.folder_name)
        self.setWindowTitle("K4a Capture App")
        # set initial values
        self = init_window(self)
        self.Delay_Edit.setText(str(self.delay))
        set_treeView(self, pwd)

        # initialize the events
        self.Exit_PB.clicked.connect(self.on_exit)
        self.Select_Saving_Dir_PB.clicked.connect(self.on_select_saving_dir)
        self.actionConnect.triggered.connect(self.on_connect)
        self.actionDisconnect.triggered.connect(self.on_disconnect)

        self.menuResolution.triggered.connect(self.on_resolution)
        self.menuDepth_Mode.triggered.connect(self.on_depth_mode)
        self.menuColor_Format.triggered.connect(self.on_color_format)
        self.menuFPS.triggered.connect(self.on_fps)
        self.actionReset_2.triggered.connect(self.on_reset_2)

        self.actionSaving_Dir.triggered.connect(self.on_select_saving_dir)
        self.menuSaving_Format.triggered.connect(self.on_saving_format)

        self.SelectView_Combo.currentIndexChanged.connect(self.on_select_view)
        self.Preview_PB.clicked.connect(self.on_preview)
        self.AutoName_Radio.clicked.connect(self.on_auto_name)
        self.Reset_PB.clicked.connect(self.on_reset)
        self.Set_PB.clicked.connect(self.on_set)
        self.Capture_PB.clicked.connect(self.on_capture)

    
    def empty_queue(self):
        while not self.Image_Queue.empty():
            self.Image_Queue = queue.Queue().queue.clear()
            self.Image_Queue = queue.Queue(maxsize=args.max_qsize)
        return self
    
    @pyqtSlot(np.ndarray)
    def update_image(self, im):
        self.im_H, self.im_W, channels = im.shape
        bytesPerLine = channels * self.im_W
        qImg = QtGui.QImage(im.data, self.im_W, self.im_H, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.Image_Pane.setPixmap(QtGui.QPixmap.fromImage(qImg))
        self.Image_Pane.setScaledContents(True)
        self.Image_Pane.resize(self.scrollArea.size())
        self.Image_Pane.update()
        return self
    
    # @pyqtSlot(int)
    # def capture_update(self, msg):
        # create a message box posting the message
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(str(msg))
        return self
    
    def disconnect_device(self):
        if self.preview_on:
            self.stop_preview()
        try:
            self.device.stop()
            self.device = None
            set_status(self, "Closing the Device... ")
            self.cam_connected = False
        except:
            pass
        
        return self
        
    def connect_device(self):
        # try:
        if self.device is None:
            self.device = PyK4A(config=self.config)
            self.device.start()
            self.device.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH)
            
            self.device_serial_number = self.device.serial
            set_status(self, "Connected to device: " + self.device_serial_number)
            self.cam_connected = True
        else:
            set_status(self, "Device is already connected, SN: " + self.device_serial_number)
        # except:
        #     set_status(self, "No device is Found")
        return self
        
    def update_config(self, restart=False):
        # stop preview
        if self.preview_on:
            self.stop_preview()
            
        if restart:
            self.disconnect_device()
        
        if self.selected_fps > pyk4a.FPS.FPS_15 & (self.selected_res >pyk4a.ColorResolution.RES_2160P | self.selected_depth > pyk4a.DepthMode.WFOV_2X2BINNED):
                self.selected_fps = pyk4a.FPS.FPS_15
            
        self.config = Config(color_resolution=self.selected_res,
                             depth_mode=self.selected_depth,
                             color_format=self.selected_color_format,
                             camera_fps=self.selected_fps)
        # update the check marks on the menu
        # resolution
        for i in self.menuResolution.actions():
            if i.text() == self.selected_res.name:
                i.setChecked(True)
            else:
                i.setChecked(False)
                
        # depth mode
        for i in self.menuDepth_Mode.actions():
            if i.text() == self.selected_depth.name:
                i.setChecked(True)
            else:
                i.setChecked(False)
                
        # color format
        for i in self.menuColor_Format.actions():
            if i.text() == self.selected_color_format.name:
                i.setChecked(True)
            else:
                i.setChecked(False)
                
        # fps
        for i in self.menuFPS.actions():
            if i.text() == self.selected_fps.name:
                i.setChecked(True)
            else:
                i.setChecked(False)
        
        # update the device config with out restarting the device
        if not restart:
            self.device._config = self.config
            
        if restart:
            self.connect_device()
        
        if self.preview_on:
            self.start_preview()
        
        return self
        
    def start_preview(self):
        if self.device is not None:
            if self.stream_thread is None:
                self.counter = 0
                self.stream_thread = streamThread()
                self.stream_thread.window = self
                self.stream_thread.image_signal.connect(self.update_image)
                self.stream_thread.start()
                self.preview_on = True
                self.Preview_PB.setText("Stop Preview")
            else:
                set_status(self, "Preview is already running")
        else:
            set_status(self, "No device is connected")
            
        return self
            
    def stop_preview(self):
        if self.preview_on:
            self.preview_on = False
            self.stream_thread.stop()
            time.sleep(0.5)
            self.stream_thread = None
            self.Preview_PB.setText("Preview")
        else:
            set_status(self, "Preview is not running")
            
        return self
    
    def on_exit(self):
        # do something when Exit_PB is clicked
        self.disconnect_device()
        self.update()
        time.sleep(0.1)
        
        # delete data and queue
        self.Dummy = None
        self.Image_Queue = None
        
        # delete the stream thread
        self.stream_thread = None
        self.saving_thread = None
        self.capture_thread = None
        
        close_app()
        
    def on_select_saving_dir(self):
        # do something when Select_Saving_Dir_PB is clicked
        self.working_dir = select_folder(self)  
        return self
    
    def on_saving_format(self):
        checked_actions = []
        for checked_action in self.menuSaving_Format.actions():
            if checked_action.isChecked()& (checked_action.text() != self.fmt):
                checked_actions.append(checked_action)
                
        checked_action = None
        if len(checked_actions) == 1:
            checked_action = checked_actions[0]
        
        if checked_action is not None:
            self.fmt = checked_action.text()
            set_status(self, "Saving format is changed to: " + self.fmt)
            
            for i in self.menuSaving_Format.actions():
                if i.text() == self.fmt:
                    i.setChecked(True)
                else:
                    i.setChecked(False)
                
        return self
    
    def on_connect(self):
        
        self.connect_device()
        
        # check if the device is connected
        if self.device is not None:
        
            self.menuResolution.clear()
            for i in self.resolutions:
                self.menuResolution.addAction(i)
                action = self.menuResolution.actions()[-1]
                action.setCheckable(True)
                if i == 'RES_720P':
                    action.setChecked(True)
       
            self.menuDepth_Mode.clear()
            for i in self.depth_modes:
                self.menuDepth_Mode.addAction(i)
                action = self.menuDepth_Mode.actions()[-1]
                action.setCheckable(True)
                if i == 'NFOV_UNBINNED':
                    action.setChecked(True)
                    
            self.menuColor_Format.clear()
            for i in self.color_formats:
                self.menuColor_Format.addAction(i)
                action = self.menuColor_Format.actions()[-1]
                action.setCheckable(True)
                if i == 'COLOR_BGRA32':
                    action.setChecked(True)
                    
            self.menuFPS.clear()
            for i in self.fps_modes:
                self.menuFPS.addAction(i)
                action = self.menuFPS.actions()[-1]
                action.setCheckable(True)
                if i == 'FPS_30':
                    action.setChecked(True)
            
            
            # enable the disconnect button
            self.actionDisconnect.setEnabled(True)
            self.menuSettings.setEnabled(True)
            self.SelectView_Combo.setEnabled(True)
            self.Preview_PB.setEnabled(True)
            self.actionConnect.setEnabled(False)
            
            self.cam_connected = True
        else:
            # device is not connected, print in the status bar
            set_status(self, "Could NOT find any K4a device! Please check the connection!") 
            self.cam_connected = False 
            
        return self
    
    def on_disconnect(self):
        
        self.disconnect_device()
        set_status(self, "Disconnected from device: "+self.device_serial_number)
        self.actionDisconnect.setEnabled(False)
        self.menuSettings.setEnabled(False)
        self.SelectView_Combo.setEnabled(False)
        self.Preview_PB.setEnabled(False)
        self.groupBox.setEnabled(False)
        self.Capture_PB.setEnabled(False)
        
        self.actionConnect.setEnabled(True)
        
        self.cam_connected = False
        
        return self

    def on_resolution(self):
        
        checked_actions = []
        for action in self.menuResolution.actions():
            if action.isChecked() & (self.selected_res != eval('pyk4a.ColorResolution.'+action.text())):
                checked_actions.append(action)
        
        checked_action = None
        if len(checked_actions) == 1:
            checked_action = checked_actions[0]
        
        if checked_action is not None:
            res = checked_action.text()
        
            set_status(self, "Resolution set to: " + res)
            
            for i in self.menuResolution.actions():
                if i.text() != res:
                    i.setChecked(False)
            
            self.selected_res = eval('pyk4a.ColorResolution.'+res)
            self.update_config(restart=True)
            
        return self
        
    def on_depth_mode(self):
        checked_actions = []
        for action in self.menuDepth_Mode.actions():
            if action.isChecked() & (self.selected_depth != eval('pyk4a.DepthMode.'+action.text())):
                checked_actions.append(action)
                
        checked_action = None
        if len(checked_actions) == 1:
            checked_action = checked_actions[0]
        
        if checked_action is not None:
            depth_mode = checked_action.text()
            set_status(self, "Depth mode set to: " + depth_mode)
            for i in self.menuDepth_Mode.actions():
                if i.text() != depth_mode:
                    i.setChecked(False)
            self.selected_depth = eval('pyk4a.DepthMode.'+depth_mode)
            self.update_config(restart=True)
            
        return self
    
    def on_color_format(self):
        checked_actions = []
        for action in self.menuColor_Format.actions():
            if action.isChecked() & (self.selected_color_format != eval('pyk4a.ImageFormat.'+action.text())):
                checked_actions.append(action)
        
        checked_action = None
        if len(checked_actions) == 1:
            checked_action = checked_actions[0]
            
        if checked_action is not None:
            color_format = checked_action.text()
            set_status(self, "Color format set to: " + color_format)
            for i in self.menuColor_Format.actions():
                if i.text() != color_format:
                    i.setChecked(False)
            self.selected_color_format = eval('pyk4a.ImageFormat.'+color_format)
            self.update_config(restart=True)
            
        return self
            
    def on_fps(self):
        checked_actions = []
        for action in self.menuFPS.actions():
            if action.isChecked() & (self.selected_fps != eval('pyk4a.FPS.'+action.text())):
                checked_actions.append(action)
        
        checked_action = None
        if len(checked_actions) == 1:
            checked_action = checked_actions[0]
            
        if checked_action is not None:
            fps = checked_action.text()
            set_status(self, "FPS set to: " + fps)
            for i in self.menuFPS.actions():
                if i.text() != fps:
                    i.setChecked(False)
            self.selected_fps = eval('pyk4a.FPS.'+fps)
            self.update_config(restart=True)    
            
        return self
            
    def on_reset_2(self):
        self.selected_color_format = pyk4a.ImageFormat.COLOR_BGRA32
        self.selected_depth = pyk4a.DepthMode.NFOV_UNBINNED
        self.selected_res = pyk4a.ColorResolution.RES_720P
        self.selected_fps = pyk4a.FPS.FPS_30
        
        self.update_config(restart=True)
        
        # check the default settings in the menu bar
        for i in self.menuResolution.actions():
            if i.text() == 'RES_720P':
                i.setChecked(True)
            else:
                i.setChecked(False)
        
        for i in self.menuDepth_Mode.actions():
            if i.text() == 'NFOV_UNBINNED':
                i.setChecked(True)
            else:
                i.setChecked(False)
        
        for i in self.menuColor_Format.actions():
            if i.text() == 'COLOR_BGRA32':
                i.setChecked(True)
            else:
                i.setChecked(False)
                
        for i in self.menuFPS.actions():
            if i.text() == 'FPS_30':
                i.setChecked(True)
            else:
                i.setChecked(False)
                
        set_status(self, "Camera settings reset to default!")
        
        return self

    def on_select_view(self):
        # get the selected view
        self.view = self.SelectView_Combo.currentText()
        self.counter = 0
        set_status(self, "Selected view: " + self.view)
        
        return self
        
    def on_preview(self):
        
        PB_text = self.Preview_PB.text()
        
        if PB_text == "Preview":
            
            if self.cam_connected == False:
                set_status(self, "No camera connected!")
                return
            
            self.start_preview()
            
            set_status(self, "Preview started!")
            
            self.Preview_PB.setText("Stop Preview")
            self.preview_on = True
            self.groupBox.setEnabled(True)
            
        elif PB_text == "Stop Preview":
            self.stop_preview()
            self.Preview_PB.setText("Preview")
            
        return self
        
    def on_auto_name(self):
        if self.AutoName_Radio.isChecked():
            self.Save_Folder_Edit.setEnabled(False)
            self.folder_name = datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S") # create a folder name based on the current date and time (YYYY-MM-DD_HH-MM-SS-MS)
            self.Save_Folder_Edit.setText(self.folder_name)
            self.saving_dir = os.path.join(self.working_dir, self.folder_name)
            set_status(self, "Folder name set to: " + self.folder_name)
            time.sleep(0.5)
        else:
            self.Save_Folder_Edit.setEnabled(True)
            
        return self
    
    def on_set(self):
        self.folder_name = self.Save_Folder_Edit.text()
        self.delay = int(self.Delay_Edit.text())
        
        if self.selected_fps == pyk4a.FPS.FPS_5:
            min_delay = 220
        elif self.selected_fps == pyk4a.FPS.FPS_15:
            min_delay = 75
        elif self.selected_fps == pyk4a.FPS.FPS_30:
            min_delay = 40
        
        if self.delay < min_delay:
            # throw a warning, with 2 buttons: "Continue" and "Cancel"
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Delay time is too short, this may cause problems with the camera!\n\n Value will be set to: " + str(min_delay) + " ms")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
            # change the text on cancel button to "Overrride"
            msg.button(QtWidgets.QMessageBox.Cancel).setText('Override')
            msg.setDefaultButton(QtWidgets.QMessageBox.Ok)
            
            x = msg.exec_()
            
            if x == QtWidgets.QMessageBox.Ok:
                self.delay = min_delay
                self.Delay_Edit.setText(str(self.delay))
            elif x == QtWidgets.QMessageBox.Cancel:
                pass
        
        self.saving_dir = os.path.join(self.working_dir, self.folder_name)
        
        # create a dummy
        img_size = (self.im_H, self.im_W, 3)
        dummy_C = np.zeros(img_size, dtype=np.uint8)
        dummy_D = np.zeros((self.im_H, self.im_W), dtype=np.uint16)
        dummy_IR = np.zeros((self.im_H, self.im_W), dtype=np.uint16)
        
        self.Dummy = {'rgb': dummy_C, 'depth': dummy_D, 'ir': dummy_IR}
        
        set_status(self, "Variables have been set!")
        
        self.Capture_PB.setEnabled(True)
        
        return self
    
    def on_reset(self):
        self.folder_name = self.saving_defaults['folder_name']
        self.delay = self.saving_defaults['delay']
        self.working_dir = self.saving_defaults['working_dir']
        
        # update the values in the GUI
        self.Save_Folder_Edit.setText(self.folder_name)
        self.Delay_Edit.setText(str(self.delay))
        set_treeView(self, self.working_dir)
        
        img_size = (self.im_H, self.im_W, 3)
        dummy_C = np.zeros(img_size, dtype=np.uint8)
        dummy_D = np.zeros((self.im_H, self.im_W), dtype=np.uint16)
        dummy_IR = np.zeros((self.im_H, self.im_W), dtype=np.uint16)
        
        self.Dummy = {'color': dummy_C, 'depth': dummy_D, 'ir': dummy_IR}
        
        set_status(self, "Saving Parameters have been reset!")
        
        self.Capture_PB.setEnabled(True)
        
        return self
    
    def on_capture(self):
        
        PB_text = self.Capture_PB.text()
        
        if PB_text=="Start Capture":
            # set button text to pause capture
            self.Capture_PB.setText("Stop Capturing")
            # stop the preview if it is running
            if self.preview_on == True:
                self.stop_preview()
                self.Preview_PB.setText("Preview")
                self.Preview_PB.setEnabled(False)
                
            # create folder to save the data
            if os.path.exists(self.saving_dir):
                shutil.rmtree(self.saving_dir)
                
            os.makedirs(self.saving_dir)
            os.makedirs(os.path.join(self.saving_dir, 'rgb'))
            os.makedirs(os.path.join(self.saving_dir, 'depth'))
            os.makedirs(os.path.join(self.saving_dir, 'ir'))
                
            set_status(self, "Saving Directory created: " + self.saving_dir)
            
            
            # start the saving thread
            self.saving = True
            self.saving_thread = savingThread()
            self.saving_thread.window = self
            self.saving_thread.start()
            
            # start the capturing thread
            self.capturing = True
            self.capture_thread = captureThread()
            self.capture_thread.window = self
            self.capture_thread.image_signal.connect(self.update_image)
            self.capture_thread.start()
            
            set_status(self, "Capturing data...")
        else:
            self.capturing = False
            self.saving = False
            
            self.saving_thread.stop()
            self.capture_thread.stop()
            
            self.empty_queue()
            
            self.saving_thread = None
            self.capture_thread = None
            
            # reset the progress bar
            self.progressBar.setValue(0)
    
            self.Capture_PB.setText("Start Capture")
            self.Preview_PB.setEnabled(True)
            self.Preview_PB.setText("Preview")
            set_status(self, "Capture stopped!")
            time.sleep(2)
            
            # restart the preview
            self.start_preview()
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    
    window = MyWindow(args)
    
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
    gc.collect()
    