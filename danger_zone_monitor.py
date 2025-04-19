import os
import cv2
import time
import io
import threading
import queue
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import oss2
from datetime import datetime
import logging
import json

# Configuration
OSS_ACCESS_KEY_ID = 'YOUR_OSS_ACCESS_KEY'
OSS_ACCESS_KEY_SECRET = 'YOUR_OSS_ACCESS_KEY_SECRET'
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_BUCKET = 'your-bucket-name'

# Logging Configuration
LOG_FILE = "danger_zone_violations.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("摄像头画面")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create frame for the camera display
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create label for the camera image
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="启动摄像头中...")
        self.camera_label.pack(fill="both", expand=True)
        
        # Image holder
        self.current_image = None
        
        # Drawing mode
        self.drawing_mode = False
        self.drawing_points = []
        self.danger_zone = None
        
        # Flag to indicate if window is closed
        self.is_closed = False
    
    def update_frame(self, img, intrusion_detected=False):
        """Update camera frame with new image"""
        if self.is_closed:
            return
            
        try:
            if img:
                # Convert PIL Image to OpenCV format for drawing
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Draw danger zone if defined
                if self.danger_zone is not None:
                    color = (0, 0, 255) if intrusion_detected else (0, 255, 0)
                    cv2.polylines(img_cv, [np.array(self.danger_zone)], True, color, 2)
                
                # Draw current selection if in drawing mode
                if self.drawing_mode and len(self.drawing_points) > 0:
                    points = np.array(self.drawing_points)
                    cv2.polylines(img_cv, [points], False, (0, 0, 255), 2)
                
                # Convert back to PIL Image
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Resize the image to fit the window nicely
                img_resized = img_pil.copy()
                img_resized.thumbnail((800, 600))
                
                # Convert to CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(800, 600))
                
                # Update the label
                self.camera_label.configure(image=ctk_img, text="")
                
                # Store a reference to prevent garbage collection
                self.current_image = ctk_img
        except Exception as e:
            print(f"Error updating camera frame: {e}")
    
    def enable_drawing_mode(self):
        """Enable drawing mode for danger zone definition"""
        self.drawing_mode = True
        self.drawing_points = []
        self.camera_label.bind("<Button-1>", self.on_mouse_click)
        self.camera_label.bind("<Motion>", self.on_mouse_move)
        self.camera_label.bind("<Button-3>", self.finish_drawing)
    
    def disable_drawing_mode(self):
        """Disable drawing mode"""
        self.drawing_mode = False
        self.camera_label.unbind("<Button-1>")
        self.camera_label.unbind("<Motion>")
        self.camera_label.unbind("<Button-3>")
    
    def on_mouse_click(self, event):
        """Handle mouse click for drawing"""
        if self.drawing_mode:
            x = event.x
            y = event.y
            self.drawing_points.append([x, y])
    
    def on_mouse_move(self, event):
        """Handle mouse movement for drawing"""
        if self.drawing_mode and len(self.drawing_points) > 0:
            # Update the last point to follow the mouse
            self.drawing_points[-1] = [event.x, event.y]
    
    def finish_drawing(self, event):
        """Finish drawing the danger zone"""
        if self.drawing_mode and len(self.drawing_points) >= 3:
            self.danger_zone = self.drawing_points
            self.disable_drawing_mode()
            # Notify the main app that danger zone is defined
            self.master.danger_zone_defined(self.danger_zone)
    
    def on_closing(self):
        """Handle window close event"""
        self.is_closed = True
        self.withdraw()  # Hide instead of destroy to allow reopening

class WebcamHandler:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False
        self.processing = False
        self.cap = None
        self.webcam_thread = None
        self.last_webcam_image = None
        self.debug = True
        
        # Camera window
        self.camera_window = None
        
        # Person detection parameters
        self.person_detector = None  # Initialize your person detection model
    
    def start(self):
        """Start webcam capture process"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.update_status("无法打开摄像头")
                    return False
                
                self.running = True
                
                # Create the camera window
                self.create_camera_window()
                
                # Start processing thread
                self.webcam_thread = threading.Thread(target=self._process_webcam)
                self.webcam_thread.daemon = True
                self.webcam_thread.start()
                
                return True
            except Exception as e:
                self.app.update_status(f"启动摄像头时出错: {e}")
                return False
        return False
    
    def create_camera_window(self):
        """Create a window to display the camera feed"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("摄像头画面")
            # Position the window to the right of the main window
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            self.camera_window.geometry(f"800x600+{main_x + self.app.winfo_width() + 10}+{main_y}")
    
    def stop(self):
        """Stop webcam capture process"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        # Close the camera window
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
    
    def _process_webcam(self):
        """Main webcam processing loop"""
        last_ui_update_time = 0
        ui_update_interval = 0.05  # Update UI at 20 fps
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.app.update_status("无法捕获画面")
                    time.sleep(0.1)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Store the most recent image
                self.last_webcam_image = img
                
                # Check for intrusions if not paused
                intrusion_detected = False
                if not self.paused:
                    intrusion_detected = self._check_intrusion(frame)
                
                # Update camera window with the current frame
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img, intrusion_detected)
                    last_ui_update_time = current_time
                
                time.sleep(0.03)  # ~30 fps for capture
            except Exception as e:
                error_msg = f"摄像头错误: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # Pause before retry
    
    def _check_intrusion(self, frame):
        """Check for intrusions in the danger zone"""
        if not self.camera_window or not self.camera_window.danger_zone:
            return False
        
        try:
            # Detect people in the frame
            people_detected = self._detect_people(frame)
            
            # Check if any detected person is in the danger zone
            for person in people_detected:
                if self._is_in_danger_zone(person):
                    # Save screenshot of the intrusion
                    self._save_intrusion_screenshot(frame)
                    return True
            
            return False
        except Exception as e:
            print(f"入侵检测错误: {e}")
            return False
    
    def _detect_people(self, frame):
        """Detect people in the frame"""
        # TODO: Implement person detection using your preferred model
        # For now, return dummy detections
        return []
    
    def _is_in_danger_zone(self, person):
        """Check if a detected person is in the danger zone"""
        if not self.camera_window or not self.camera_window.danger_zone:
            return False
        
        # TODO: Implement point-in-polygon check for the person's position
        # For now, return a dummy result
        return False
    
    def _save_intrusion_screenshot(self, frame):
        """Save a screenshot of the intrusion event"""
        try:
            # Create screenshots directory if it doesn't exist
            os.makedirs("screenshots", exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/intrusion_{timestamp}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame)
            
            # Notify the main app about the new screenshot
            self.app.new_intrusion_screenshot(filename)
        except Exception as e:
            print(f"保存截图错误: {e}")
    
    def toggle_pause(self):
        """Toggle the paused state of the monitoring"""
        self.paused = not self.paused
        status = "已暂停监控" if self.paused else "已恢复监控"
        self.app.update_status(status)
        print(status)

class DangerZoneMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize system components
        self.setup_ui()
        self.webcam_handler = WebcamHandler(self)
        
        # Danger zone monitoring specific attributes
        self.intrusion_events = []  # Store intrusion event records
        self.danger_zone = None  # Store danger zone boundaries
        self.last_screenshot = None  # Store path to last intrusion screenshot
        
        # Set up key bindings
        self.setup_key_bindings()
        
        # Start webcam after a short delay
        self.after(1000, self.start_webcam)
    
    def setup_ui(self):
        """Initialize the user interface"""
        self.title("危险区域入侵检测系统")
        self.geometry("1200x800")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create status display
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Status labels
        self.zone_status_label = ctk.CTkLabel(self.status_frame, text="危险区域: 未定义", font=("Arial", 14))
        self.zone_status_label.pack(side="left", padx=10)
        
        self.monitoring_status_label = ctk.CTkLabel(self.status_frame, text="监控状态: 未开始", font=("Arial", 14))
        self.monitoring_status_label.pack(side="left", padx=10)
        
        # Create event log
        self.log_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        # Create screenshot display
        self.screenshot_frame = ctk.CTkFrame(self.main_frame)
        self.screenshot_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.screenshot_label = ctk.CTkLabel(self.screenshot_frame, text="最新入侵截图")
        self.screenshot_label.pack(side="left", padx=10)
        
        # Control buttons
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        
        self.set_zone_button = ctk.CTkButton(
            self.control_frame,
            text="设置危险区域",
            command=self.set_danger_zone
        )
        self.set_zone_button.pack(side="left", padx=5)
        
        self.toggle_monitoring_button = ctk.CTkButton(
            self.control_frame,
            text="开始监控",
            command=self.toggle_monitoring
        )
        self.toggle_monitoring_button.pack(side="left", padx=5)
    
    def setup_key_bindings(self):
        """Set up keyboard shortcuts"""
        self.bind("<space>", lambda e: self.toggle_monitoring())
        self.bind("<z>", lambda e: self.set_danger_zone())
    
    def start_webcam(self):
        """Start webcam capture after UI initialization"""
        if not self.webcam_handler.start():
            self.update_status("无法启动摄像头。请检查摄像头连接。")
    
    def set_danger_zone(self):
        """Allow user to define danger zone boundaries"""
        if self.webcam_handler.camera_window:
            # Enable drawing mode in camera window
            self.webcam_handler.camera_window.enable_drawing_mode()
    
    def danger_zone_defined(self, points):
        """Handle danger zone definition completion"""
        self.danger_zone = points
        self.zone_status_label.configure(text="危险区域: 已定义")
        self.update_status("危险区域已定义")
    
    def toggle_monitoring(self):
        """Toggle intrusion monitoring on/off"""
        self.webcam_handler.toggle_pause()
        status = "暂停监控" if self.webcam_handler.paused else "开始监控"
        self.toggle_monitoring_button.configure(text=status)
        self.monitoring_status_label.configure(text=f"监控状态: {status}")
    
    def log_intrusion(self, screenshot_path):
        """Log intrusion event"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create event record
        event = {
            "timestamp": timestamp,
            "screenshot": screenshot_path
        }
        
        # Add to event log
        event_text = f"{timestamp} - 检测到入侵"
        event_label = ctk.CTkLabel(
            self.log_frame,
            text=event_text,
            text_color="red",
            font=("Arial", 12)
        )
        event_label.pack(fill="x", padx=5, pady=2)
        
        # Store event record
        self.intrusion_events.append(event)
        
        # Log to file
        logging.info(event_text)
        
        # Trigger alarm
        self.trigger_alarm()
    
    def new_intrusion_screenshot(self, screenshot_path):
        """Handle new intrusion screenshot"""
        self.last_screenshot = screenshot_path
        
        # Update screenshot display
        try:
            img = Image.open(screenshot_path)
            img.thumbnail((200, 150))  # Resize for display
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            self.screenshot_label.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Error loading screenshot: {e}")
        
        # Log the intrusion
        self.log_intrusion(screenshot_path)
    
    def trigger_alarm(self):
        """Trigger visual and audio alarms for intrusions"""
        # Flash the window
        self.flash_window()
        
        # TODO: Add audio alarm
    
    def flash_window(self):
        """Flash the window to draw attention"""
        original_color = self.cget("fg_color")
        self.configure(fg_color="red")
        self.after(500, lambda: self.configure(fg_color=original_color))
    
    def update_status(self, text):
        """Update the status message"""
        print(text)  # For now, just print to console

def main():
    # Set appearance mode and default theme
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = DangerZoneMonitorApp()
    app.mainloop()

if __name__ == "__main__":
    main() 