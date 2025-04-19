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

# Configuration
OSS_ACCESS_KEY_ID = 'YOUR_OSS_ACCESS_KEY'
OSS_ACCESS_KEY_SECRET = 'YOUR_OSS_ACCESS_KEY_SECRET'
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_BUCKET = 'your-bucket-name'

# Logging Configuration
LOG_FILE = "safety_violations.log"
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
        self.work_zone = None
        
        # Flag to indicate if window is closed
        self.is_closed = False
    
    def update_frame(self, img):
        """Update camera frame with new image"""
        if self.is_closed:
            return
            
        try:
            if img:
                # Convert PIL Image to OpenCV format for drawing
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Draw work zone if defined
                if self.work_zone is not None:
                    cv2.polylines(img_cv, [np.array(self.work_zone)], True, (0, 255, 0), 2)
                
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
        """Enable drawing mode for work zone definition"""
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
        """Finish drawing the work zone"""
        if self.drawing_mode and len(self.drawing_points) >= 3:
            self.work_zone = self.drawing_points
            self.disable_drawing_mode()
            # Notify the main app that work zone is defined
            self.master.work_zone_defined(self.work_zone)
    
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
        
        # Safety detection parameters
        self.hard_hat_detector = None  # Initialize your hard hat detection model
        self.protective_clothing_detector = None  # Initialize your protective clothing detection model
    
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
                
                # Update camera window with the current frame
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img)
                    last_ui_update_time = current_time
                
                # Perform safety checks if not paused
                if not self.paused:
                    self._check_safety(frame)
                
                time.sleep(0.03)  # ~30 fps for capture
            except Exception as e:
                error_msg = f"摄像头错误: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # Pause before retry
    
    def _check_safety(self, frame):
        """Check for safety equipment and work zone compliance"""
        try:
            # Detect hard hat
            hard_hat_detected = self._detect_hard_hat(frame)
            
            # Detect protective clothing
            protective_clothing_detected = self._detect_protective_clothing(frame)
            
            # Check if person is in work zone
            in_work_zone = self._check_work_zone(frame)
            
            # Update safety status
            self.app.update_safety_status(
                hard_hat_detected,
                protective_clothing_detected,
                in_work_zone
            )
            
        except Exception as e:
            print(f"安全检查错误: {e}")
    
    def _detect_hard_hat(self, frame):
        """Detect if person is wearing a hard hat"""
        # TODO: Implement hard hat detection using your preferred model
        # For now, return a dummy result
        return True
    
    def _detect_protective_clothing(self, frame):
        """Detect if person is wearing protective clothing"""
        # TODO: Implement protective clothing detection using your preferred model
        # For now, return a dummy result
        return True
    
    def _check_work_zone(self, frame):
        """Check if person is within the defined work zone"""
        if not self.camera_window or not self.camera_window.work_zone:
            return True  # If no work zone defined, consider always in zone
        
        # TODO: Implement person detection and work zone checking
        # For now, return a dummy result
        return True
    
    def toggle_pause(self):
        """Toggle the paused state of the monitoring"""
        self.paused = not self.paused
        status = "已暂停监控" if self.paused else "已恢复监控"
        self.app.update_status(status)
        print(status)

class SafetyMonitoringApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize system components
        self.setup_ui()
        self.webcam_handler = WebcamHandler(self)
        
        # Safety monitoring specific attributes
        self.safety_violations = []  # Store safety violation records
        self.work_zone = None  # Store work zone boundaries
        self.safety_equipment_status = {
            "hard_hat": False,
            "protective_clothing": False
        }
        
        # Set up key bindings
        self.setup_key_bindings()
        
        # Start webcam after a short delay
        self.after(1000, self.start_webcam)
        
        # Start timestamp check
        self.check_timestamp()
    
    def setup_ui(self):
        """Initialize the user interface"""
        self.title("安全作业监控系统")
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
        
        # Safety equipment status
        self.hard_hat_label = ctk.CTkLabel(self.status_frame, text="安全帽: 未检测", font=("Arial", 14))
        self.hard_hat_label.pack(side="left", padx=10)
        
        self.protective_clothing_label = ctk.CTkLabel(self.status_frame, text="防护服: 未检测", font=("Arial", 14))
        self.protective_clothing_label.pack(side="left", padx=10)
        
        self.work_zone_label = ctk.CTkLabel(self.status_frame, text="作业区域: 未检测", font=("Arial", 14))
        self.work_zone_label.pack(side="left", padx=10)
        
        # Create violation log
        self.log_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        # Control buttons
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.set_zone_button = ctk.CTkButton(
            self.control_frame,
            text="设置作业区域",
            command=self.set_work_zone
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
        self.bind("<z>", lambda e: self.set_work_zone())
    
    def start_webcam(self):
        """Start webcam capture after UI initialization"""
        if not self.webcam_handler.start():
            self.update_status("无法启动摄像头。请检查摄像头连接。")
    
    def set_work_zone(self):
        """Allow user to define work zone boundaries"""
        if self.webcam_handler.camera_window:
            # Enable drawing mode in camera window
            self.webcam_handler.camera_window.enable_drawing_mode()
    
    def work_zone_defined(self, points):
        """Handle work zone definition completion"""
        self.work_zone = points
        self.update_status("作业区域已定义")
    
    def toggle_monitoring(self):
        """Toggle safety monitoring on/off"""
        self.webcam_handler.toggle_pause()
        status = "暂停监控" if self.webcam_handler.paused else "开始监控"
        self.toggle_monitoring_button.configure(text=status)
    
    def update_safety_status(self, hard_hat_detected, protective_clothing_detected, in_work_zone):
        """Update safety equipment and work zone status"""
        self.safety_equipment_status["hard_hat"] = hard_hat_detected
        self.safety_equipment_status["protective_clothing"] = protective_clothing_detected
        
        # Update UI labels
        self.hard_hat_label.configure(
            text=f"安全帽: {'已佩戴' if hard_hat_detected else '未佩戴'}",
            text_color="green" if hard_hat_detected else "red"
        )
        
        self.protective_clothing_label.configure(
            text=f"防护服: {'已穿戴' if protective_clothing_detected else '未穿戴'}",
            text_color="green" if protective_clothing_detected else "red"
        )
        
        self.work_zone_label.configure(
            text=f"作业区域: {'在区域内' if in_work_zone else '在区域外'}",
            text_color="green" if in_work_zone else "red"
        )
        
        # Log violations if any
        if not (hard_hat_detected and protective_clothing_detected and in_work_zone):
            self.log_violation(hard_hat_detected, protective_clothing_detected, in_work_zone)
    
    def log_violation(self, hard_hat_detected, protective_clothing_detected, in_work_zone):
        """Log safety violations"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violations = []
        
        if not hard_hat_detected:
            violations.append("未佩戴安全帽")
        if not protective_clothing_detected:
            violations.append("未穿戴防护服")
        if not in_work_zone:
            violations.append("不在作业区域内")
        
        violation_text = f"{timestamp} - 违规: {', '.join(violations)}"
        
        # Add to violation log
        violation_label = ctk.CTkLabel(
            self.log_frame,
            text=violation_text,
            text_color="red",
            font=("Arial", 12)
        )
        violation_label.pack(fill="x", padx=5, pady=2)
        
        # Store violation record
        self.safety_violations.append({
            "timestamp": timestamp,
            "violations": violations,
            "hard_hat": hard_hat_detected,
            "protective_clothing": protective_clothing_detected,
            "in_work_zone": in_work_zone
        })
        
        # Log to file
        logging.info(violation_text)
        
        # Trigger alarm
        self.trigger_alarm(violations)
    
    def trigger_alarm(self, violations):
        """Trigger visual and audio alarms for violations"""
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
    
    app = SafetyMonitoringApp()
    app.mainloop()

if __name__ == "__main__":
    main() 