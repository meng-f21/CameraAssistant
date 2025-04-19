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
from safety_monitor import SafetyMonitoringApp
from danger_zone_monitor import DangerZoneMonitorApp

# 配置
OSS_ACCESS_KEY_ID = 'YOUR_OSS_ACCESS_KEY'
OSS_ACCESS_KEY_SECRET = 'YOUR_OSS_ACCESS_KEY_SECRET'
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_BUCKET = 'your-bucket-name'

# 日志配置
SAFETY_LOG_FILE = "safety_violations.log"
INTRUSION_LOG_FILE = "danger_zone_violations.log"

# 配置安全违规日志
safety_logger = logging.getLogger('safety')
safety_logger.setLevel(logging.INFO)
safety_handler = logging.FileHandler(SAFETY_LOG_FILE)
safety_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
safety_logger.addHandler(safety_handler)

# 配置入侵检测日志
intrusion_logger = logging.getLogger('intrusion')
intrusion_logger.setLevel(logging.INFO)
intrusion_handler = logging.FileHandler(INTRUSION_LOG_FILE)
intrusion_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
intrusion_logger.addHandler(intrusion_handler)

class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("摄像头画面")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建摄像头显示框架
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建摄像头图像标签
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="启动摄像头中...")
        self.camera_label.pack(fill="both", expand=True)
        
        # 图像持有者
        self.current_image = None
        
        # 绘图模式
        self.drawing_mode = False
        self.drawing_points = []
        self.work_zone = None
        self.danger_zone = None
        
        # 绘图类型
        self.drawing_type = None  # 'work_zone' 或 'danger_zone'
        
        # 标记窗口是否关闭
        self.is_closed = False
    
    def update_frame(self, img, safety_violations=None, intrusion_detected=False):
        """更新摄像头画面"""
        if self.is_closed:
            return
            
        try:
            if img:
                # 将PIL图像转换为OpenCV格式以便绘图
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # 绘制工作区域（如果已定义）
                if self.work_zone is not None:
                    cv2.polylines(img_cv, [np.array(self.work_zone)], True, (0, 255, 0), 2)
                
                # 绘制危险区域（如果已定义）
                if self.danger_zone is not None:
                    color = (0, 0, 255) if intrusion_detected else (255, 0, 0)
                    cv2.polylines(img_cv, [np.array(self.danger_zone)], True, color, 2)
                
                # 如果在绘图模式下，绘制当前选择
                if self.drawing_mode and len(self.drawing_points) > 0:
                    points = np.array(self.drawing_points)
                    color = (0, 255, 0) if self.drawing_type == 'work_zone' else (255, 0, 0)
                    cv2.polylines(img_cv, [points], False, color, 2)
                
                # 转换回PIL图像
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # 调整图像大小以适应窗口
                img_resized = img_pil.copy()
                img_resized.thumbnail((800, 600))
                
                # 转换为CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(800, 600))
                
                # 更新标签
                self.camera_label.configure(image=ctk_img, text="")
                
                # 存储引用以防止垃圾回收
                self.current_image = ctk_img
        except Exception as e:
            print(f"更新摄像头画面时出错: {e}")
    
    def enable_drawing_mode(self, zone_type):
        """启用绘图模式以定义区域"""
        self.drawing_mode = True
        self.drawing_points = []
        self.drawing_type = zone_type
        self.camera_label.bind("<Button-1>", self.on_mouse_click)
        self.camera_label.bind("<Motion>", self.on_mouse_move)
        self.camera_label.bind("<Button-3>", self.finish_drawing)
    
    def disable_drawing_mode(self):
        """禁用绘图模式"""
        self.drawing_mode = False
        self.camera_label.unbind("<Button-1>")
        self.camera_label.unbind("<Motion>")
        self.camera_label.unbind("<Button-3>")
    
    def on_mouse_click(self, event):
        """处理鼠标点击绘图"""
        if self.drawing_mode:
            x = event.x
            y = event.y
            self.drawing_points.append([x, y])
    
    def on_mouse_move(self, event):
        """处理鼠标移动绘图"""
        if self.drawing_mode and len(self.drawing_points) > 0:
            # 更新最后一个点以跟随鼠标
            self.drawing_points[-1] = [event.x, event.y]
    
    def finish_drawing(self, event):
        """完成区域绘制"""
        if self.drawing_mode and len(self.drawing_points) >= 3:
            if self.drawing_type == 'work_zone':
                self.work_zone = self.drawing_points
                self.disable_drawing_mode()
                # 通知主应用程序工作区域已定义
                self.master.work_zone_defined(self.work_zone)
            elif self.drawing_type == 'danger_zone':
                self.danger_zone = self.drawing_points
                self.disable_drawing_mode()
                # 通知主应用程序危险区域已定义
                self.master.danger_zone_defined(self.danger_zone)
    
    def on_closing(self):
        """处理窗口关闭事件"""
        self.is_closed = True
        self.withdraw()  # 隐藏而不是销毁，以允许重新打开

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
        
        # 摄像头窗口
        self.camera_window = None
        
        # 安全检测参数
        self.hard_hat_detector = None  # 初始化安全帽检测模型
        self.protective_clothing_detector = None  # 初始化防护服检测模型
        
        # 人员检测参数
        self.person_detector = None  # 初始化人员检测模型
    
    def start(self):
        """启动摄像头捕获过程"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.update_status("无法打开摄像头")
                    return False
                
                self.running = True
                
                # 创建摄像头窗口
                self.create_camera_window()
                
                # 启动处理线程
                self.webcam_thread = threading.Thread(target=self._process_webcam)
                self.webcam_thread.daemon = True
                self.webcam_thread.start()
                
                return True
            except Exception as e:
                self.app.update_status(f"启动摄像头时出错: {e}")
                return False
        return False
    
    def create_camera_window(self):
        """创建显示摄像头画面的窗口"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("摄像头画面")
            # 将窗口定位在主窗口右侧
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            self.camera_window.geometry(f"800x600+{main_x + self.app.winfo_width() + 10}+{main_y}")
    
    def stop(self):
        """停止摄像头捕获过程"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        # 关闭摄像头窗口
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
    
    def _process_webcam(self):
        """主摄像头处理循环"""
        last_ui_update_time = 0
        ui_update_interval = 0.05  # 以20 fps更新UI
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.app.update_status("无法捕获画面")
                    time.sleep(0.1)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 存储最近的图像
                self.last_webcam_image = img
                
                # 如果未暂停，执行安全检查
                safety_violations = None
                if not self.paused:
                    safety_violations = self._check_safety(frame)
                
                # 检查入侵（如果未暂停）
                intrusion_detected = False
                if not self.paused:
                    intrusion_detected = self._check_intrusion(frame)
                
                # 更新摄像头窗口
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img, safety_violations, intrusion_detected)
                    last_ui_update_time = current_time
                
                time.sleep(0.03)  # 约30 fps捕获
            except Exception as e:
                error_msg = f"摄像头错误: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # 重试前暂停
    
    def _check_safety(self, frame):
        """检查安全装备和工作区域合规性"""
        try:
            # 检测安全帽
            hard_hat_detected = self._detect_hard_hat(frame)
            
            # 检测防护服
            protective_clothing_detected = self._detect_protective_clothing(frame)
            
            # 检查人员是否在工作区域内
            in_work_zone = self._check_work_zone(frame)
            
            # 更新安全状态
            self.app.update_safety_status(
                hard_hat_detected,
                protective_clothing_detected,
                in_work_zone
            )
            
            # 返回违规情况
            violations = {
                "hard_hat": not hard_hat_detected,
                "protective_clothing": not protective_clothing_detected,
                "work_zone": not in_work_zone
            }
            
            # 如果有任何违规，记录违规
            if any(violations.values()):
                self._save_safety_violation_screenshot(frame, violations)
            
            return violations
        except Exception as e:
            print(f"安全检查错误: {e}")
            return None
    
    def _detect_hard_hat(self, frame):
        """检测人员是否佩戴安全帽"""
        # TODO: 使用您首选的模型实现安全帽检测
        # 目前返回一个虚拟结果
        return True
    
    def _detect_protective_clothing(self, frame):
        """检测人员是否穿着防护服"""
        # TODO: 使用您首选的模型实现防护服检测
        # 目前返回一个虚拟结果
        return True
    
    def _check_work_zone(self, frame):
        """检查人员是否在定义的工作区域内"""
        if not self.camera_window or not self.camera_window.work_zone:
            return True  # 如果未定义工作区域，则认为始终在区域内
        
        # TODO: 实现人员检测和工作区域检查
        # 目前返回一个虚拟结果
        return True
    
    def _check_intrusion(self, frame):
        """检查危险区域内的入侵"""
        if not self.camera_window or not self.camera_window.danger_zone:
            return False
        
        try:
            # 检测画面中的人员
            people_detected = self._detect_people(frame)
            
            # 检查是否有任何检测到的人员在危险区域内
            for person in people_detected:
                if self._is_in_danger_zone(person):
                    # 保存入侵截图
                    self._save_intrusion_screenshot(frame)
                    return True
            
            return False
        except Exception as e:
            print(f"入侵检测错误: {e}")
            return False
    
    def _detect_people(self, frame):
        """检测画面中的人员"""
        # TODO: 使用您首选的模型实现人员检测
        # 目前返回虚拟检测结果
        return []
    
    def _is_in_danger_zone(self, person):
        """检查检测到的人员是否在危险区域内"""
        if not self.camera_window or not self.camera_window.danger_zone:
            return False
        
        # TODO: 实现人员位置的点内多边形检查
        # 目前返回一个虚拟结果
        return False
    
    def _save_safety_violation_screenshot(self, frame, violations):
        """保存安全违规截图"""
        try:
            # 如果不存在，创建截图目录
            os.makedirs("screenshots", exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/safety_violation_{timestamp}.jpg"
            
            # 保存图像
            cv2.imwrite(filename, frame)
            
            # 通知主应用程序有关新截图
            self.app.new_safety_violation_screenshot(filename, violations)
        except Exception as e:
            print(f"保存安全违规截图错误: {e}")
    
    def _save_intrusion_screenshot(self, frame):
        """保存入侵事件截图"""
        try:
            # 如果不存在，创建截图目录
            os.makedirs("screenshots", exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/intrusion_{timestamp}.jpg"
            
            # 保存图像
            cv2.imwrite(filename, frame)
            
            # 通知主应用程序有关新截图
            self.app.new_intrusion_screenshot(filename)
        except Exception as e:
            print(f"保存入侵截图错误: {e}")
    
    def toggle_pause(self):
        """切换监控的暂停状态"""
        self.paused = not self.paused
        status = "已暂停监控" if self.paused else "已恢复监控"
        self.app.update_status(status)
        print(status)

class IntegratedSafetyMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 初始化系统组件
        self.setup_ui()
        self.webcam_handler = WebcamHandler(self)
        
        # 安全监控特定属性
        self.safety_violations = []  # 存储安全违规记录
        self.work_zone = None  # 存储工作区域边界
        self.safety_equipment_status = {
            "hard_hat": False,
            "protective_clothing": False
        }
        
        # 危险区域监控特定属性
        self.intrusion_events = []  # 存储入侵事件记录
        self.danger_zone = None  # 存储危险区域边界
        self.last_safety_screenshot = None  # 存储最近的安全违规截图路径
        self.last_intrusion_screenshot = None  # 存储最近的入侵截图路径
        
        # 设置按键绑定
        self.setup_key_bindings()
        
        # 短暂延迟后启动摄像头
        self.after(1000, self.start_webcam)
    
    def setup_ui(self):
        """初始化用户界面"""
        self.title("集成安全监控系统")
        self.geometry("1200x800")
        
        # 配置网格
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 创建主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # 创建状态显示
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # 安全装备状态
        self.hard_hat_label = ctk.CTkLabel(self.status_frame, text="安全帽: 未检测", font=("Arial", 14))
        self.hard_hat_label.pack(side="left", padx=10)
        
        self.protective_clothing_label = ctk.CTkLabel(self.status_frame, text="防护服: 未检测", font=("Arial", 14))
        self.protective_clothing_label.pack(side="left", padx=10)
        
        self.work_zone_label = ctk.CTkLabel(self.status_frame, text="作业区域: 未检测", font=("Arial", 14))
        self.work_zone_label.pack(side="left", padx=10)
        
        # 危险区域状态
        self.danger_zone_label = ctk.CTkLabel(self.status_frame, text="危险区域: 未定义", font=("Arial", 14))
        self.danger_zone_label.pack(side="left", padx=10)
        
        self.monitoring_status_label = ctk.CTkLabel(self.status_frame, text="监控状态: 未开始", font=("Arial", 14))
        self.monitoring_status_label.pack(side="left", padx=10)
        
        # 创建选项卡视图
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.tabview.grid_columnconfigure(0, weight=1)
        self.tabview.grid_rowconfigure(0, weight=1)
        
        # 创建安全违规选项卡
        self.safety_tab = self.tabview.add("安全装备监控")
        self.safety_tab.grid_columnconfigure(0, weight=1)
        self.safety_tab.grid_rowconfigure(0, weight=1)
        
        # 创建入侵检测选项卡
        self.intrusion_tab = self.tabview.add("危险区域监控")
        self.intrusion_tab.grid_columnconfigure(0, weight=1)
        self.intrusion_tab.grid_rowconfigure(0, weight=1)
        
        # 创建安全违规日志
        self.safety_log_frame = ctk.CTkScrollableFrame(self.safety_tab)
        self.safety_log_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        self.safety_log_frame.grid_columnconfigure(0, weight=1)
        
        # 创建安全违规截图显示
        self.safety_screenshot_frame = ctk.CTkFrame(self.safety_tab)
        self.safety_screenshot_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.safety_screenshot_label = ctk.CTkLabel(self.safety_screenshot_frame, text="最新安全违规截图")
        self.safety_screenshot_label.pack(side="left", padx=10)
        
        # 创建入侵检测日志
        self.intrusion_log_frame = ctk.CTkScrollableFrame(self.intrusion_tab)
        self.intrusion_log_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        self.intrusion_log_frame.grid_columnconfigure(0, weight=1)
        
        # 创建入侵检测截图显示
        self.intrusion_screenshot_frame = ctk.CTkFrame(self.intrusion_tab)
        self.intrusion_screenshot_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.intrusion_screenshot_label = ctk.CTkLabel(self.intrusion_screenshot_frame, text="最新入侵截图")
        self.intrusion_screenshot_label.pack(side="left", padx=10)
        
        # 控制按钮
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.set_work_zone_button = ctk.CTkButton(
            self.control_frame,
            text="设置作业区域",
            command=self.set_work_zone
        )
        self.set_work_zone_button.pack(side="left", padx=5)
        
        self.set_danger_zone_button = ctk.CTkButton(
            self.control_frame,
            text="设置危险区域",
            command=self.set_danger_zone
        )
        self.set_danger_zone_button.pack(side="left", padx=5)
        
        self.toggle_monitoring_button = ctk.CTkButton(
            self.control_frame,
            text="开始监控",
            command=self.toggle_monitoring
        )
        self.toggle_monitoring_button.pack(side="left", padx=5)
    
    def setup_key_bindings(self):
        """设置键盘快捷键"""
        self.bind("<space>", lambda e: self.toggle_monitoring())
        self.bind("<w>", lambda e: self.set_work_zone())
        self.bind("<d>", lambda e: self.set_danger_zone())
    
    def start_webcam(self):
        """UI初始化后启动摄像头捕获"""
        if not self.webcam_handler.start():
            self.update_status("无法启动摄像头。请检查摄像头连接。")
    
    def set_work_zone(self):
        """允许用户定义工作区域边界"""
        if self.webcam_handler.camera_window:
            # 在摄像头窗口中启用绘图模式
            self.webcam_handler.camera_window.enable_drawing_mode('work_zone')
    
    def set_danger_zone(self):
        """允许用户定义危险区域边界"""
        if self.webcam_handler.camera_window:
            # 在摄像头窗口中启用绘图模式
            self.webcam_handler.camera_window.enable_drawing_mode('danger_zone')
    
    def work_zone_defined(self, points):
        """处理工作区域定义完成"""
        self.work_zone = points
        self.work_zone_label.configure(text="作业区域: 已定义")
        self.update_status("作业区域已定义")
    
    def danger_zone_defined(self, points):
        """处理危险区域定义完成"""
        self.danger_zone = points
        self.danger_zone_label.configure(text="危险区域: 已定义")
        self.update_status("危险区域已定义")
    
    def toggle_monitoring(self):
        """切换安全监控开/关"""
        self.webcam_handler.toggle_pause()
        status = "暂停监控" if self.webcam_handler.paused else "开始监控"
        self.toggle_monitoring_button.configure(text=status)
        self.monitoring_status_label.configure(text=f"监控状态: {status}")
    
    def update_safety_status(self, hard_hat_detected, protective_clothing_detected, in_work_zone):
        """更新安全装备和工作区域状态"""
        self.safety_equipment_status["hard_hat"] = hard_hat_detected
        self.safety_equipment_status["protective_clothing"] = protective_clothing_detected
        
        # 更新UI标签
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
    
    def log_safety_violation(self, violations, screenshot_path):
        """记录安全违规"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violation_types = []
        
        if violations["hard_hat"]:
            violation_types.append("未佩戴安全帽")
        if violations["protective_clothing"]:
            violation_types.append("未穿戴防护服")
        if violations["work_zone"]:
            violation_types.append("不在作业区域内")
        
        violation_text = f"{timestamp} - 违规: {', '.join(violation_types)}"
        
        # 添加到违规日志
        violation_label = ctk.CTkLabel(
            self.safety_log_frame,
            text=violation_text,
            text_color="red",
            font=("Arial", 12)
        )
        violation_label.pack(fill="x", padx=5, pady=2)
        
        # 存储违规记录
        self.safety_violations.append({
            "timestamp": timestamp,
            "violations": violation_types,
            "screenshot": screenshot_path
        })
        
        # 记录到文件
        safety_logger.info(violation_text)
        
        # 触发警报
        self.trigger_alarm("safety", violation_types)
    
    def log_intrusion(self, screenshot_path):
        """记录入侵事件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建事件记录
        event = {
            "timestamp": timestamp,
            "screenshot": screenshot_path
        }
        
        # 添加到事件日志
        event_text = f"{timestamp} - 检测到入侵"
        event_label = ctk.CTkLabel(
            self.intrusion_log_frame,
            text=event_text,
            text_color="red",
            font=("Arial", 12)
        )
        event_label.pack(fill="x", padx=5, pady=2)
        
        # 存储事件记录
        self.intrusion_events.append(event)
        
        # 记录到文件
        intrusion_logger.info(event_text)
        
        # 触发警报
        self.trigger_alarm("intrusion")
    
    def new_safety_violation_screenshot(self, screenshot_path, violations):
        """处理新的安全违规截图"""
        self.last_safety_screenshot = screenshot_path
        
        # 更新截图显示
        try:
            img = Image.open(screenshot_path)
            img.thumbnail((200, 150))  # 调整大小以便显示
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            self.safety_screenshot_label.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"加载安全违规截图错误: {e}")
        
        # 记录违规
        self.log_safety_violation(violations, screenshot_path)
    
    def new_intrusion_screenshot(self, screenshot_path):
        """处理新的入侵截图"""
        self.last_intrusion_screenshot = screenshot_path
        
        # 更新截图显示
        try:
            img = Image.open(screenshot_path)
            img.thumbnail((200, 150))  # 调整大小以便显示
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            self.intrusion_screenshot_label.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"加载入侵截图错误: {e}")
        
        # 记录入侵
        self.log_intrusion(screenshot_path)
    
    def trigger_alarm(self, alarm_type, details=None):
        """触发视觉和音频警报"""
        # 闪烁窗口
        self.flash_window()
        
        # 根据警报类型显示不同的消息
        if alarm_type == "safety":
            message = f"安全违规警报: {', '.join(details)}"
        else:  # intrusion
            message = "危险区域入侵警报"
        
        self.update_status(message)
        
        # TODO: 添加音频警报
    
    def flash_window(self):
        """闪烁窗口以引起注意"""
        original_color = self.cget("fg_color")
        self.configure(fg_color="red")
        self.after(500, lambda: self.configure(fg_color=original_color))
    
    def update_status(self, text):
        """更新状态消息"""
        print(text)  # 目前只是打印到控制台

def main():
    # 设置外观模式和默认主题
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = IntegratedSafetyMonitorApp()
    app.mainloop()

if __name__ == "__main__":
    main() 