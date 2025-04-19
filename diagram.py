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
from datetime import datetime, timedelta
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from openai import OpenAI
import matplotlib.font_manager as fm

# ---------------- 配置参数 ----------------
# OSS配置
OSS_ACCESS_KEY_ID = 'xxxxxx'
OSS_ACCESS_KEY_SECRET = 'xxxxxx'
OSS_ENDPOINT = 'xxxxxx'
OSS_BUCKET = 'xxxxxx'

# Qwen-VL API配置
QWEN_API_KEY = "xxxxxx"
QWEN_BASE_URL = "xxxxxx"

# 日志配置
LOG_FILE = "behavior_logg.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置中文字体支持
# 尝试加载系统默认中文字体
try:
    # 尝试常见中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    chinese_font = None
    
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path):
                chinese_font = font_name
                break
        except:
            continue
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    else:
        # 如果没有找到中文字体，使用默认字体并记录警告
        print("警告：未找到中文字体，某些文本可能显示不正确")
        
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# ---------------- API客户端初始化 ----------------
# Qwen-VL客户端
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

# ---------------- 工具函数 ----------------
def extract_behavior_type(analysis_text):
    """从AI分析文本中提取行为类型编号"""
    # 尝试在文本中查找行为类型编号(1-7)
    pattern = r'(\d+)\s*[.、:]?\s*(认真专注工作|吃东西|用杯子喝水|喝饮料|玩手机|睡觉|其他)'
    match = re.search(pattern, analysis_text)
    
    if match:
        behavior_num = match.group(1)
        behavior_desc = match.group(2)
        return behavior_num, behavior_desc
    
    # 如果第一种模式失败，尝试替代模式
    patterns = [
        (r'认真专注工作', '1'),
        (r'吃东西', '2'),
        (r'用杯子喝水', '3'),
        (r'喝饮料', '4'),
        (r'玩手机', '5'),
        (r'睡觉', '6'),
        (r'其他', '7')
    ]
    
    for pattern, num in patterns:
        if re.search(pattern, analysis_text):
            return num, pattern
    
    return "0", "未识别"  # 如果没有匹配项，返回默认值

# ---------------- 摄像头显示窗口 ----------------
class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("摄像头视图")
        self.geometry("640x480")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.configure(fg_color="#1a1a1a")  # 深色背景
        
        # 创建摄像头显示框架
        self.camera_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建摄像头图像标签
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="正在启动摄像头...", text_color="white")
        self.camera_label.pack(fill="both", expand=True)
        
        # 图像保存器
        self.current_image = None
        
        # 标记窗口是否关闭
        self.is_closed = False
    
    def update_frame(self, img):
        """更新摄像头帧"""
        if self.is_closed:
            return
            
        try:
            if img:
                # 调整图像大小以适应窗口
                img_resized = img.copy()
                img_resized.thumbnail((640, 480))
                
                # 转换为CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(640, 480))
                
                # 更新标签
                self.camera_label.configure(image=ctk_img, text="")
                
                # 保存引用以防止垃圾回收
                self.current_image = ctk_img
        except Exception as e:
            print(f"更新摄像头帧出错: {e}")
    
    def on_closing(self):
        """处理窗口关闭事件"""
        self.is_closed = True
        self.withdraw()  # 隐藏而不是销毁，以便重新打开

# ---------------- 行为可视化类 ----------------
class BehaviorVisualizer:
    """处理检测到的行为的可视化"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.behavior_map = {
            "1": "专注工作",
            "2": "吃东西",
            "3": "喝水",
            "4": "喝饮料",
            "5": "玩手机",
            "6": "睡觉",
            "7": "其他"
        }
        
        # 不同行为的颜色（确保两个图表中的颜色一致）
        self.behavior_colors = {
            "1": "#4CAF50",  # 绿色表示工作
            "2": "#FFC107",  # 琥珀色表示吃东西
            "3": "#2196F3",  # 蓝色表示喝水
            "4": "#9C27B0",  # 紫色表示喝饮料
            "5": "#F44336",  # 红色表示玩手机
            "6": "#607D8B",  # 蓝灰色表示睡觉
            "7": "#795548"   # 棕色表示其他
        }
        
        # 数据存储
        self.behavior_history = []  # (时间戳, 行为编号) 元组列表
        self.behavior_counts = {key: 0 for key in self.behavior_map}
        
        # 图表更新频率
        self.update_interval = 2  # 秒
        
        # 设置图表
        self.setup_charts()
        
        # 启动更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_charts_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def setup_charts(self):
        """创建并设置折线图和饼图"""
        # 创建图表主框架
        self.charts_frame = ctk.CTkFrame(self.parent_frame, fg_color="#1a1a1a")
        self.charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建左侧面板放置折线图（占据大部分空间）
        self.line_chart_frame = ctk.CTkFrame(self.charts_frame, fg_color="#1a1a1a")
        self.line_chart_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # 创建右侧面板放置饼图
        self.right_panel = ctk.CTkFrame(self.charts_frame, fg_color="#1a1a1a")
        self.right_panel.pack(side="right", fill="both", expand=False, padx=5, pady=5, ipadx=10)
        
        # 创建饼图框架
        self.pie_chart_frame = ctk.CTkFrame(self.right_panel, fg_color="#1a1a1a")
        self.pie_chart_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 设置折线图
        self.setup_line_chart()
        
        # 设置饼图
        self.setup_pie_chart()
        
        # 添加刷新按钮
        self.refresh_button = ctk.CTkButton(
            self.right_panel, 
            text="刷新图表", 
            command=self.refresh_charts,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.refresh_button.pack(pady=10, padx=10)
        
        # 初始化空的统计标签字典（仍需保留以避免其他方法的引用错误）
        self.stat_labels = {}
        self.color_frames = {}
    
    def setup_line_chart(self):
        """设置行为跟踪随时间变化的折线图"""
        # 创建matplotlib图形和轴 - 增加宽度以充分利用900px宽度
        self.line_fig = Figure(figsize=(7, 3.8), dpi=100)
        self.line_fig.patch.set_facecolor('#1a1a1a')  # 设置图形背景为黑色
        self.line_ax = self.line_fig.add_subplot(111)
        self.line_ax.set_facecolor('#1a1a1a')  # 设置绘图区背景为黑色
        
        # 设置标题和标签颜色为白色
        self.line_ax.set_title("行为随时间变化", color='white')
        self.line_ax.set_xlabel("时间", color='white')
        self.line_ax.set_ylabel("行为", color='white')
        
        # 设置刻度标签为白色
        self.line_ax.tick_params(axis='x', colors='white')
        self.line_ax.tick_params(axis='y', colors='white')
        
        # 设置边框颜色为白色
        for spine in self.line_ax.spines.values():
            spine.set_edgecolor('white')
        
        # 设置y轴显示行为类型
        self.line_ax.set_yticks(list(range(1, 8)))
        self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
        
        # 添加网格
        self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 嵌入到Tkinter
        self.line_canvas = FigureCanvasTkAgg(self.line_fig, master=self.line_chart_frame)
        self.line_canvas.draw()
        self.line_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def setup_pie_chart(self):
        """设置行为分布饼图"""
        # 创建matplotlib图形和轴 - 设置更大的底部空间给图例
        self.pie_fig = Figure(figsize=(3.5, 3.8), dpi=100)
        self.pie_fig.patch.set_facecolor('#1a1a1a')  # 设置图形背景为黑色
        self.pie_ax = self.pie_fig.add_subplot(111)
        self.pie_ax.set_facecolor('#1a1a1a')  # 设置绘图区背景为黑色
        # 调整子图位置，腾出底部空间给图例
        self.pie_fig.subplots_adjust(bottom=0.2)
        
        # 设置标题颜色为白色
        self.pie_ax.set_title("行为分布", color='white')
        
        # 初始时不显示任何数据，只显示一个空的圆
        self.pie_ax.text(0, 0, "等待数据...", ha='center', va='center', color='white', fontsize=12)
        self.pie_ax.set_aspect('equal')
        self.pie_ax.axis('off')  # 隐藏坐标轴
        
        # 嵌入到Tkinter
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=self.pie_chart_frame)
        self.pie_canvas.draw()
        self.pie_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def add_behavior_data(self, timestamp, behavior_num, behavior_desc):
        """向可视化添加新的行为数据点"""
        try:
            # 添加到历史记录
            self.behavior_history.append((timestamp, behavior_num))
            
            # 更新计数
            self.behavior_counts[behavior_num] = self.behavior_counts.get(behavior_num, 0) + 1
            
            # 限制历史记录长度以提高性能（保留最近100个条目）
            if len(self.behavior_history) > 100:
                self.behavior_history = self.behavior_history[-100:]
                
            print(f"添加行为数据: {behavior_num} - {behavior_desc}")
            
            # 不立即更新图表，更新线程会处理此操作
        except Exception as e:
            print(f"添加行为数据时出错: {e}")
    
    def _update_charts_thread(self):
        """定期更新图表的线程"""
        while self.running:
            try:
                # 更新折线图
                self.update_line_chart()
                
                # 更新饼图
                self.update_pie_chart()
                
                # 更新统计信息
                self.update_statistics()
            except Exception as e:
                print(f"更新图表时出错: {e}")
            
            # 等待下次更新
            time.sleep(self.update_interval)
    
    def update_line_chart(self):
        """用最新数据更新折线图"""
        try:
            self.line_ax.clear()
            
            # 设置背景颜色
            self.line_ax.set_facecolor('#1a1a1a')
            
            # 设置文本颜色为白色
            self.line_ax.set_title("行为随时间变化", color='white')
            self.line_ax.set_xlabel("时间", color='white')
            self.line_ax.set_ylabel("行为", color='white')
            self.line_ax.tick_params(axis='x', colors='white')
            self.line_ax.tick_params(axis='y', colors='white')
            
            # 设置边框颜色为白色
            for spine in self.line_ax.spines.values():
                spine.set_edgecolor('white')
            
            if not self.behavior_history:
                # 尚无数据，显示带有正确标签的空图表
                self.line_ax.set_yticks(list(range(1, 8)))
                self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
                self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                self.line_canvas.draw()
                return
            
            # 提取数据
            times, behaviors = zip(*self.behavior_history)
            
            # 将行为编号转换为整数以便绘图
            behavior_ints = [int(b) for b in behaviors]
            
            # 为每种行为创建散点图和线
            for i in range(1, 8):
                # 筛选此行为的数据
                indices = [j for j, b in enumerate(behavior_ints) if b == i]
                if indices:
                    behavior_times = [times[j] for j in indices]
                    behavior_vals = [behavior_ints[j] for j in indices]
                    
                    # 用正确的颜色绘制散点
                    self.line_ax.scatter(
                        behavior_times, 
                        behavior_vals, 
                        color=self.behavior_colors[str(i)],
                        s=50,  # 点的大小
                        label=self.behavior_map[str(i)]
                    )
            
            # 绘制连接相邻点的线
            self.line_ax.plot(times, behavior_ints, 'k-', alpha=0.3, color='white')
            
            # 将x轴格式化为时间
            self.line_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # 设置时间范围，最多显示1小时的数据，如果数据较少则显示较少时间
            now = datetime.now()
            min_time = now - timedelta(hours=1)
            if times and times[0] < min_time:
                self.line_ax.set_xlim(min_time, now)
            elif times:
                self.line_ax.set_xlim(times[0], now)
            
            # 设置y轴
            self.line_ax.set_yticks(list(range(1, 8)))
            self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
            self.line_ax.set_ylim(0.5, 7.5)  # 添加一些填充
            
            # 添加网格
            self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            # 更新画布
            self.line_fig.tight_layout()
            self.line_canvas.draw()
            
        except Exception as e:
            print(f"更新折线图时出错: {e}")
    
    def update_pie_chart(self):
        """用最新分布更新饼图"""
        try:
            self.pie_ax.clear()
            
            # 设置背景颜色
            self.pie_ax.set_facecolor('#1a1a1a')
            
            # 设置标题颜色为白色
            self.pie_ax.set_title("行为分布", color='white')
            
            # 获取当前计数
            sizes = [self.behavior_counts.get(str(i), 0) for i in range(1, 8)]
            labels = list(self.behavior_map.values())
            colors = [self.behavior_colors[str(i)] for i in range(1, 8)]
            
            # 检查是否有数据
            if sum(sizes) == 0:
                # 没有数据，显示等待消息
                self.pie_ax.text(0, 0, "等待数据...", ha='center', va='center', color='white', fontsize=12)
                self.pie_ax.set_aspect('equal')
                self.pie_ax.axis('off')  # 隐藏坐标轴
            else:
                # 有数据，显示饼图
                wedges, texts, autotexts = self.pie_ax.pie(
                    sizes,
                    labels=None,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'color': 'white'}
                )
                
                # 添加图例到饼图下方而不是右侧
                legend = self.pie_ax.legend(wedges, labels, title="行为类型", 
                              loc="upper center", bbox_to_anchor=(0.5, -0.1),
                              frameon=False, labelcolor='white', fontsize='small', ncol=2)
                # 单独设置标题颜色
                plt.setp(legend.get_title(), color='white')
            
            # 更新画布
            self.pie_canvas.draw()
            
        except Exception as e:
            print(f"更新饼图时出错: {e}")
    
    def update_statistics(self):
        """用最新数据更新统计标签"""
        # 由于我们已删除统计标签区域，此方法保留但不执行任何操作
        pass
    
    def refresh_charts(self):
        """手动刷新所有图表"""
        self.update_line_chart()
        self.update_pie_chart()
        self.update_statistics()
    
    def stop(self):
        """停止更新线程"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

# ---------------- 摄像头处理类 ----------------
class WebcamHandler:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False  # 标记分析是否暂停
        self.processing = False  # 标记分析是否正在进行
        self.cap = None
        self.webcam_thread = None
        self.last_webcam_image = None  # 存储最近的摄像头图像
        self.debug = True  # 设置为True启用调试输出
        
        # 顺序处理控制
        self.analysis_running = False
        
        # 摄像头窗口
        self.camera_window = None
    
    def start(self):
        """启动摄像头捕获进程"""
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
                
                # 启动分析（重要 - 这将启动第一次捕获）
                self.analysis_running = True
                
                # 短暂延迟后启动首次分析
                self.app.after(2000, self.trigger_next_capture)
                
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
            # 将窗口定位在主窗口下方
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            main_height = self.app.winfo_height()
            self.camera_window.geometry(f"640x480+{main_x}+{main_y + main_height + 10}")
    
    def stop(self):
        """停止摄像头捕获进程"""
        self.running = False
        self.analysis_running = False
        if self.cap:
            self.cap.release()
        
        # 关闭摄像头窗口
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
    
    def _process_webcam(self):
        """主摄像头处理循环 - 仅保留最近的帧"""
        last_ui_update_time = 0
        ui_update_interval = 0.05  # 以20fps更新UI
        
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
                
                # 用当前帧更新摄像头窗口
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img)
                    last_ui_update_time = current_time
                
                time.sleep(0.03)  # ~30fps捕获
            except Exception as e:
                error_msg = f"摄像头错误: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # 暂停后重试
    
    def trigger_next_capture(self):
        """触发下一次捕获和分析循环"""
        if self.running and self.analysis_running and not self.paused and not self.processing:
            print(f"触发新一轮图像分析 {time.strftime('%H:%M:%S')}")
            self.capture_and_analyze()
    
    def capture_and_analyze(self):
        """捕获截图并发送进行分析"""
        if self.processing or self.paused:
            return
        
        try:
            self.processing = True
            self.app.update_status("捕捉图像中...")
            
            # 获取分析用的截图和当前显示用的截图
            screenshots, current_screenshot = self._capture_screenshots()
            
            # 在另一个线程中处理分析以保持UI响应
            analysis_thread = threading.Thread(
                target=self._analyze_screenshots, 
                args=(screenshots, current_screenshot)
            )
            analysis_thread.daemon = True
            analysis_thread.start()
                
        except Exception as e:
            error_msg = f"捕获/分析出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.processing = False
            # 延迟后重试
            self.app.after(2000, self.trigger_next_capture)
    
    def _analyze_screenshots(self, screenshots, current_screenshot):
        """分析截图并更新UI"""
        try:
            self.app.update_status("正在分析图像...")
            
            # 将截图上传到OSS
            screenshot_urls = self._upload_screenshots(screenshots)
            
            if screenshot_urls:
                print(f"已上传 {len(screenshot_urls)} 张图片，开始分析")
                
                # 发送进行分析并等待结果（阻塞）
                analysis_text = self._get_image_analysis(screenshot_urls)
                
                if analysis_text:
                    print(f"分析完成")
                    
                    # 从分析文本中提取行为类型
                    behavior_num, behavior_desc = extract_behavior_type(analysis_text)
                    
                    # 记录行为到日志
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
                    logging.info(log_message)
                    print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
                    
                    # 发送到行为可视化器更新图表
                    self.app.add_behavior_data(datetime.now(), behavior_num, behavior_desc, analysis_text)
                    
                    self.app.update_status(f"检测到行为: {behavior_desc}")
                else:
                    print("图像分析返回空结果")
            else:
                print("未能上传截图，无法进行分析")
        except Exception as e:
            error_msg = f"分析截图时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        finally:
            # 重要：标记为未处理并触发下一次捕获
            self.processing = False
            # 下次捕获前添加延迟 - 增加此值以减少API调用
            next_capture_delay = 10000  # 捕获间隔10秒
            self.app.after(next_capture_delay, self.trigger_next_capture)

    def _get_image_analysis(self, image_urls):
        """发送图像到Qwen-VL API并获取分析文本"""
        try:
            print("调用Qwen-VL API分析图像...")
            
            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": "详细观察这个人正在做什么。务必判断他属于以下哪种情况：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。分析他的表情、姿势、手部动作和周围环境来作出判断。使用中文回答，并明确指出是哪种情况。"}]
            }]
            
            message_payload = {
                "role": "user",
                "content": [
                    {"type": "video", "video": image_urls},
                    {"type": "text", "text": "这个人正在做什么？请判断他是：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。请详细描述你观察到的内容并明确指出判断结果。"}
                ]
            }
            messages.append(message_payload)
            
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
            )
            analysis_text = completion.choices[0].message.content
            print(f"图像分析完成，分析长度: {len(analysis_text)} 字符")
            
            return analysis_text
            
        except Exception as e:
            error_msg = f"Qwen-VL API错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            return None
            
    def toggle_pause(self):
        """切换分析循环的暂停状态"""
        self.paused = not self.paused
        status = "已暂停分析" if self.paused else "已恢复分析"
        self.app.update_status(status)
        print(status)
        
        # 如果取消暂停，触发下一次捕获
        if not self.paused and not self.processing:
            self.app.after(500, self.trigger_next_capture)
    
    def get_current_screenshot(self):
        """获取最近的摄像头图像"""
        return self.last_webcam_image
    
    def _capture_screenshots(self, num_shots=4, interval=0.1):
        """从摄像头捕获多个截图用于分析
           返回完整集合（用于分析）和一张当前截图（用于显示）"""
        screenshots = []
        for i in range(num_shots):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            screenshots.append(img)
            time.sleep(interval)
        
        # 再捕获一张当前帧专门用于显示
        ret, current_frame = self.cap.read()
        current_screenshot = None
        if ret:
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            current_screenshot = Image.fromarray(current_frame_rgb)
        
        if self.debug:
            print(f"已捕获 {len(screenshots)} 张截图用于分析和 1 张当前截图")
            
        return screenshots, current_screenshot
    
    def _upload_screenshots(self, screenshots):
        """将截图上传到OSS并返回URL"""
        try:
            auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
            bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
            
            if self.debug:
                print(f"正在上传 {len(screenshots)} 张截图到OSS")
                
            oss_urls = []
            for i, img in enumerate(screenshots):
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                
                object_key = f"screenshots/{int(time.time())}_{i}.jpg"
                
                result = bucket.put_object(object_key, buffer)
                if result.status == 200:
                    url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{object_key}"
                    oss_urls.append(url)
                    if self.debug:
                        print(f"已上传图片 {i+1}: {url}")
                else:
                    error_msg = f"上传错误，状态码: {result.status}"
                    print(error_msg)
                    self.app.update_status(error_msg)
            
            return oss_urls
        except Exception as e:
            error_msg = f"上传图片时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            return []

# ---------------- 主应用类 ----------------
class BehaviorVisualizationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 初始化系统组件
        self.setup_ui()
        self.webcam_handler = WebcamHandler(self)
        
        # 设置按键绑定
        self.setup_key_bindings()
        
        # 短暂延迟后启动摄像头
        self.after(1000, self.start_webcam)
        
        # 启动时间戳检查
        self.check_timestamp()
        
        # 设置观察历史
        self.observation_history = []
        
        # 标题和当前行为
        self.current_behavior = "未知"
    
    def start_webcam(self):
        """UI初始化后启动摄像头捕获"""
        if not self.webcam_handler.start():
            self.update_status("启动摄像头失败。请检查您的摄像头。")
    
    def setup_ui(self):
        """初始化用户界面"""
        self.title("行为监测与可视化系统")
        self.geometry("900x600")  # 修改界面尺寸为900x600
        
        # 设置暗色主题
        self.configure(fg_color="#1a1a1a")  # 深色背景
        
        # 配置网格
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # 标题
        self.grid_rowconfigure(1, weight=1)  # 主要内容
        self.grid_rowconfigure(2, weight=0)  # 状态栏
        
        # 标题框架
        self.title_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        
        self.title_label = ctk.CTkLabel(
            self.title_frame,
            text="行为监测与可视化系统",
            font=("Arial", 20, "bold"),
            text_color="white"
        )
        self.title_label.pack(pady=10)
        
        # 主内容框架
        self.main_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # 初始化行为可视化器（图表）
        self.behavior_visualizer = BehaviorVisualizer(self.main_frame)
        
        # 状态栏
        self.status_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # 当前行为显示
        self.behavior_label = ctk.CTkLabel(
            self.status_frame,
            text="当前行为: 未知",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.behavior_label.pack(side="left", padx=10, pady=5)
        
        # 状态标签
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="就绪",
            font=("Arial", 12),
            text_color="white"
        )
        self.status_label.pack(side="right", padx=10, pady=5)
        
        # 控制按钮
        self.controls_frame = ctk.CTkFrame(self.status_frame, fg_color="#1a1a1a")
        self.controls_frame.pack(side="top", fill="x")
        
        self.toggle_button = ctk.CTkButton(
            self.controls_frame,
            text="暂停分析",
            command=self.toggle_analysis,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.toggle_button.pack(side="left", padx=10, pady=5)
        
        self.toggle_camera_button = ctk.CTkButton(
            self.controls_frame,
            text="显示/隐藏摄像头",
            command=self.toggle_camera,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.toggle_camera_button.pack(side="left", padx=10, pady=5)
    
    def setup_key_bindings(self):
        """设置键盘快捷键"""
        self.bind("<space>", lambda e: self.toggle_analysis())
        self.bind("<c>", lambda e: self.toggle_camera())
    
    def toggle_analysis(self):
        """切换分析循环的暂停状态"""
        self.webcam_handler.toggle_pause()
        
        # 更新按钮文本
        new_text = "恢复分析" if self.webcam_handler.paused else "暂停分析"
        self.toggle_button.configure(text=new_text)
    
    def toggle_camera(self):
        """显示或隐藏摄像头窗口"""
        if self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()
        else:
            self.webcam_handler.create_camera_window()
    
    def add_behavior_data(self, timestamp, behavior_num, behavior_desc, analysis_text):
        """将检测到的行为添加到可视化和历史中"""
        # 添加到观察历史
        observation = {
            "timestamp": timestamp,
            "behavior_num": behavior_num,
            "behavior_desc": behavior_desc,
            "analysis": analysis_text
        }
        self.observation_history.append(observation)
        
        # 限制历史长度
        if len(self.observation_history) > 100:
            self.observation_history = self.observation_history[-100:]
        
        # 添加到行为可视化器
        self.behavior_visualizer.add_behavior_data(timestamp, behavior_num, behavior_desc)
        
        # 更新当前行为显示
        self.current_behavior = behavior_desc
        self.behavior_label.configure(text=f"当前行为: {behavior_desc}")
        
        # 根据行为更新UI颜色
        behavior_colors = {
            "1": "#4CAF50",  # 绿色表示工作
            "2": "#FFC107",  # 琥珀色表示吃东西
            "3": "#2196F3",  # 蓝色表示喝水
            "4": "#9C27B0",  # 紫色表示喝饮料
            "5": "#F44336",  # 红色表示玩手机
            "6": "#607D8B",  # 蓝灰色表示睡觉
            "7": "#795548"   # 棕色表示其他
        }
        
        try:
            # 根据行为设置标签文本颜色
            color = behavior_colors.get(behavior_num, "#000000")
            self.behavior_label.configure(text_color=color)
        except Exception as e:
            print(f"更新UI颜色时出错: {e}")
    
    def update_status(self, text):
        """更新状态消息"""
        self.status_label.configure(text=text)
    
    def check_timestamp(self):
        """检查周期性更新（用于刷新图表）"""
        # 定期刷新可视化
        self.behavior_visualizer.refresh_charts()
        
        # 安排下一次检查
        self.after(30000, self.check_timestamp)  # 每30秒

# ---------------- 主函数 ----------------
def main():
    # 设置外观模式和默认主题
    ctk.set_appearance_mode("Dark")  # 设置为深色模式
    ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
    
    app = BehaviorVisualizationApp()
    app.protocol("WM_DELETE_WINDOW", lambda: quit_app(app))
    app.mainloop()

def quit_app(app):
    """干净地关闭应用程序"""
    # 停止所有线程
    if hasattr(app, 'webcam_handler'):
        app.webcam_handler.stop()
    
    if hasattr(app, 'behavior_visualizer'):
        app.behavior_visualizer.stop()
    
    # 关闭应用
    app.destroy()

if __name__ == "__main__":
    main()