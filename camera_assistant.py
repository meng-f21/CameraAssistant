import os
import cv2
import time
import io
import threading
import queue
import numpy as np
import pyaudio
import wave
# import keyboard
import customtkinter as ctk
from PIL import Image, ImageTk
import oss2
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from datetime import datetime
import re
import logging
import random

# ---------------- Configuration ----------------
# Import configuration from local config file
try:
    from config import (
        OSS_ACCESS_KEY_ID,
        OSS_ACCESS_KEY_SECRET,
        OSS_ENDPOINT,
        OSS_BUCKET,
        DEEPSEEK_API_KEY,
        DEEPSEEK_BASE_URL,
        QWEN_API_KEY,
        QWEN_BASE_URL
    )
except ImportError:
    # Fallback to placeholder values if config.py is not found
    print("Warning: config.py not found. Using placeholder values.")
    # OSS Configuration
    OSS_ACCESS_KEY_ID = "YOUR_OSS_ACCESS_KEY_ID"
    OSS_ACCESS_KEY_SECRET = "YOUR_OSS_ACCESS_KEY_SECRET"
    OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
    OSS_BUCKET = 'felixdscamera'

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    # Qwen-VL API Configuration
    QWEN_API_KEY = "YOUR_QWEN_API_KEY"
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# TTS Configuration
dashscope.api_key = QWEN_API_KEY
TTS_MODEL = "cosyvoice-v1"
TTS_VOICE = "longwan"

# SenseVoice ASR Configuration
MODEL_DIR = "iic/SenseVoiceSmall"

# Audio Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "output.wav"

# Logging Configuration
LOG_FILE = "behavior_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# ---------------- API Clients Initialization ----------------
# DeepSeek Client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# Qwen-VL Client
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

# ASR Model
asr_model = AutoModel(
    model=MODEL_DIR,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

# ---------------- Utility Functions ----------------
def extract_language_emotion_content(text):
    """Extract clean content from ASR output"""
    # Extract language
    language_start = text.find("|") + 1
    language_end = text.find("|", language_start)
    language = text[language_start:language_end]
    
    # Extract emotion
    emotion_start = text.find("|", language_end + 1) + 1
    emotion_end = text.find("|", emotion_start)
    emotion = text[emotion_start:emotion_end]
    
    # Extract content
    content_start = text.find(">", emotion_end) + 1
    content = text[content_start:]
    
    # Clean up any remaining tags
    while content.startswith("<|"):
        end_tag = content.find(">", 2) + 1
        content = content[end_tag:]
    
    return content.strip()

def extract_behavior_type(analysis_text):
    """Extract behavior type number from AI analysis text"""
    # Try to find behavior type number in the text (1-4)
    pattern = r'(\d+)\s*[.、:]?\s*(正确佩戴安全帽和防护服|未佩戴安全帽或防护服|在作业区域内|在作业区域外)'
    match = re.search(pattern, analysis_text)
    
    if match:
        behavior_num = match.group(1)
        behavior_desc = match.group(2)
        return behavior_num, behavior_desc
    
    # Alternative pattern if the first one fails
    patterns = [
        (r'正确佩戴安全帽和防护服', '1'),
        (r'未佩戴安全帽或防护服', '2'),
        (r'在作业区域内', '3'),
        (r'在作业区域外', '4')
    ]
    
    for pattern, num in patterns:
        if re.search(pattern, analysis_text):
            return num, pattern
    
    return "0", "未识别"  # Default if no pattern matches

# ---------------- Camera Display Window ----------------
class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Camera Feed")
        self.geometry("640x480")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create frame for the camera display
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create label for the camera image
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Starting camera...")
        self.camera_label.pack(fill="both", expand=True)
        
        # Image holder
        self.current_image = None
        
        # Flag to indicate if window is closed
        self.is_closed = False
    
    def update_frame(self, img):
        """Update camera frame with new image"""
        if self.is_closed:
            return
            
        try:
            if img:
                # Resize the image to fit the window nicely
                img_resized = img.copy()
                img_resized.thumbnail((640, 480))
                
                # Convert to CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(640, 480))
                
                # Update the label
                self.camera_label.configure(image=ctk_img, text="")
                
                # Store a reference to prevent garbage collection
                self.current_image = ctk_img
        except Exception as e:
            print(f"Error updating camera frame: {e}")
    
    def on_closing(self):
        """Handle window close event"""
        self.is_closed = True
        self.withdraw()  # Hide instead of destroy to allow reopening

# ---------------- Core Functionality Classes ----------------
class AudioRecorder:
    def __init__(self, app):
        self.app = app
        self.recording = False
        self.stop_recording_flag = False
        self.audio_thread = None
        
    def start_recording(self):
        """Begin audio recording when 'r' key is pressed"""
        if not self.recording:
            self.recording = True
            self.stop_recording_flag = False
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.app.update_status("Recording...")
    
    def stop_recording(self):
        """Stop audio recording when 's' key is pressed"""
        if self.recording:
            self.stop_recording_flag = True
            self.recording = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            self.app.update_status("Processing audio...")
    
    def _record_audio(self):
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)
        
        frames = []
        
        while self.recording and not self.stop_recording_flag:
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except Exception as e:
                self.app.update_status(f"Error recording audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if frames:
            try:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                self.app.transcribe_audio(WAVE_OUTPUT_FILENAME)
            except Exception as e:
                self.app.update_status(f"Error saving audio: {e}")

class VoiceActivityDetector:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.listening_thread = None
        self.detection_thread = None
        
        # Voice activity detection parameters - MUCH lower threshold
        self.energy_threshold = 80  # Further reduced for better sensitivity
        self.dynamic_threshold = True  # Dynamically adjust threshold based on environment noise
        self.silence_threshold = 0.8  # Seconds of silence to consider speech ended
        self.min_speech_duration = 0.3  # Shorter minimum duration to catch brief utterances
        self.max_speech_duration = 30.0  # Maximum speech duration
        
        # Speech detection state
        self.is_speaking = False
        self.speech_started = 0
        self.silence_started = 0
        self.speech_frames = []
        
        # For dynamic threshold adjustment
        self.noise_levels = []
        self.max_noise_levels = 100
        
        # Audio stream
        self.audio = None
        self.stream = None
        
        # Debug mode
        self.debug = True  # Set to True to enable energy level debugging
        
        # Add a calibration phase
        self.is_calibrating = True
        self.calibration_duration = 3  # seconds
        self.calibration_start_time = 0
    
    def start_monitoring(self):
        """Begin continuous voice monitoring"""
        if not self.running:
            self.running = True
            self.listening_thread = threading.Thread(target=self._monitor_audio)
            self.listening_thread.daemon = True
            self.listening_thread.start()
            self.app.update_status("语音监测启动中... 正在校准麦克风")
    
    def stop_monitoring(self):
        """Stop voice monitoring"""
        self.running = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        if self.audio and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.audio = None
            self.stream = None
    
    def _get_energy(self, audio_data):
        """Calculate audio energy level"""
        try:
            # Convert bytes to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Ensure we have valid data
            if len(data) == 0 or np.all(data == 0):
                return 0.0
                
            # Calculate RMS energy
            # Use np.mean(np.abs(data)) as it's more robust than squaring
            energy = np.mean(np.abs(data))
            return energy
        except Exception as e:
            print(f"Error calculating energy: {e}")
            return 0.0
    
    def _is_speech(self, audio_data, energy=None):
        """Detect if audio chunk contains speech based on energy level"""
        try:
            # Skip speech detection if audio is playing
            if hasattr(self.app, 'is_playing_audio') and self.app.is_playing_audio:
                if self.debug and time.time() % 2 < 0.1:
                    print("语音监测暂停中 - 正在播放系统语音")
                return False
            
            # Use provided energy or calculate it
            if energy is None:
                energy = self._get_energy(audio_data)
            
            # If we're calibrating, just collect noise levels
            if self.is_calibrating:
                self.noise_levels.append(energy)
                return False
            
            # Adjust threshold dynamically if enabled
            threshold = self.energy_threshold
            if self.dynamic_threshold and len(self.noise_levels) > 0:
                # Set threshold to be 2.5x the average noise level
                noise_avg = sum(self.noise_levels) / len(self.noise_levels)
                dynamic_threshold = noise_avg * 2.5
                threshold = max(threshold, dynamic_threshold)
            
            # Debug output for energy levels
            if self.debug and time.time() % 1 < 0.1:  # Print every second
                print(f"能量: {energy:.1f}, 阈值: {threshold:.1f}, " + 
                      f"平均噪音: {sum(self.noise_levels) / max(1, len(self.noise_levels)):.1f}")
            
            # Detect speech when energy is above threshold
            return energy > threshold
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False
    
    def _calibrate_microphone(self):
        """Calibrate microphone by measuring background noise"""
        try:
            self.calibration_start_time = time.time()
            self.is_calibrating = True
            self.noise_levels = []
            
            print("开始麦克风校准...")
            self.app.update_status("校准麦克风中，请保持安静...")
            
            # Wait for calibration to complete
            while self.is_calibrating and time.time() - self.calibration_start_time < self.calibration_duration:
                time.sleep(0.1)
            
            # Calculate noise threshold
            if len(self.noise_levels) > 0:
                avg_noise = sum(self.noise_levels) / len(self.noise_levels)
                self.energy_threshold = max(100, avg_noise * 2.5)  # Set threshold to 2.5x average noise
                
                print(f"麦克风校准完成: 平均噪音级别 {avg_noise:.1f}, 阈值设为 {self.energy_threshold:.1f}")
                self.app.update_status(f"语音监测已启动 (阈值: {self.energy_threshold:.1f})")
            else:
                print("校准失败: 没有收集到噪音样本")
                self.app.update_status("语音监测已启动，但校准失败")
            
            self.is_calibrating = False
        except Exception as e:
            print(f"麦克风校准错误: {e}")
            self.is_calibrating = False
            self.app.update_status("语音监测已启动，但校准出错")
    
    def _monitor_audio(self):
        """Continuously monitor audio for speech"""
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Perform initial calibration
            self._calibrate_microphone()
            
            # Continuous audio analysis loop
            while self.running:
                try:
                    # Read audio chunk
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Calculate energy once to avoid duplicate work
                    energy = self._get_energy(audio_data)
                    
                    # Update noise level (only when not speaking)
                    if not self.is_speaking and len(self.noise_levels) < self.max_noise_levels:
                        self.noise_levels.append(energy)
                        if len(self.noise_levels) > self.max_noise_levels:
                            self.noise_levels.pop(0)  # Keep the list size limited
                    
                    # Check if it's speech
                    if self._is_speech(audio_data, energy):
                        # If we weren't already speaking, mark the start
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.speech_started = time.time()
                            self.speech_frames = []
                            # Show visual feedback immediately
                            print("语音开始检测中...")
                            self.app.after(0, lambda: self.app.update_status("检测到语音输入..."))
                        
                        # Reset silence counter
                        self.silence_started = 0
                        
                        # Add frame to speech buffer
                        self.speech_frames.append(audio_data)
                        
                        # Check if we've exceeded max duration
                        if time.time() - self.speech_started > self.max_speech_duration:
                            print(f"达到最大语音长度 ({self.max_speech_duration}s)，开始处理")
                            self._process_speech()
                    
                    elif self.is_speaking:
                        # If we were speaking, but now detected silence
                        if self.silence_started == 0:
                            self.silence_started = time.time()
                            print(f"检测到语音之后的静音")
                        
                        # Add the silent frame (for smoother audio)
                        self.speech_frames.append(audio_data)
                        
                        # If silence continues for threshold duration, process the speech
                        silence_duration = time.time() - self.silence_started
                        if silence_duration > self.silence_threshold:
                            print(f"静音时长达到阈值 ({silence_duration:.2f}s > {self.silence_threshold}s)，开始处理语音")
                            self._process_speech()
                    
                    time.sleep(0.01)  # Small sleep to reduce CPU usage
                    
                except Exception as e:
                    error_msg = f"音频监测错误: {e}"
                    print(error_msg)
                    self.app.update_status(error_msg)
                    time.sleep(0.5)  # Sleep before retry
                    
        except Exception as e:
            error_msg = f"语音监测失败: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
    
    def _process_speech(self):
        """Process detected speech segment"""
        speech_duration = time.time() - self.speech_started
        
        # Only process if speech is long enough and has frames
        if speech_duration >= self.min_speech_duration and len(self.speech_frames) > 0:
            print(f"处理语音片段: {speech_duration:.2f}秒, {len(self.speech_frames)} 帧")
            
            # Reset speech state
            is_speaking_was = self.is_speaking
            self.is_speaking = False
            self.silence_started = 0
            
            # Save a copy of speech frames before resetting
            frames_copy = self.speech_frames.copy()
            self.speech_frames = []
            
            # Check if we truly had meaningful speech
            if is_speaking_was and speech_duration > 0.5:  # Additional validation
                # Process in a separate thread to not block monitoring
                self.detection_thread = threading.Thread(
                    target=self._save_and_transcribe, 
                    args=(frames_copy,)
                )
                self.detection_thread.daemon = True
                self.detection_thread.start()
            else:
                print(f"语音太短或者无效: {speech_duration:.2f}秒")
                self.app.update_status("Ready")
        else:
            # Too short, reset without processing
            print(f"语音太短 ({speech_duration:.2f}秒 < {self.min_speech_duration}秒)，忽略")
            self.is_speaking = False
            self.silence_started = 0
            self.speech_frames = []
            self.app.update_status("Ready")
    
    def _save_and_transcribe(self, frames):
        """Save speech frames to file and start transcription"""
        try:
            os.makedirs("output", exist_ok=True)
            temp_filename = os.path.join("output", f"speech_{int(time.time())}.wav")
            print(f"保存语音到 {temp_filename}")
            
            # Ensure the audio object exists
            if not self.audio:
                print("错误: 音频对象不存在，无法保存语音")
                return
            
            # Check if we have frames
            if not frames or len(frames) == 0:
                print("错误: 没有语音帧可以保存")
                return
            
            # Save frames to WAV file

            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Verify the file was saved
            # Create output directory if it doesn't exist
            
            # output_path = os.path.join("output", temp_filename)
            
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                print(f"语音文件已保存: {temp_filename}, 大小: {os.path.getsize(temp_filename)} 字节")
            else:
                print(f"保存语音文件失败: {temp_filename}")
                return
            
            # 不再创建占位符，直接发送进行转录
            # 确保UI响应完成后再进入繁重的语音处理
            self.app.after(100, lambda: self._send_for_transcription(temp_filename))
            
        except Exception as e:
            error_msg = f"处理语音出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
    
    def _send_for_transcription(self, audio_file):
        """Send audio file for transcription after UI is updated"""
        try:
            print(f"发送语音文件进行转写: {audio_file}")
            # Send for transcription - without placeholder ID
            self.app.transcribe_audio(audio_file, priority=True)
        except Exception as e:
            error_msg = f"发送转写请求时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)

class WebcamHandler:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False  # Flag to indicate if analysis is paused
        self.processing = False  # Flag to indicate if analysis is in progress
        self.cap = None
        self.webcam_thread = None
        self.last_webcam_image = None  # Store the most recent webcam image
        self.debug = True  # Set to True to enable debugging output
        
        # Sequential processing control
        self.analysis_running = False
        
        # Camera window
        self.camera_window = None
    
    def start(self):
        """Start webcam capture process"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.update_status("Cannot open webcam")
                    return False
                
                self.running = True
                
                # Create the camera window
                self.create_camera_window()
                
                # Start processing thread
                self.webcam_thread = threading.Thread(target=self._process_webcam)
                self.webcam_thread.daemon = True
                self.webcam_thread.start()
                
                # Start analysis (important - this kicks off the first capture)
                self.analysis_running = True
                
                # Start first analysis after a short delay
                self.app.after(2000, self.trigger_next_capture)
                
                return True
            except Exception as e:
                self.app.update_status(f"Error starting webcam: {e}")
                return False
        return False
    
    def create_camera_window(self):
        """Create a window to display the camera feed"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("Camera Feed")
            # Position the window to the right of the main window
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            self.camera_window.geometry(f"640x480+{main_x + self.app.winfo_width() + 10}+{main_y}")
    
    def stop(self):
        """Stop webcam capture process"""
        self.running = False
        self.analysis_running = False
        if self.cap:
            self.cap.release()
        
        # Close the camera window
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
    
    def _process_webcam(self):
        """Main webcam processing loop - just keeps the most recent frame"""
        last_ui_update_time = 0
        ui_update_interval = 0.05  # Update UI at 20 fps
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.app.update_status("Failed to capture frame")
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
                
                time.sleep(0.03)  # ~30 fps for capture
            except Exception as e:
                error_msg = f"Webcam error: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # Pause before retry
    
    def trigger_next_capture(self):
        """Trigger the next capture and analysis cycle"""
        if self.running and self.analysis_running and not self.paused and not self.processing:
            print(f"触发新一轮图像分析 {time.strftime('%H:%M:%S')}")
            self.capture_and_analyze()
    
    def capture_and_analyze(self):
        """Capture screenshots and send for analysis"""
        if self.processing or self.paused:
            return
        
        try:
            self.processing = True
            self.app.update_status("捕捉图像中...")
            
            # Get both analysis screenshots and current display screenshot
            screenshots, current_screenshot = self._capture_screenshots()
            
            # Show immediate feedback with the current screenshot
            if current_screenshot:
                # Generate placeholder ID for tracking
                placeholder_id = f"img_{int(time.time())}"
                
                # Show a placeholder message in the UI while we wait for analysis
                self.app.add_ai_message("正在分析当前画面...", current_screenshot, is_placeholder=True, placeholder_id=placeholder_id)
                
                if self.debug:
                    print(f"已添加图像占位符到UI: {placeholder_id}")
                
                # Process analysis in another thread to keep UI responsive
                analysis_thread = threading.Thread(
                    target=self._analyze_screenshots, 
                    args=(screenshots, current_screenshot, placeholder_id)
                )
                analysis_thread.daemon = True
                analysis_thread.start()
            else:
                print("未能获取有效截图，跳过分析")
                self.processing = False
                # Try again after a short delay
                self.app.after(1000, self.trigger_next_capture)
                
        except Exception as e:
            error_msg = f"捕获/分析出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.processing = False
            # Try again after a delay
            self.app.after(2000, self.trigger_next_capture)
    

    def _analyze_screenshots(self, screenshots, current_screenshot, placeholder_id):
        """Analyze screenshots and update UI"""
        try:
            self.app.update_status("正在分析图像...")
            
            # Upload screenshots to OSS
            screenshot_urls = self._upload_screenshots(screenshots)
            
            if screenshot_urls:
                print(f"已上传 {len(screenshot_urls)} 张图片，开始分析")
                
                # Send for analysis and wait for result (blocking)
                analysis_text = self._get_image_analysis(screenshot_urls)
                
                if analysis_text:
                    print(f"分析完成，更新占位符: {placeholder_id}")
                    
                    # Extract behavior type for logging
                    behavior_num, behavior_desc = extract_behavior_type(analysis_text)
                    
                    # Log the behavior
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
                    logging.info(log_message)
                    print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
                    
                    # *** 修改：在这里直接操作app的observation_history，确保记录被添加 ***
                    current_time = time.time()
                    observation = {
                        "timestamp": current_time,
                        "behavior_num": behavior_num,
                        "behavior_desc": behavior_desc,
                        "analysis": analysis_text
                    }
                    
                    self.app.observation_history.append(observation)
                    print(f"WebcamHandler: 已添加新行为到observation_history: {behavior_num}-{behavior_desc}, 当前长度: {len(self.app.observation_history)}")
                    
                    # Process the image analysis directly 
                    if placeholder_id in self.app.placeholder_map:
                        self.app.update_status("处理分析结果...")
                        self.app.update_placeholder(
                            placeholder_id, 
                            analysis_text, 
                            screenshots=[current_screenshot] if current_screenshot else []
                        )
                    else:
                        print(f"警告: 找不到占位符 {placeholder_id}，无法更新UI")
                else:
                    print("图像分析返回空结果")
            else:
                print("未能上传截图，无法进行分析")
        except Exception as e:
            error_msg = f"分析截图时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        finally:
            # Important: Mark as not processing and trigger next capture
            self.processing = False
            # Add a slight delay before next capture
            self.app.after(1000, self.trigger_next_capture)

    
    def _get_image_analysis(self, image_urls):
        """Send images to Qwen-VL API and get analysis text"""
        try:
            print("调用Qwen-VL API分析图像...")
            
            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": "详细观察这个人的安全合规情况。务必判断他属于以下哪种情况：1.正确佩戴安全帽和防护服且在作业区域内, 2.未佩戴安全帽或防护服, 3.在作业区域内, 4.在作业区域外。分析他的安全装备佩戴情况和位置来作出判断。使用中文回答，并明确指出是哪种情况。"}]
            }]
            
            message_payload = {
                "role": "user",
                "content": [
                    {"type": "video", "video": image_urls},
                    {"type": "text", "text": "这个人的安全合规情况如何？请判断他是：1.正确佩戴安全帽和防护服且在作业区域内, 2.未佩戴安全帽或防护服, 3.在作业区域内, 4.在作业区域外。请详细描述你观察到的内容并明确指出判断结果。"}
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
        """Toggle the paused state of the analysis cycle"""
        self.paused = not self.paused
        status = "已暂停分析" if self.paused else "已恢复分析"
        self.app.update_status(status)
        print(status)
        
        # If unpausing, trigger next capture
        if not self.paused and not self.processing:
            self.app.after(500, self.trigger_next_capture)
    
    def get_current_screenshot(self):
        """Get the most recent webcam image"""
        return self.last_webcam_image
    
    def _capture_screenshots(self, num_shots=4, interval=0.1):
        """Capture multiple screenshots from webcam for analysis
           Return both the full set (for analysis) and one current screenshot for display"""
        screenshots = []
        for i in range(num_shots):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            screenshots.append(img)
            time.sleep(interval)
        
        # Capture one more current frame specifically for display
        ret, current_frame = self.cap.read()
        current_screenshot = None
        if ret:
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            current_screenshot = Image.fromarray(current_frame_rgb)
        
        if self.debug:
            print(f"已捕获 {len(screenshots)} 张截图用于分析和 1 张当前截图")
            
        return screenshots, current_screenshot
    
    def _upload_screenshots(self, screenshots):
        """Upload screenshots to OSS and return URLs"""
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

class AudioPlayer:
    def __init__(self, app):
        self.app = app
        self.current_audio = None
        self.playing = False
        self.play_thread = None
        self.skip_requested = False
        
        # 修改为优先级队列
        self.tts_queue = queue.PriorityQueue()
        self.tts_thread = None
        self.tts_running = False
        
        # 最大队列长度限制
        self.max_queue_size = 1
    
    def start_tts_thread(self):
        """启动TTS处理线程"""
        if not self.tts_running:
            self.tts_running = True
            self.tts_thread = threading.Thread(target=self._process_tts_queue)
            self.tts_thread.daemon = True
            self.tts_thread.start()
            print("TTS处理线程已启动")
    
    def _process_tts_queue(self):
        """处理TTS队列中的文本，按优先级播放"""
        while self.tts_running:
            try:
                if not self.tts_queue.empty() and not self.playing:
                    # 获取优先级最高的项目 (priority, timestamp, text)
                    priority, timestamp, text = self.tts_queue.get()
                    
                    # 检查是否过期（超过10秒的低优先级消息被视为过期）
                    current_time = time.time()
                    if priority > 1 and current_time - timestamp > 10:
                        print(f"忽略过期的TTS请求 (已过{current_time - timestamp:.1f}秒): '{text[:30]}...'")
                        self.tts_queue.task_done()
                        continue
                    
                    print(f"从TTS队列获取文本 (优先级: {priority}): '{text[:30]}...'")
                    self._synthesize_and_play(text)
                    self.tts_queue.task_done()
                time.sleep(0.1)
            except Exception as e:
                print(f"处理TTS队列时出错: {e}")
                time.sleep(1)
    
    def play_text(self, text, priority=2):
        """将文本添加到TTS队列，支持优先级
           优先级: 1=用户语音回复(最高), 2=图像分析(普通)
        """
        if not text or len(text.strip()) == 0:
            print("警告: 尝试播放空文本，已忽略")
            return
        
        # 清理队列，如果是高优先级请求或队列已满
        if priority == 1 or self.tts_queue.qsize() >= self.max_queue_size:
            self._clean_queue(priority)
            
        print(f"添加文本到TTS队列 (优先级: {priority}): '{text[:30]}...'")
        
        # 确保TTS处理线程已启动
        if not self.tts_running or not self.tts_thread or not self.tts_thread.is_alive():
            self.start_tts_thread()
        
        # 添加到队列（包含优先级和时间戳）
        self.tts_queue.put((priority, time.time(), text))
    
    def _clean_queue(self, new_priority):
        """清理队列，保留更高优先级的项目"""
        if self.tts_queue.empty():
            return
            
        # 如果是最高优先级请求，清空所有正在排队的音频
        if new_priority == 1:
            print("收到高优先级语音请求，清空当前TTS队列")
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except:
                    pass
            return
        
        # 对于普通优先级，仅保持队列在最大长度以下
        while self.tts_queue.qsize() >= self.max_queue_size:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                print("队列已满，移除最旧的TTS请求")
            except:
                break

    # 之前的方法保持不变...
    def _synthesize_and_play(self, text):
        """合成并播放语音（内部方法，由队列处理器调用）"""
        self.app.update_status("正在合成语音...")
        print(f"TTS合成: '{text}'")
        
        # Set playing status to disable voice detection
        self.app.is_playing_audio = True
        
        try:
            synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE)
            audio = synthesizer.call(text)
            
            if audio is None:
                error_msg = "TTS返回空数据，跳过语音播放"
                print(error_msg)
                self.app.update_status(error_msg)
                self.app.is_playing_audio = False
                return
            
            output_file = f'output_{int(time.time())}.mp3'
            with open(output_file, 'wb') as f:
                f.write(audio)
            
            print(f"TTS文件已保存: {output_file}")
            self._play_audio_file_internal(output_file)
        except Exception as e:
            error_msg = f"TTS错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.app.is_playing_audio = False
    
    def play_audio_file(self, file_path):
        """公共方法用于播放音频文件"""
        print(f"请求播放音频文件: {file_path}")
        
        # 跳过当前播放并等待
        if self.playing:
            self.skip_requested = True
            if self.play_thread and self.play_thread.is_alive():
                print("等待当前播放结束...")
                self.play_thread.join(timeout=2.0)
                
        # 直接播放文件，不通过队列
        self._play_audio_file_internal(file_path)
    
    def _play_audio_file_internal(self, file_path):
        """内部方法用于实际播放音频文件"""
        print(f"开始播放音频文件: {file_path}")
        
        # 确保之前的播放已停止
        if self.playing:
            self.skip_requested = True
            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join(timeout=1.0)
        
        self.skip_requested = False
        self.playing = True
        
        # Mark system as playing audio to disable voice detection
        self.app.is_playing_audio = True
        
        self.play_thread = threading.Thread(target=self._play_audio, args=(file_path,))
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _play_audio(self, file_path):
        """Audio playback worker thread"""
        self.app.update_status("正在播放语音...")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"音频文件不存在: {file_path}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
                
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"音频文件大小: {file_size} 字节")
            if file_size == 0:
                error_msg = f"音频文件为空: {file_path}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
            
            # Load audio file
            try:
                sound = AudioSegment.from_file(file_path, format="mp3")
                print(f"成功加载音频: 长度 {len(sound)/1000:.2f}秒")
            except Exception as e:
                error_msg = f"加载音频失败: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
            
            # Play the audio
            try:
                player = play(sound)
                print("音频开始播放")
                
                # Wait until playing is done or skip is requested
                while self.playing and not self.skip_requested:
                    if not player.is_alive():
                        print("音频播放完成")
                        break
                    time.sleep(0.1)
                    
                if self.skip_requested:
                    print("音频播放被跳过")
            except Exception as e:
                error_msg = f"播放时出错: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                
            # 尝试删除临时文件
            try:
                if os.path.exists(file_path) and file_path.startswith('output_'):
                    os.remove(file_path)
                    print(f"临时文件已删除: {file_path}")
            except Exception as e:
                print(f"删除临时文件出错: {e}")
                
        except Exception as e:
            error_msg = f"音频播放错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        
        self.playing = False
        # Reset playing status to re-enable voice detection
        self.app.is_playing_audio = False
        
        self.app.update_status("Ready")
    
    def skip_current(self):
        """Skip the currently playing audio"""
        if self.playing:
            self.skip_requested = True
            self.app.update_status("跳过当前音频...")
            print("已请求跳过当前音频")
            
            # Reset playing status immediately to re-enable voice detection
            self.app.is_playing_audio = False
            
    def stop(self):
        """停止所有播放和处理"""
        self.skip_current()
        self.tts_running = False
        
        # 清空队列
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except:
                pass

# ---------------- UI Class ----------------
class MultimediaAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Set up priority queue for async processing
        self.message_queue = queue.PriorityQueue()
        self.processing_thread = None
        self.processing_running = False
        
        # Message sequence tracking (for updating placeholders)
        self.message_id = 0
        self.placeholder_map = {}  # Maps placeholder IDs to their row indexes
        
        # 先定义系统消息
        self.system_message = {"role": "system", "content": """你是一个安全监督AI助手，负责确保工作场所的安全合规。

        你需要：
        1. 根据观察到的安全合规情况，分为以下几类并作出相应回应：
        - 如果人员正确佩戴安全帽和防护服：给予肯定，鼓励继续保持
        - 如果人员未佩戴安全帽或防护服：立即发出警告，要求立即佩戴
        - 如果人员在作业区域内：确认合规，给予鼓励
        - 如果人员在作业区域外：发出警告，要求立即回到作业区域
        2. 对合规行为使用鼓励赞赏的语气
        3. 对违规行为使用严厉警告的语气
        4. 每次回应控制在30字以内，简短有力
        5. 语气要根据行为类型明显区分 - 鼓励时温和友好，警告时严厉直接
        6. 非常重要：当用户询问安全合规情况时，你必须查看提供的历史记录和统计数据，根据实际历史回答，不要臆测

        记住：你的目标是确保工作场所的安全合规，防止事故发生！
        """}

        # 然后初始化聊天上下文，使用系统消息
        self.chat_context = [self.system_message]
        self.observation_history = []  # 存储历史观察记录
        self.behavior_counters = {
            "proper_equipment": 0,  # 正确佩戴安全装备计数
            "missing_equipment": 0,  # 未佩戴安全装备计数
            "in_work_zone": 0,  # 在作业区域内计数
            "out_of_zone": 0  # 在作业区域外计数
        }
        self.last_behavior = None  # 上次检测到的行为
        self.continuous_behavior_time = 0  # 持续行为的开始时间
        self.reminder_thresholds = {
            "missing_equipment": 1,  # 未佩戴安全装备提醒阈值
            "out_of_zone": 1,  # 离开作业区域提醒阈值
        }
        self.last_reminder_time = {  # 上次提醒时间
            "missing_equipment": 0,
            "out_of_zone": 0,
            "encouragement": 0  # 鼓励的上次时间
        }
        self.reminder_interval = 5*60  # 两次提醒之间的最小间隔（5分钟）
        self.sitting_start_time = time.time()  # 开始坐下的时间
        
        # Last image analysis for context
        self.last_image_analysis = ""
        
        # Timestamp tracker
        self.last_timestamp = 0
        self.timestamp_interval = 60  # Show timestamp every 60 seconds
        
        # Audio playback status to prevent voice detection during playback
        self.is_playing_audio = False
        
        # Setup UI
        self.setup_ui()
        
        # Initialize system components after UI
        self.audio_recorder = AudioRecorder(self)
        self.webcam_handler = WebcamHandler(self)
        self.audio_player = AudioPlayer(self)
        self.voice_detector = VoiceActivityDetector(self)
        
        # Setup key bindings
        self.setup_key_bindings()
        
        # Start background processing
        self.start_processing_thread()
        
        # Start webcam after a short delay
        self.after(1000, self.start_webcam)
        
        # Start voice monitoring after webcam init
        self.after(2000, self.start_voice_monitoring)
        
        # Start timestamp check
        self.check_timestamp()
        
        # Start audio player TTS thread
        self.after(3000, self.audio_player.start_tts_thread)
    
    def start_webcam(self):
        """Start webcam capture after UI initialization"""
        if not self.webcam_handler.start():
            self.update_status("Failed to start webcam. Check your camera.")
    
    def start_voice_monitoring(self):
        """Start continuous voice activity detection"""
        self.voice_detector.start_monitoring()
        self.update_status("语音监测已启动")
    
    def setup_ui(self):
        """Initialize the user interface"""
        self.title("监控助手")
        self.geometry("1000x800")
        self.default_font_family = "Arial"  # 可以替换为任何你想用的字体，如"Arial", "Times New Roman", "黑体"等
        
        # 定义不同大小的字体
        self.title_font = (self.default_font_family, 16, "bold")
        self.message_font = (self.default_font_family, 12)
        self.name_font = (self.default_font_family, 12, "bold")
        self.status_font = (self.default_font_family, 10)
        self.timestamp_font = (self.default_font_family, 9)
                
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)
        
        # Create chat display
        self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Create status bar
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Instruction label
        self.instruction_label = ctk.CTkLabel(
            self.status_frame, 
            text="自动语音检测已启用, 'Space' 跳过语音/暂停分析",
            font=("Arial", 10)
        )
        self.instruction_label.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        
        # 检查头像图片是否存在
        ai_avatar_path = "./浙江清华长三角研究院.png"  # 在程序目录下放置此图片
        user_avatar_path = "./user.png"  # 在程序目录下放置此图片
        
        # 加载头像（如果本地图片存在则使用本地图片，否则使用生成的圆形）
        self.ai_avatar = self.create_circle_avatar((50, 50), "blue", "DS", image_path=ai_avatar_path)
        self.user_avatar = self.create_circle_avatar((50, 50), "green", "USER", image_path=user_avatar_path)
        
        # Add welcome message
        self.chat_row = 0
        self.add_ai_message("欢迎使用监控助手! 我会实时分析摄像头画面并回应。"
                        "系统已启用自动语音检测，直接说话即可。空格键可跳过当前语音播放并暂停/恢复分析。")
    

    def create_circle_avatar(self, size, color, text, image_path=None):
        """创建一个圆形头像，可以使用本地图片或生成带文字的圆形"""
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        
        if image_path and os.path.exists(image_path):
            try:
                # 加载本地图片
                original_img = Image.open(image_path)
                # 调整大小
                original_img = original_img.resize(size, Image.LANCZOS)
                
                # 创建一个透明的圆形遮罩
                mask = Image.new('L', size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, size[0], size[1]), fill=255)
                
                # 将图片裁剪成圆形
                img = Image.new('RGBA', size, (0, 0, 0, 0))
                img.paste(original_img, (0, 0), mask)
                
                # 转换为CTkImage
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                return ctk_img
                
            except Exception as e:
                print(f"加载头像图片出错: {e}, 使用默认头像")
                # 如果图片加载失败，回退到默认头像
                pass
        
        # 如果没有提供图片路径或加载失败，生成默认头像
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 绘制圆形
        cx, cy = size[0] // 2, size[1] // 2
        radius = min(cx, cy) - 2
        
        if color == "blue":
            fill_color = (0, 100, 200, 255)
        elif color == "green":
            fill_color = (0, 150, 100, 255)
        else:
            fill_color = (100, 100, 100, 255)
        
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill_color)
        
        # 添加文字
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            # font = ImageFont.truetype("arial.ttf", size=radius // 2)
            font = ImageFont.truetype(font_path, size=radius // 2) 
        except IOError:
            font = ImageFont.load_default()
        
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (radius, radius//2)
        draw.text((cx - text_width // 2, cy - text_height // 2), text, fill="white", font=font)
        
        # 转换为CTkImage
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        return ctk_img

    
    def setup_key_bindings(self):
        """Set up keyboard shortcuts"""
        self.bind("<r>", lambda e: self.start_voice_recording())
        self.bind("<s>", lambda e: self.stop_voice_recording())
        self.bind("<space>", lambda e: self.skip_audio())
        
        # Also add keyboard module hotkeys for global control
        # keyboard.add_hotkey('r', self.start_voice_recording)
        # keyboard.add_hotkey('s', self.stop_voice_recording)
        # keyboard.add_hotkey('space', self.skip_audio)
    
    def start_processing_thread(self):
        """Start the background message processing thread"""
        self.processing_running = True
        self.processing_thread = threading.Thread(target=self.process_message_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_message_queue(self):
        """Process messages from the priority queue"""
        while self.processing_running:
            try:
                if not self.message_queue.empty():
                    # Get message with priority (lower number = higher priority)
                    priority, msg_id, message = self.message_queue.get()
                    print(f"处理消息: 类型={message['type']}, 优先级={priority}, ID={msg_id}")
                    self.handle_message(message, msg_id)
                    self.message_queue.task_done()
                time.sleep(0.05)  # Reduced sleep for more responsiveness
            except Exception as e:
                error_msg = f"Error processing message: {e}"
                print(error_msg)
                self.update_status(error_msg)
    
    def handle_message(self, message, msg_id=None):
        """Handle different message types from the queue"""
        try:
            if message["type"] == "image_analysis":
                # Check if this is a placeholder update
                if "placeholder_id" in message and message["placeholder_id"]:
                    placeholder_id = message["placeholder_id"]
                    print(f"更新图像分析占位符: {placeholder_id}")
                    self.update_placeholder(
                        placeholder_id, 
                        message["content"], 
                        screenshots=message.get("screenshots", [])
                    )
                else:
                    print(f"处理新图像分析")
                    self.process_image_analysis(
                        message["content"], 
                        message.get("urls", []), 
                        message.get("screenshots", [])
                    )
            elif message["type"] == "voice_input":
                print(f"处理语音输入: {message['content']}")
                self.process_voice_input(
                    message["content"],
                    placeholder_id=message.get("placeholder_id")
                )
        except Exception as e:
            error_msg = f"处理消息时出错: {e}"
            print(error_msg)
            self.update_status(error_msg)
    
    def update_placeholder(self, placeholder_id, new_content, screenshots=None):
        """Update a placeholder message with actual content"""
        print(f"更新占位符: {placeholder_id} 内容长度: {len(new_content)}")
        if placeholder_id in self.placeholder_map:
            print(f"找到占位符在位置: {self.placeholder_map[placeholder_id]}")
            
            # Actually replace the placeholder with real content
            if placeholder_id.startswith("img_"):
                # This is an image analysis placeholder - add the real analysis
                print(f"添加图像分析结果到UI: {new_content[:50]}...")
                
                # Store the analysis for context
                self.last_image_analysis = new_content
                
                # Find the old row number
                row_num = self.placeholder_map[placeholder_id]
                
                # Get the frame within the chat_frame at that row
                for widget in self.chat_frame.winfo_children():
                    if int(widget.grid_info()['row']) == row_num:
                        frame = widget
                        # Find the text label within the frame
                        for child in frame.winfo_children():
                            if isinstance(child, ctk.CTkLabel) and child.cget("text") == "正在分析当前画面...":
                                # Update the label text
                                child.configure(text=f"📷 {new_content}")
                                # Change the appearance from placeholder to normal
                                frame.configure(fg_color=("#EAEAEA", "#2B2B2B"))
                                child.configure(text_color=("black", "white"))
                                print(f"成功更新占位符内容")
                                break
                
                # Extract behavior type for logging
                behavior_num, behavior_desc = extract_behavior_type(new_content)
                
                # Now generate an AI response based on the analysis
                try:
                    print("调用DeepSeek生成回应...")
                    messages = [
                        self.system_message,
                        {"role": "user", "content": f"基于这个观察: {new_content}, 根据检测到的行为类型给出相应回应。如果是工作或喝水，给予鼓励；如果是吃东西、玩手机、喝饮料或睡觉，给予批评和提醒."}
                    ]
                    
                    response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=False
                    )
                    assistant_reply = response.choices[0].message.content
                    print(f"DeepSeek回应: {assistant_reply}")
                    
                    # Add the AI response to the chat
                    self.add_ai_message(assistant_reply)
                    
                    # Play the text as audio
                    self.audio_player.play_text(assistant_reply)
                except Exception as e:
                    error_msg = f"DeepSeek API错误: {e}"
                    print(error_msg)
                    self.update_status(error_msg)
                
                # Remove the placeholder from our tracking map
                del self.placeholder_map[placeholder_id]
            
            elif placeholder_id.startswith("voice_"):
                # This is a voice input placeholder - we'll handle in process_voice_input
                pass
    

    def process_voice_input(self, text, placeholder_id=None):
        """Process voice input and generate AI response with historical context"""
        print(f"处理语音输入: '{text}'")
        
        # 调试信息：检查observation_history的内容
        print(f"当前observation_history长度: {len(self.observation_history)}")
        for i, obs in enumerate(self.observation_history):
            print(f"记录[{i}]: 行为={obs['behavior_num']}-{obs['behavior_desc']}, 时间={datetime.fromtimestamp(obs['timestamp']).strftime('%H:%M:%S')}")
        
        # 跳过当前音频播放
        print("打断当前语音播放")
        self.audio_player.skip_current()
        
        # 临时禁用语音检测
        was_playing_audio = self.is_playing_audio
        self.is_playing_audio = True
        
        # 记录语音处理开始时间
        voice_start_time = time.time()
        
        # 添加用户消息到UI
        self.add_user_message(text)
        
        # 定义行为映射表
        behavior_map = {
            "1": "正确佩戴安全帽和防护服且在作业区域内",
            "2": "未佩戴安全帽或防护服",
            "3": "在作业区域内",
            "4": "在作业区域外"
        }
        
        # 创建行为统计摘要
        safety_duration = time.time() - self.safety_start_time if self.safety_start_time > 0 else 0
        
        # 检查是否询问特定行为
        is_asking_about_equipment = any(keyword in text for keyword in ["有没有戴安全帽", "戴安全帽了吗", "防护服", "安全装备"])
        is_asking_about_zone = any(keyword in text for keyword in ["在作业区域", "在区域内", "在区域外"])
        is_asking_about_behavior = is_asking_about_equipment or is_asking_about_zone or "我做了什么" in text
        
        # 创建相关行为的详细记录
        relevant_history = []
        behavior_filter = None
        
        if is_asking_about_equipment:
            behavior_filter = "2"  # 未佩戴安全装备的行为编号
            print("检测到用户询问安全装备相关行为")
        elif is_asking_about_zone:
            behavior_filter = "4"  # 在作业区域外的行为编号
            print("检测到用户询问作业区域相关行为")
        
        # *** 修改：如果behavior_counters显示有相关行为，但observation_history为空，则从日志文件恢复 ***
        if behavior_filter and len(self.observation_history) == 0:
            # 先检查行为计数器
            behavior_key = {
                "1": "proper_equipment",
                "2": "missing_equipment",
                "3": "in_work_zone",
                "4": "out_of_zone"
            }.get(behavior_filter, "other")
            
            # 如果行为计数器显示有这个行为，但observation_history为空，则添加一个恢复记录
            if self.behavior_counters.get(behavior_key, 0) > 0:
                print(f"检测到计数器显示存在{behavior_key}行为，但observation_history为空，添加恢复记录")
                behavior_desc = behavior_map.get(behavior_filter, "未知行为")
                
                # 创建恢复记录
                recovery_observation = {
                    "timestamp": time.time() - 300,  # 假设发生在5分钟前
                    "behavior_num": behavior_filter,
                    "behavior_desc": behavior_desc,
                    "analysis": f"系统检测到{behavior_desc}（从行为计数器恢复的记录）"
                }
                self.observation_history.append(recovery_observation)
                print(f"已从行为计数器恢复记录：{behavior_filter}-{behavior_desc}")
        
        # 如果是询问特定行为，搜索所有历史记录
        if behavior_filter:
            print(f"搜索历史记录中的行为编号: {behavior_filter}")
            for obs in reversed(self.observation_history):
                print(f"比较: {obs['behavior_num']} vs {behavior_filter}, 类型: {type(obs['behavior_num'])} vs {type(behavior_filter)}")
                if str(obs['behavior_num']) == str(behavior_filter):  # 确保类型一致
                    obs_time = datetime.fromtimestamp(obs["timestamp"]).strftime("%H:%M:%S")
                    relevant_history.append(f"- {obs_time}: {obs['behavior_desc']} - {obs['analysis'][:150]}...")
                    print(f"找到匹配记录: {obs_time}")
            
            if not relevant_history:
                # *** 修改：检查行为计数器 ***
                behavior_key = {
                    "1": "proper_equipment",
                    "2": "missing_equipment",
                    "3": "in_work_zone",
                    "4": "out_of_zone"
                }.get(behavior_filter, "other")
                
                if self.behavior_counters.get(behavior_key, 0) > 0:
                    # 如果计数器显示有这个行为，但没有找到记录，添加基于计数器的回复
                    relevant_history.append(f"根据系统记录，今天有过{behavior_map.get(behavior_filter, '未知')}的情况（从行为计数器推断）")
                    print(f"未找到行为编号为{behavior_filter}的历史记录，但计数器显示有这个行为")
                else:
                    relevant_history.append(f"未在历史记录中找到相关的'{behavior_map.get(behavior_filter, '未知')}'情况")
                    print(f"未找到行为编号为{behavior_filter}的历史记录")
        
        # 创建最近行为的详细记录（最多5条）
        recent_observations = []
        for obs in reversed(self.observation_history[-5:]):
            obs_time = datetime.fromtimestamp(obs["timestamp"]).strftime("%H:%M:%S")
            behavior_desc = obs["behavior_desc"]
            analysis_brief = obs["analysis"][:100] + ("..." if len(obs["analysis"]) > 100 else "")
            recent_observations.append(f"- {obs_time}: {behavior_desc} - {analysis_brief}")
        
        recent_observations_text = "\n".join(recent_observations)
        if not recent_observations:
            recent_observations_text = "没有最近的安全合规记录"
        
        # 添加最后一次观察的完整内容
        last_observation = ""
        if self.observation_history:
            last_obs = self.observation_history[-1]
            last_time = datetime.fromtimestamp(last_obs["timestamp"]).strftime("%H:%M:%S")
            last_observation = f"最后一次观察 ({last_time}):\n{last_obs['analysis']}"
        else:
            last_observation = "没有观察记录"
        
        # 构建上下文信息，包括特定行为查询结果
        context_summary = f"""
    当前安全合规统计：
    - 正确佩戴安全装备且在作业区域内: {self.behavior_counters['proper_equipment']}次
    - 未佩戴安全装备: {self.behavior_counters['missing_equipment']}次 {'(检测到用户询问此情况)' if is_asking_about_equipment else ''}
    - 在作业区域内: {self.behavior_counters['in_work_zone']}次
    - 在作业区域外: {self.behavior_counters['out_of_zone']}次 {'(检测到用户询问此情况)' if is_asking_about_zone else ''}
    - 安全合规持续时间: {int(safety_duration/60)}分钟

    """

        # 如果询问特定行为，添加相关历史记录
        if is_asking_about_behavior and relevant_history:
            context_summary += f"""
    相关安全合规历史记录:
    {chr(10).join(relevant_history)}

    """
        
        # 添加最近观察记录
        context_summary += f"""
    最近的安全合规记录:
    {recent_observations_text}

    {last_observation}
    """
        
        # 将用户问题添加到聊天上下文
        user_message = {"role": "user", "content": f"{context_summary}\n\n用户说: {text}"}
        self.chat_context.append(user_message)
        
        # 限制上下文长度
        if len(self.chat_context) > 20:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]
        
        try:
            print(f"调用DeepSeek生成回应，消息历史长度: {len(self.chat_context)}")
            
            # 使用完整的对话历史发送请求
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=self.chat_context,
                stream=False
            )
            assistant_reply = response.choices[0].message.content
            print(f"DeepSeek回应: {assistant_reply}")
            
            # 记录语音处理结束时间
            voice_end_time = time.time()
            print(f"语音处理总耗时: {voice_end_time - voice_start_time:.2f}秒")
            
            # 将AI回应添加到对话历史
            assistant_message = {"role": "assistant", "content": assistant_reply}
            self.chat_context.append(assistant_message)
            
            # 添加AI回应到聊天记录
            self.add_ai_message(assistant_reply)
            
            # 使用高优先级播放回复
            self.audio_player.play_text(assistant_reply, priority=1)
        except Exception as e:
            error_msg = f"DeepSeek API error: {e}"
            print(error_msg)
            self.update_status(error_msg)
            # 恢复原来的语音检测状态
            self.is_playing_audio = was_playing_audio


    def process_image_analysis(self, analysis_text, image_urls, screenshots, placeholder_id=None):
        """Process image analysis results, track behavior patterns, and generate context-aware AI response"""
        print(f"处理图像分析: 分析长度 {len(analysis_text)} 字符, 占位符ID: {placeholder_id}")
        
        # 提取行为类型
        behavior_num, behavior_desc = extract_behavior_type(analysis_text)
        
        # 记录到日志
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
        logging.info(log_message)
        print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
        
        # 存储观察记录
        current_time = time.time()
        observation = {
            "timestamp": current_time,
            "behavior_num": behavior_num,  # 确保这里存储的是字符串类型
            "behavior_desc": behavior_desc,
            "analysis": analysis_text
        }
        
        # 将观察添加到历史记录，保留最近20条
        self.observation_history.append(observation)
        
        # 调试信息：确认添加成功
        print(f"已添加新行为到observation_history: {behavior_num}-{behavior_desc}, 当前长度: {len(self.observation_history)}")
        
        if len(self.observation_history) > 20:
            self.observation_history.pop(0)  # 保留最近20条
            
        # 更新行为计数器
        behavior_map = {
            "1": "work",
            "2": "eating",
            "3": "drinking_water",
            "4": "drinking_beverage",
            "5": "phone",
            "6": "sleeping",
            "7": "other"
        }
        current_behavior = behavior_map.get(behavior_num, "other")
        self.behavior_counters[current_behavior] += 1
        print(f"行为计数更新: {current_behavior} = {self.behavior_counters[current_behavior]}")
        
        # 跟踪持续行为
        if self.last_behavior == current_behavior:
            behavior_duration = current_time - self.continuous_behavior_time
        else:
            self.continuous_behavior_time = current_time
            behavior_duration = 0
        
        self.last_behavior = current_behavior
        
        # 如果是新的坐姿行为（不是站起来活动），更新坐姿开始时间
        if current_behavior not in ["other"]:  # 假设"other"可能包括站起来活动
            # 如果之前没有记录坐姿开始时间，记录当前时间
            if self.sitting_start_time == 0:
                self.sitting_start_time = current_time
        else:
            # 重置坐姿计时器
            self.sitting_start_time = 0
        
        # 分析是否需要提醒或鼓励
        sitting_duration = current_time - self.sitting_start_time if self.sitting_start_time > 0 else 0
        should_remind = False
        reminder_type = None
        
        # 判断是否需要提醒
        if current_behavior == "eating" and self.behavior_counters["eating"] >= self.reminder_thresholds["eating"] and \
        current_time - self.last_reminder_time["eating"] > self.reminder_interval:
            should_remind = True
            reminder_type = "eating"
            self.last_reminder_time["eating"] = current_time
        
        elif current_behavior == "drinking_beverage" and self.behavior_counters["drinking_beverage"] >= self.reminder_thresholds["drinking_beverage"] and \
            current_time - self.last_reminder_time["drinking_beverage"] > self.reminder_interval:
            should_remind = True
            reminder_type = "drinking_beverage"
            self.last_reminder_time["drinking_beverage"] = current_time
        
        elif current_behavior == "phone" and self.behavior_counters["phone"] >= self.reminder_thresholds["phone"] and \
            current_time - self.last_reminder_time["phone"] > self.reminder_interval:
            should_remind = True
            reminder_type = "phone"
            self.last_reminder_time["phone"] = current_time
        
        elif sitting_duration > self.reminder_thresholds["sitting"] and \
            current_time - self.last_reminder_time["sitting"] > self.reminder_interval:
            should_remind = True
            reminder_type = "sitting"
            self.last_reminder_time["sitting"] = current_time
        
        # 判断是否需要鼓励
        should_encourage = False
        if (current_behavior == "work" and behavior_duration > 10*60) or \
        (current_behavior == "drinking_water") and \
        (current_time - self.last_reminder_time["encouragement"] > self.reminder_interval):
            should_encourage = True
            self.last_reminder_time["encouragement"] = current_time
        
        # 存储最新的图像分析作为上下文
        self.last_image_analysis = analysis_text
        
        # 添加图像分析到聊天记录
        if not placeholder_id or placeholder_id not in self.placeholder_map:
            if screenshots and len(screenshots) > 0:
                print(f"添加新的图像分析到UI，带截图")
                self.add_ai_message(f"📷 {analysis_text}", screenshots[0], placeholder_id=placeholder_id)
            else:
                print(f"添加新的图像分析到UI，无截图")
                self.add_ai_message(f"📷 {analysis_text}", placeholder_id=placeholder_id)
        
        # 根据分析结果构建提示
        prompt_instruction = ""
        if should_remind:
            if reminder_type == "missing_equipment":
                prompt_instruction = "用户未佩戴安全装备，请严厉警告并要求立即佩戴安全帽和防护服，这是强制要求！"
            elif reminder_type == "out_of_zone":
                prompt_instruction = "用户离开作业区域，请严厉警告并要求立即回到作业区域内，这里不安全！"
        elif should_encourage:
            if current_behavior == "proper_equipment":
                prompt_instruction = "用户正确佩戴安全装备且在作业区域内，请给予肯定和鼓励，继续保持安全合规。"
            elif current_behavior == "in_work_zone":
                prompt_instruction = "用户在作业区域内，请表示赞同，鼓励继续保持安全作业。"
        else:
            # 如果没有特殊提示，使用一般性提示
            prompt_instruction = f"根据检测到的安全合规情况'{behavior_desc}'给出相应回应。如果正确佩戴安全装备且在作业区域内，给予鼓励；如果未佩戴安全装备或离开作业区域，给予严厉警告。"
        
        # 添加当前观察到聊天上下文
        user_message = {"role": "user", "content": f"观察结果: {analysis_text}\n\n{prompt_instruction}"}
        self.chat_context.append(user_message)
        
        # 限制上下文长度，避免超出token限制
        if len(self.chat_context) > 20:  # 保留最近20条消息
            # 保留系统消息和最近的消息
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]
        
        try:
            print(f"调用DeepSeek生成回应，消息历史长度: {len(self.chat_context)}")
            
            # 使用完整的聊天上下文
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=self.chat_context,  # 使用累积的对话历史
                stream=False
            )
            assistant_reply = response.choices[0].message.content
            print(f"DeepSeek回应: {assistant_reply}")
            
            # 将AI回应也添加到对话历史
            assistant_message = {"role": "assistant", "content": assistant_reply}
            self.chat_context.append(assistant_message)
            
            # 添加AI回应到聊天记录
            self.add_ai_message(assistant_reply)
            
            # 只有在需要提醒或鼓励时才播放语音
            if should_remind or should_encourage:
                self.audio_player.play_text(assistant_reply, priority=2)
        except Exception as e:
            error_msg = f"DeepSeek API error: {e}"
            print(error_msg)
            self.update_status(error_msg)



    def check_timestamp(self):
        """Check if we need to display a new timestamp"""
        current_time = time.time()
        if current_time - self.last_timestamp >= self.timestamp_interval:
            self.add_timestamp()
            self.last_timestamp = current_time
        
        # Schedule the next check
        self.after(5000, self.check_timestamp)  # Check every 5 seconds
    
    def add_timestamp(self):
        """Add a timestamp to the chat UI"""
        # Get current time in the required format
        now = datetime.now()
        time_str = now.strftime("%m月%d日 %H:%M")
        
        # Create timestamp frame
        timestamp_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#E0E0E0", "#3F3F3F"), corner_radius=15)
        timestamp_frame.grid(row=self.chat_row, column=0, pady=5)
        self.chat_row += 1
        
        # Add timestamp label - 使用自定义时间戳字体
        timestamp_label = ctk.CTkLabel(
            timestamp_frame, 
            text=time_str,
            font=self.timestamp_font,
            fg_color="transparent",
            padx=10,
            pady=2
        )
        timestamp_label.grid(row=0, column=0)
        
        # Scroll to bottom
        self.scroll_to_bottom()
    
    def add_ai_message(self, text, screenshot=None, is_placeholder=False, placeholder_id=None):
        """Add an AI message to the chat UI"""
        # Generate placeholder id if needed
        if is_placeholder and not placeholder_id:
            placeholder_id = f"ai_{self.message_id}"
            self.message_id += 1
        
        print(f"添加AI消息: 长度={len(text)}, 有截图={screenshot is not None}, 是占位符={is_placeholder}, ID={placeholder_id}")
        
        # Create message frame
        message_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#EAEAEA", "#2B2B2B"))
        message_frame.grid(row=self.chat_row, column=0, sticky="w", padx=5, pady=5)
        message_frame.grid_columnconfigure(1, weight=1)
        
        # Store placeholder row if needed
        if is_placeholder and placeholder_id:
            self.placeholder_map[placeholder_id] = self.chat_row
            print(f"存储占位符 {placeholder_id} 在行 {self.chat_row}")
        
        self.chat_row += 1
        
        # Add avatar
        avatar_label = ctk.CTkLabel(message_frame, image=self.ai_avatar, text="")
        avatar_label.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        
        # Add name
        name_label = ctk.CTkLabel(message_frame, text="DeepSeek", font=("Arial", 12, "bold"), 
                                  anchor="w", fg_color="transparent")
        name_label.grid(row=0, column=1, sticky="w", padx=5, pady=(5, 0))
        
        # Add screenshot if provided
        if screenshot is not None:
            try:
                # Create a frame for the image
                img_frame = ctk.CTkFrame(message_frame, fg_color="transparent")
                img_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)
                
                # Ensure we have a valid image
                if hasattr(screenshot, 'copy'):
                    # Resize the image for display
                    img_resized = screenshot.copy()
                    img_resized.thumbnail((200, 150))
                    
                    # Convert to CTkImage
                    ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(200, 150))
                    
                    # Create the label with the image
                    img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                    img_label.grid(row=0, column=0, padx=2, pady=2)
                    
                    # Store a reference to prevent garbage collection
                    img_label.image = ctk_img
                    
                    print(f"成功添加图片: {img_resized.size}")
                else:
                    error_msg = "图像对象无copy属性"
                    print(error_msg)
                    error_label = ctk.CTkLabel(img_frame, text=f"[图像处理错误: {error_msg}]")
                    error_label.grid(row=0, column=0, padx=2, pady=2)
            except Exception as e:
                # If image processing fails, show error
                print(f"图像处理错误: {e}")
                error_label = ctk.CTkLabel(message_frame, text=f"[图像处理错误: {str(e)}]")
                error_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            
            # Add text below the image
            text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="left", 
                                     anchor="w", fg_color="transparent")
            text_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        else:
            # Text only (no image)
            text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="left", 
                                     anchor="w", fg_color="transparent")
            text_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Mark as placeholder with different color if needed
        if is_placeholder:
            message_frame.configure(fg_color=("#F5F5F5", "#3B3B3B"))
            if screenshot is not None:
                text_label.configure(text_color=("#888888", "#AAAAAA"))
            else:
                text_label.configure(text_color=("#888888", "#AAAAAA"))
        
        # Scroll to bottom
        self.after(100, self.scroll_to_bottom)
        
        # Return placeholder id if applicable
        return placeholder_id if is_placeholder else None
    
    def add_user_message(self, text, is_placeholder=False, replace_placeholder=None, placeholder_id=None):
        """Add a user message to the chat UI"""
        print(f"添加用户消息: '{text[:30]}...', 占位符={is_placeholder}, 替换ID={replace_placeholder}, 新ID={placeholder_id}")
        
        # If replacing a placeholder, remove the old one from the map
        if replace_placeholder and replace_placeholder in self.placeholder_map:
            print(f"从映射中移除占位符: {replace_placeholder}")
            # In a full implementation, we would update the existing widget
            # But for simplicity, we just add a new message
            del self.placeholder_map[replace_placeholder]
        
        # Generate placeholder id if needed
        if is_placeholder and not placeholder_id:
            placeholder_id = f"user_{self.message_id}"
            self.message_id += 1
            print(f"生成新占位符ID: {placeholder_id}")
        
        # Create message frame
        message_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#C7E9C0", "#2D3F2D"))
        message_frame.grid(row=self.chat_row, column=0, sticky="e", padx=5, pady=5)
        
        # Store placeholder row if needed
        if is_placeholder and placeholder_id:
            self.placeholder_map[placeholder_id] = self.chat_row
            print(f"存储占位符 {placeholder_id} 在行 {self.chat_row}")
            
        self.chat_row += 1
        
        # Add avatar
        avatar_label = ctk.CTkLabel(message_frame, image=self.user_avatar, text="")
        avatar_label.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
        
        # Add name
        name_label = ctk.CTkLabel(message_frame, text="User", font=("Arial", 12, "bold"), 
                                  anchor="e", fg_color="transparent")
        name_label.grid(row=0, column=0, sticky="e", padx=5, pady=(5, 0))
        
        # Add text
        text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="right", 
                                  anchor="e", fg_color="transparent")
        text_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        
        # Mark as placeholder with different color if needed
        if is_placeholder:
            message_frame.configure(fg_color=("#DCF0D5", "#394639"))
            text_label.configure(text_color=("#888888", "#AAAAAA"))
        
        # Scroll to bottom
        self.after(100, self.scroll_to_bottom)
        
        # Return placeholder id if applicable
        return placeholder_id if is_placeholder else None
    
    def scroll_to_bottom(self):
        """更可靠地滚动聊天视图到底部"""
        try:
            # 使用after方法确保在UI更新后执行滚动
            self.after(10, lambda: self._do_scroll_to_bottom())
        except Exception as e:
            print(f"Scroll error: {e}")

    def _do_scroll_to_bottom(self):
        """实际执行滚动的内部方法"""
        try:
            # 获取可滚动区域的画布
            canvas = self.chat_frame._parent_canvas
            
            # 获取画布的内容高度
            canvas.update_idletasks()  # 确保更新布局
            
            # 明确设置滚动区域底部位置
            canvas.yview_moveto(1.0)
            
            # 额外的方法确保滚动到底部
            canvas.update_idletasks()
            canvas.yview_scroll(1000000, "units")  # 大数字确保滚动到底部
        except Exception as e:
            print(f"Detailed scroll error: {e}")
            

    
    def update_preview(self, img):
        """This method is now deprecated but kept for compatibility"""
        pass
    
    def update_status(self, text):
        """Update the status message"""
        self.status_label.configure(text=text)
    
    def analyze_images(self, image_urls, screenshots, current_screenshot, placeholder_id=None):
        """Send images to Qwen-VL for analysis"""
        if not image_urls:
            print("没有图像URL可供分析")
            return
        
        self.update_status("正在分析图像...")
        print(f"分析图像: {len(image_urls)} URLs, 占位符ID: {placeholder_id}")
        
        messages = [{
            "role": "system",
            "content": [{"type": "text", "text": "详细观察这个人的安全合规情况。务必判断他属于以下哪种情况：1.正确佩戴安全帽和防护服且在作业区域内, 2.未佩戴安全帽或防护服, 3.在作业区域内, 4.在作业区域外。分析他的安全装备佩戴情况和位置来作出判断。使用中文回答，并明确指出是哪种情况。"}]
        }]
        
        message_payload = {
            "role": "user",
            "content": [
                {"type": "video", "video": image_urls},
                {"type": "text", "text": "这个人的安全合规情况如何？请判断他是：1.正确佩戴安全帽和防护服且在作业区域内, 2.未佩戴安全帽或防护服, 3.在作业区域内, 4.在作业区域外。请详细描述你观察到的内容并明确指出判断结果。"}
            ]
        }
        messages.append(message_payload)
        
        try:
            print("调用Qwen-VL API进行图像分析...")
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
            )
            analysis_text = completion.choices[0].message.content
            print(f"图像分析完成，分析长度: {len(analysis_text)} 字符")
            
            # Extract behavior type for logging
            behavior_num, behavior_desc = extract_behavior_type(analysis_text)
            
            # Log the behavior
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
            logging.info(log_message)
            print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
            
            # Add to message queue for processing with appropriate priority
            # Priority 2 for normal image analysis (voice input would be priority 1)
            print("添加分析结果到消息队列")
            self.message_queue.put((
                2,  # priority (lower number = higher priority)
                self.message_id,  # message id for sequence
                {
                    "type": "image_analysis",
                    "content": analysis_text,
                    "urls": image_urls,
                    "screenshots": [current_screenshot] if current_screenshot else [],
                    "placeholder_id": placeholder_id
                }
            ))
            self.message_id += 1
            
        except Exception as e:
            error_msg = f"Qwen-VL API error: {e}"
            print(error_msg)
            self.update_status(error_msg)
    
    def transcribe_audio(self, audio_file, priority=False, placeholder_id=None):
        """Transcribe recorded audio using SenseVoice"""
        self.update_status("正在转录语音...")
        print(f"转录音频: {audio_file}, 优先级: {priority}, 占位ID: {placeholder_id}")
        
        try:
            # Check if the file exists
            if not os.path.exists(audio_file):
                error_msg = f"音频文件不存在: {audio_file}"
                print(error_msg)
                self.update_status(error_msg)
                return
            
            # Check file size
            file_size = os.path.getsize(audio_file)
            print(f"音频文件大小: {file_size} 字节")
            if file_size == 0:
                error_msg = "音频文件为空"
                print(error_msg)
                self.update_status(error_msg)
                return
            
            print("调用ASR模型转录...")
            res = asr_model.generate(
                input=audio_file,
                cache={},
                language="auto",
                use_itn=False,
                ban_emo_unk=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            print(f"ASR结果: {res}")
            
            if len(res) > 0 and "text" in res[0]:
                text = res[0]["text"]
                extracted_text = extract_language_emotion_content(text)
                print(f"提取的文本内容: {extracted_text}")
                
                # 新增：检查提取的文本是否为空或太短（可能是噪音）
                if not extracted_text or len(extracted_text.strip()) < 2:
                    print(f"检测到空语音或噪音: '{extracted_text}'，忽略处理")
                    self.update_status("检测到噪音，忽略")
                    return
                
                # Add to message queue with high priority if requested
                priority_level = 1 if priority else 2
                
                print(f"添加语音输入到消息队列，优先级: {priority_level}")
                self.message_queue.put((
                    priority_level,  # priority (lower number = higher priority)
                    self.message_id,  # message id for sequence
                    {
                        "type": "voice_input",
                        "content": extracted_text,
                        "placeholder_id": placeholder_id
                    }
                ))
                self.message_id += 1
                
                # Voice inputs should interrupt the current analysis cycle
                if priority:
                    print("语音输入优先，跳过当前语音播放")
                    self.audio_player.skip_current()
            else:
                error_msg = "未检测到语音或转录失败"
                print(error_msg)
                self.update_status(error_msg)
                
        except Exception as e:
            error_msg = f"转录错误: {e}"
            print(error_msg)
            self.update_status(error_msg)
    
    def start_voice_recording(self):
        """Start recording voice when 'r' key is pressed"""
        # This is retained for backwards compatibility, but the continuous
        # voice detection has replaced this functionality
        self.update_status("使用自动语音检测 - 直接说话即可")
    
    def stop_voice_recording(self):
        """Stop recording voice when 's' key is pressed"""
        # This is retained for backwards compatibility, but the continuous
        # voice detection has replaced this functionality
        pass
    
    def skip_audio(self):
        """Skip currently playing audio and toggle analysis pause when spacebar is pressed"""
        self.audio_player.skip_current()
        self.webcam_handler.toggle_pause()
        
        # Show/hide camera window
        if self.webcam_handler.camera_window and self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.create_camera_window()
        elif self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()

    def _analyze_behavior(self, behavior_type):
        """分析行为并生成相应的回应"""
        current_time = time.time()
        
        # 更新行为计数器
        if behavior_type == 1:  # 正确佩戴安全装备
            self.behavior_counters["proper_equipment"] += 1
            behavior_key = "proper_equipment"
        elif behavior_type == 2:  # 未佩戴安全装备
            self.behavior_counters["missing_equipment"] += 1
            behavior_key = "missing_equipment"
        elif behavior_type == 3:  # 在作业区域内
            self.behavior_counters["in_work_zone"] += 1
            behavior_key = "in_work_zone"
        elif behavior_type == 4:  # 在作业区域外
            self.behavior_counters["out_of_zone"] += 1
            behavior_key = "out_of_zone"
        else:
            return None

        # 检查是否需要发出提醒
        if behavior_key in self.reminder_thresholds:
            if (self.behavior_counters[behavior_key] >= self.reminder_thresholds[behavior_key] and
                current_time - self.last_reminder_time.get(behavior_key, 0) >= self.reminder_interval):
                self.last_reminder_time[behavior_key] = current_time
                
                # 生成提醒消息
                if behavior_key == "missing_equipment":
                    return "警告：请立即佩戴安全帽和防护服！这是强制要求！"
                elif behavior_key == "out_of_zone":
                    return "警告：请立即回到作业区域内！这里不安全！"
        
        # 生成常规回应
        if behavior_type == 1:
            return "很好！请继续保持正确的安全装备佩戴。"
        elif behavior_type == 2:
            return "警告：必须佩戴安全帽和防护服才能作业！"
        elif behavior_type == 3:
            return "很好！请继续在作业区域内安全作业。"
        elif behavior_type == 4:
            return "警告：请立即回到作业区域内！"

        return None

    def _update_behavior_stats(self, behavior_type):
        """更新行为统计信息"""
        current_time = time.time()
        
        # 更新行为历史
        self.observation_history.append({
            "time": current_time,
            "behavior": behavior_type
        })
        
        # 保持历史记录在合理范围内
        if len(self.observation_history) > 100:
            self.observation_history = self.observation_history[-100:]
        
        # 更新持续行为时间
        if self.last_behavior == behavior_type:
            self.continuous_behavior_time = current_time
        else:
            self.last_behavior = behavior_type
            self.continuous_behavior_time = current_time

    def process_frame(self, frame):
        """处理每一帧图像"""
        if frame is None:
            return None

        # 更新帧率
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

        # 检测安全装备和位置
        has_safety_gear = self._detect_safety_gear(frame)
        in_work_zone = self._detect_work_zone(frame)

        # 确定行为类型
        if has_safety_gear and in_work_zone:
            behavior_type = 1  # 正确佩戴安全装备且在作业区域内
        elif not has_safety_gear:
            behavior_type = 2  # 未佩戴安全装备
        elif in_work_zone:
            behavior_type = 3  # 在作业区域内
        else:
            behavior_type = 4  # 在作业区域外

        # 更新行为统计
        self._update_behavior_stats(behavior_type)

        # 分析行为并获取回应
        response = self._analyze_behavior(behavior_type)
        if response:
            self._update_chat_context(response)

        # 在图像上绘制信息
        frame = self._draw_info(frame, has_safety_gear, in_work_zone)

        return frame

    def _detect_safety_gear(self, frame):
        """检测是否佩戴安全装备"""
        # 这里应该实现实际的安全装备检测逻辑
        # 目前使用随机值模拟
        return random.random() > 0.3

    def _detect_work_zone(self, frame):
        """检测是否在作业区域内"""
        # 这里应该实现实际的作业区域检测逻辑
        # 目前使用随机值模拟
        return random.random() > 0.2

    def _draw_info(self, frame, has_safety_gear, in_work_zone):
        """在图像上绘制信息"""
        # 绘制帧率
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制安全状态
        safety_status = "安全装备: " + ("已佩戴" if has_safety_gear else "未佩戴")
        zone_status = "作业区域: " + ("在区域内" if in_work_zone else "在区域外")
        
        cv2.putText(frame, safety_status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, zone_status, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制警告框
        if not has_safety_gear or not in_work_zone:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                         (0, 0, 255), 3)

        return frame

# ---------------- Main Function ----------------
def main():
    # Set appearance mode and default theme
    ctk.set_appearance_mode("System")  # "System", "Dark" or "Light"
    ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
    
    app = MultimediaAssistantApp()
    app.protocol("WM_DELETE_WINDOW", lambda: quit_app(app))
    app.mainloop()

def quit_app(app):
    """Clean shutdown of the application"""
    # Stop all threads
    if hasattr(app, 'webcam_handler'):
        app.webcam_handler.stop()
    
    if hasattr(app, 'voice_detector'):
        app.voice_detector.stop_monitoring()
    
    if hasattr(app, 'processing_running'):
        app.processing_running = False
        
    if hasattr(app, 'audio_player'):
        app.audio_player.stop()
        
    # Clean up keyboard handlers
    # keyboard.unhook_all()
    
    # Clean up temporary files
    try:
        for file in os.listdir():
            if file.startswith("output") and (file.endswith(".mp3") or file.endswith(".wav")):
                os.remove(file)
                print(f"删除临时文件: {file}")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")
    
    # Close the app
    app.destroy()

if __name__ == "__main__":
    main()