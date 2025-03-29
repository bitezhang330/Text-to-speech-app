import sys
import requests
import json
import io
import threading
import os

# --- UI 和音频库 ---
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QPushButton, QMessageBox, QSizePolicy, QComboBox, QDoubleSpinBox,
    QFormLayout, QTabWidget, QFileDialog, QGroupBox, QProgressBar, QToolButton,
    QStyleFactory, QFrame, QSpacerItem
)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette, QPixmap

try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False
    print("警告: 未找到 pygame 库。音频播放功能将被禁用。请使用 'pip install pygame' 命令安装。")


# --- 配置 ---
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
TRANSLATION_MODEL = "Qwen/QwQ-32B"  # 翻译模型
TTS_MODEL_BASE = "FunAudioLLM/CosyVoice2-0.5B"  # TTS 基础模型

# --- 应用程序样式表 ---
APP_STYLESHEET = """
QWidget {
    font-family: "Microsoft YaHei", Arial, sans-serif;
    font-size: 10pt;
}

QTabWidget::pane {
    border: 1px solid #cccccc;
    border-radius: 4px;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f0f0f0;
    border: 1px solid #cccccc;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    border-bottom: 1px solid #ffffff;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #cccccc;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}

QPushButton {
    background-color: #4a86e8;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3a76d8;
}

QPushButton:pressed {
    background-color: #2a66c8;
}

QPushButton:disabled {
    background-color: #cccccc;
    color: #888888;
}

QLineEdit, QTextEdit, QComboBox, QDoubleSpinBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}

QTextEdit {
    background-color: #ffffff;
}

QLabel#statusLabel {
    background-color: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px;
    font-weight: bold;
}

QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 4px;
}

QToolButton:hover {
    background-color: #f0f0f0;
}
"""

# --- 用于线程处理的 Worker 对象 ---
class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    translation_ready = pyqtSignal(str)
    audio_ready = pyqtSignal(bytes)
    progress_update = pyqtSignal(int)  # 新增进度信号

class TranslationWorker(QObject):
    def __init__(self, api_key, text_to_translate):
        super().__init__()
        self.signals = WorkerSignals()
        self.api_key = api_key
        self.text_to_translate = text_to_translate
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if not self.is_running: return
            self.signals.status_update.emit("正在翻译文本...")
            self.signals.progress_update.emit(30)  # 开始翻译
            translated_text = self.translate_text_api(self.text_to_translate)
            self.signals.progress_update.emit(90)  # 翻译接近完成

            if translated_text and self.is_running:
                self.signals.translation_ready.emit(translated_text)
                self.signals.status_update.emit("翻译完成")
                self.signals.progress_update.emit(100)  # 完成
            elif self.is_running:
                self.signals.error.emit("翻译文本失败。")
        except requests.exceptions.RequestException as e:
            if self.is_running:
                self.signals.error.emit(f"网络错误: {e}")
        except Exception as e:
             if self.is_running:
                status_code = getattr(e, 'response', None)
                if status_code is not None: status_code = status_code.status_code
                self.signals.error.emit(f"发生错误 (状态码: {status_code}): {e}")
        finally:
            if self.is_running:
                self.signals.finished.emit()

    def translate_text_api(self, text_to_translate):
        """调用 SiliconFlow API 进行翻译"""
        url = f"{SILICONFLOW_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        prompt = f"Translate the following text to Simplified Chinese. Output only the translation itself, without any introductory phrases.\n\nOriginal Text:\n{text_to_translate}\n\nSimplified Chinese Translation:"

        payload = {
            "model": TRANSLATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.5,
            "stream": False,
            "stop": None,
            "top_p": 0.7,
            "frequency_penalty": 0.0,
            "n": 1,
            "response_format": {"type": "text"},
        }
        print("翻译接口 Payload:", json.dumps(payload, indent=2))

        response = requests.post(url, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            error_details = response.text
            try:
                error_json = json.loads(error_details)
                error_details = json.dumps(error_json, indent=2)
            except json.JSONDecodeError:
                pass
            print(f"翻译 API 错误 - 状态码: {response.status_code}\n来自 API 的详细信息:\n{error_details}")
            response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            if content.startswith('"') and content.endswith('"'):
                 content = content[1:-1].strip()
            if content:
                 return content
            else:
                 print("翻译警告 - 收到空内容:", result)
                 raise ValueError("从 API 收到了空的翻译内容。")
        else:
            print("翻译错误 - 即便状态码 200 OK，返回格式仍无效:", result)
            raise ValueError("翻译 API 返回格式无效 (缺少 choices/message/content)。")

class TTSWorker(QObject):
    def __init__(self, api_key, text_to_speak, tts_voice_name, tts_speed, tts_gain):
        super().__init__()
        self.signals = WorkerSignals()
        self.api_key = api_key
        self.text_to_speak = text_to_speak
        self.tts_voice_name = tts_voice_name
        self.tts_speed = tts_speed
        self.tts_gain = tts_gain
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if not self.is_running: return
            self.signals.status_update.emit("正在合成语音...")
            self.signals.progress_update.emit(30)  # 开始语音合成
            audio_data = self.text_to_speech_api(
                self.text_to_speak,
                self.tts_voice_name,
                self.tts_speed,
                self.tts_gain
            )
            self.signals.progress_update.emit(90)  # 语音合成接近完成

            if audio_data and self.is_running:
                self.signals.audio_ready.emit(audio_data)
                self.signals.status_update.emit("音频已就绪。")
                self.signals.progress_update.emit(100)  # 完成
            elif self.is_running:
                self.signals.error.emit(f"合成语音失败 (音色: {self.tts_voice_name}, 语速: {self.tts_speed}, 增益: {self.tts_gain})。请检查 API 文档/参数。")
        except requests.exceptions.RequestException as e:
            if self.is_running:
                self.signals.error.emit(f"网络错误: {e}")
        except Exception as e:
             if self.is_running:
                status_code = getattr(e, 'response', None)
                if status_code is not None: status_code = status_code.status_code
                self.signals.error.emit(f"发生错误 (状态码: {status_code}): {e}")
        finally:
            if self.is_running:
                self.signals.finished.emit()

    def text_to_speech_api(self, text_to_speak, voice_name, speed, gain):
        """调用 SiliconFlow API 进行文本转语音"""
        url = f"{SILICONFLOW_API_BASE}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        full_voice_string = f"{TTS_MODEL_BASE}:{voice_name}"

        payload = {
            "model": TTS_MODEL_BASE,
            "input": text_to_speak,
            "voice": full_voice_string,
            "response_format": "mp3",
            "speed": speed,
            "gain": gain,
        }
        print("TTS 接口 Payload:", json.dumps(payload, indent=2))

        response = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)

        if response.status_code != 200:
            error_details = response.text
            try:
                error_json = json.loads(error_details)
                error_details = json.dumps(error_json, indent=2)
            except json.JSONDecodeError:
                pass
            print(f"TTS API 错误 - 状态码: {response.status_code}\n来自 API 的详细信息:\n{error_details}")
            response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'audio' in content_type:
            audio_content = b"".join(chunk for chunk in response.iter_content(chunk_size=8192))
            return audio_content
        else:
            error_text = response.text
            print(f"TTS 错误 - 状态码 200 OK 但响应非音频 ({content_type}): {error_text}")
            raise ValueError(f"期望获得音频响应 (状态码 200)，但收到 {content_type}。 Payload: {payload}")

class TranslateAndTTSWorker(QObject):
    def __init__(self, api_key, text_to_translate, tts_voice_name, tts_speed, tts_gain):
        super().__init__()
        self.signals = WorkerSignals()
        self.api_key = api_key
        self.text_to_translate = text_to_translate
        self.tts_voice_name = tts_voice_name
        self.tts_speed = tts_speed
        self.tts_gain = tts_gain
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if not self.is_running: return
            self.signals.status_update.emit("1/2: 正在翻译文本...")
            self.signals.progress_update.emit(20)  # 开始翻译
            translated_text = self.translate_text_api(self.text_to_translate)
            self.signals.progress_update.emit(50)  # 翻译完成

            if translated_text and self.is_running:
                self.signals.translation_ready.emit(translated_text)
                self.signals.status_update.emit("2/2: 正在合成语音...")
                self.signals.progress_update.emit(60)  # 开始语音合成
                audio_data = self.text_to_speech_api(
                    translated_text,
                    self.tts_voice_name,
                    self.tts_speed,
                    self.tts_gain
                )
                self.signals.progress_update.emit(90)  # 语音合成接近完成

                if audio_data and self.is_running:
                    self.signals.audio_ready.emit(audio_data)
                    self.signals.status_update.emit("处理完成。")
                    self.signals.progress_update.emit(100)  # 完成
                elif self.is_running:
                    self.signals.error.emit(f"合成语音失败 (音色: {self.tts_voice_name}, 语速: {self.tts_speed}, 增益: {self.tts_gain})。请检查 API 文档/参数。")

            elif self.is_running:
                self.signals.error.emit("翻译文本失败。")

        except requests.exceptions.RequestException as e:
            if self.is_running:
                self.signals.error.emit(f"网络错误: {e}")
        except Exception as e:
             if self.is_running:
                status_code = getattr(e, 'response', None)
                if status_code is not None: status_code = status_code.status_code
                self.signals.error.emit(f"发生错误 (状态码: {status_code}): {e}")
        finally:
            if self.is_running:
                self.signals.finished.emit()

    def translate_text_api(self, text_to_translate):
        """调用 SiliconFlow API 进行翻译"""
        url = f"{SILICONFLOW_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        prompt = f"Translate the following text to Simplified Chinese. Output only the translation itself, without any introductory phrases.\n\nOriginal Text:\n{text_to_translate}\n\nSimplified Chinese Translation:"

        payload = {
            "model": TRANSLATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.5,
            "stream": False,
            "stop": None,
            "top_p": 0.7,
            "frequency_penalty": 0.0,
            "n": 1,
            "response_format": {"type": "text"},
        }
        print("翻译接口 Payload:", json.dumps(payload, indent=2))

        response = requests.post(url, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            error_details = response.text
            try:
                error_json = json.loads(error_details)
                error_details = json.dumps(error_json, indent=2)
            except json.JSONDecodeError:
                pass
            print(f"翻译 API 错误 - 状态码: {response.status_code}\n来自 API 的详细信息:\n{error_details}")
            response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            if content.startswith('"') and content.endswith('"'):
                 content = content[1:-1].strip()
            if content:
                 return content
            else:
                 print("翻译警告 - 收到空内容:", result)
                 raise ValueError("从 API 收到了空的翻译内容。")
        else:
            print("翻译错误 - 即便状态码 200 OK，返回格式仍无效:", result)
            raise ValueError("翻译 API 返回格式无效 (缺少 choices/message/content)。")

    def text_to_speech_api(self, text_to_speak, voice_name, speed, gain):
        """调用 SiliconFlow API 进行文本转语音"""
        url = f"{SILICONFLOW_API_BASE}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        full_voice_string = f"{TTS_MODEL_BASE}:{voice_name}"

        payload = {
            "model": TTS_MODEL_BASE,
            "input": text_to_speak,
            "voice": full_voice_string,
            "response_format": "mp3",
            "speed": speed,
            "gain": gain,
        }
        print("TTS 接口 Payload:", json.dumps(payload, indent=2))

        response = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)

        if response.status_code != 200:
            error_details = response.text
            try:
                error_json = json.loads(error_details)
                error_details = json.dumps(error_json, indent=2)
            except json.JSONDecodeError:
                pass
            print(f"TTS API 错误 - 状态码: {response.status_code}\n来自 API 的详细信息:\n{error_details}")
            response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'audio' in content_type:
            audio_content = b"".join(chunk for chunk in response.iter_content(chunk_size=8192))
            return audio_content
        else:
            error_text = response.text
            print(f"TTS 错误 - 状态码 200 OK 但响应非音频 ({content_type}): {error_text}")
            raise ValueError(f"期望获得音频响应 (状态码 200)，但收到 {content_type}。 Payload: {payload}")


# --- 主应用程序窗口 (优化UI) ---
class TranslateAndTTSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_audio_data = None
        self.worker_thread = None
        
        # 设置应用程序样式
        self.setStyleSheet(APP_STYLESHEET)
        
        # 加载图标和资源
        self.load_icons()
        
        self.init_ui()  # 初始化优化后的界面
        self.init_audio()  # 初始化音频播放器

    def load_icons(self):
        """加载应用程序使用的图标"""
        # 定义常用图标路径 (这里使用字体图标的名称作为占位符，实际使用时需要替换成实际图标文件)
        self.icons = {
            "translate": QIcon.fromTheme("translate", QIcon.fromTheme("edit")),
            "audio": QIcon.fromTheme("audio-headset", QIcon.fromTheme("media-playback-start")),
            "import": QIcon.fromTheme("document-open", QIcon.fromTheme("folder-open")),
            "export": QIcon.fromTheme("document-save", QIcon.fromTheme("document-save-as")),
            "play": QIcon.fromTheme("media-playback-start", QIcon.fromTheme("player-play")),
            "settings": QIcon.fromTheme("configure", QIcon.fromTheme("preferences-system")),
            "key": QIcon.fromTheme("dialog-password", QIcon.fromTheme("object-locked")),
        }

    def init_ui(self):
        # 设置窗口标题和图标
        self.setWindowTitle('SiliconFlow 语言与语音助手')
        self.setWindowIcon(self.icons.get("translate", QIcon()))
        self.setGeometry(200, 200, 900, 750)  # 稍微增大窗口尺寸

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 头部区域 - 包含标题和API设置
        header_layout = QHBoxLayout()
        
        # 应用标题和版本
        app_title_layout = QVBoxLayout()
        title_label = QLabel("SiliconFlow 语言与语音助手")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2a66c8;")
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet("font-size: 8pt; color: #888888;")
        app_title_layout.addWidget(title_label)
        app_title_layout.addWidget(version_label)
        header_layout.addLayout(app_title_layout)
        
        # API密钥区域
        api_layout = QHBoxLayout()
        api_layout.setSpacing(5)
        
        api_icon = QLabel()
        api_icon.setPixmap(self.icons.get("key").pixmap(QSize(24, 24)))
        api_layout.addWidget(api_icon)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("输入SiliconFlow API密钥")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setMinimumWidth(250)
        api_layout.addWidget(self.api_key_input)
        
        header_layout.addStretch(1)
        header_layout.addLayout(api_layout)
        
        main_layout.addLayout(header_layout)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        main_layout.addWidget(separator)
        
        # 创建选项卡窗口
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # 更现代的选项卡外观
        main_layout.addWidget(self.tabs)
        
        # 创建并添加各个选项卡
        self.translate_tts_tab = QWidget()
        self.tabs.addTab(self.translate_tts_tab, self.icons.get("translate"), "翻译并朗读")
        self.init_translate_tts_tab()
        
        self.translate_only_tab = QWidget()
        self.tabs.addTab(self.translate_only_tab, self.icons.get("translate"), "仅翻译")
        self.init_translate_only_tab()
        
        self.tts_only_tab = QWidget()
        self.tabs.addTab(self.tts_only_tab, self.icons.get("audio"), "仅朗读")
        self.init_tts_only_tab()
        
        # 底部状态区
        footer_layout = QVBoxLayout()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(10)
        self.progress_bar.setMaximumHeight(10)
        footer_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")  # 用于样式表中特定引用
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_layout.addWidget(self.status_label)
        
        main_layout.addLayout(footer_layout)

    def init_translate_tts_tab(self):
        layout = QVBoxLayout(self.translate_tts_tab)
        layout.setSpacing(10)
        
        # 输入区域
        input_group = QGroupBox("输入文本")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)
        
        # 工具栏 - 导入按钮
        toolbar_layout = QHBoxLayout()
        
        import_btn = QPushButton("导入文本文件")
        import_btn.setIcon(self.icons.get("import"))
        import_btn.clicked.connect(self.import_txt_translate_tts)
        import_btn.setMaximumWidth(150)
        toolbar_layout.addWidget(import_btn)
        
        toolbar_layout.addStretch()
        input_layout.addLayout(toolbar_layout)
        
        # 文本输入
        self.trans_tts_input = QTextEdit()
        self.trans_tts_input.setPlaceholderText("在此输入要翻译和朗读的英文文本...")
        input_layout.addWidget(self.trans_tts_input)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 输出区域
        output_group = QGroupBox("翻译结果")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(8)
        
        # 工具栏 - 导出按钮
        export_toolbar_layout = QHBoxLayout()
        
        export_txt_btn = QPushButton("导出为文本")
        export_txt_btn.setIcon(self.icons.get("export"))
        export_txt_btn.clicked.connect(lambda: self.export_txt(self.trans_tts_output))
        export_txt_btn.setMaximumWidth(150)
        export_toolbar_layout.addWidget(export_txt_btn)
        
        export_toolbar_layout.addStretch()
        output_layout.addLayout(export_toolbar_layout)
        
        # 译文输出
        self.trans_tts_output = QTextEdit()
        self.trans_tts_output.setReadOnly(True)
        self.trans_tts_output.setStyleSheet("background-color: #f9f9f9;")
        output_layout.addWidget(self.trans_tts_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # TTS选项
        tts_group = QGroupBox("语音合成选项")
        tts_layout = QFormLayout()
        tts_layout.setSpacing(10)
        
        # 音色选择
        self.trans_tts_voice_combo = QComboBox()
        self.trans_tts_voice_combo.addItems(["david", "alex", "default"])
        self.trans_tts_voice_combo.setMinimumWidth(120)
        tts_layout.addRow("语音音色:", self.trans_tts_voice_combo)
        
        # 语速和增益控制放在一行
        speed_gain_layout = QHBoxLayout()
        
        # 语速控制
        speed_layout = QVBoxLayout()
        speed_label = QLabel("语速:")
        self.trans_tts_speed = QDoubleSpinBox()
        self.trans_tts_speed.setRange(0.25, 4.0)
        self.trans_tts_speed.setSingleStep(0.1)
        self.trans_tts_speed.setValue(1.0)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.trans_tts_speed)
        speed_gain_layout.addLayout(speed_layout)
        
        # 增益控制
        gain_layout = QVBoxLayout()
        gain_label = QLabel("增益 (dB):")
        self.trans_tts_gain = QDoubleSpinBox()
        self.trans_tts_gain.setRange(-20.0, 20.0)
        self.trans_tts_gain.setSingleStep(0.5)
        self.trans_tts_gain.setValue(0.0)
        self.trans_tts_gain.setSuffix(" dB")
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(self.trans_tts_gain)
        speed_gain_layout.addLayout(gain_layout)
        
        tts_layout.addRow("", speed_gain_layout)
        
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)
        
        # 操作按钮区域
        buttons_layout = QHBoxLayout()
        
        self.trans_tts_process_btn = QPushButton("翻译并朗读")
        self.trans_tts_process_btn.setIcon(self.icons.get("translate"))
        self.trans_tts_process_btn.clicked.connect(self.start_translate_tts)
        
        self.trans_tts_play_btn = QPushButton("播放音频")
        self.trans_tts_play_btn.setIcon(self.icons.get("play"))
        self.trans_tts_play_btn.clicked.connect(self.play_audio)
        self.trans_tts_play_btn.setEnabled(False)
        
        self.trans_tts_export_mp3_btn = QPushButton("导出MP3")
        self.trans_tts_export_mp3_btn.setIcon(self.icons.get("export"))
        self.trans_tts_export_mp3_btn.clicked.connect(self.export_mp3)
        self.trans_tts_export_mp3_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.trans_tts_process_btn)
        buttons_layout.addWidget(self.trans_tts_play_btn)
        buttons_layout.addWidget(self.trans_tts_export_mp3_btn)
        layout.addLayout(buttons_layout)

    def init_translate_only_tab(self):
        layout = QVBoxLayout(self.translate_only_tab)
        layout.setSpacing(10)
        
        # 输入区域
        input_group = QGroupBox("输入文本")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)
        
        # 工具栏 - 导入按钮
        toolbar_layout = QHBoxLayout()
        
        import_btn = QPushButton("导入文本文件")
        import_btn.setIcon(self.icons.get("import"))
        import_btn.clicked.connect(self.import_txt_translate)
        import_btn.setMaximumWidth(150)
        toolbar_layout.addWidget(import_btn)
        
        toolbar_layout.addStretch()
        input_layout.addLayout(toolbar_layout)
        
        # 文本输入
        self.trans_input = QTextEdit()
        self.trans_input.setPlaceholderText("在此输入要翻译的英文文本...")
        input_layout.addWidget(self.trans_input)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 输出区域
        output_group = QGroupBox("翻译结果")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(8)
        
        # 工具栏 - 导出按钮
        export_toolbar_layout = QHBoxLayout()
        
        export_txt_btn = QPushButton("导出为文本")
        export_txt_btn.setIcon(self.icons.get("export"))
        export_txt_btn.clicked.connect(lambda: self.export_txt(self.trans_output))
        export_txt_btn.setMaximumWidth(150)
        export_toolbar_layout.addWidget(export_txt_btn)
        
        export_toolbar_layout.addStretch()
        output_layout.addLayout(export_toolbar_layout)
        
        # 译文输出
        self.trans_output = QTextEdit()
        self.trans_output.setReadOnly(True)
        self.trans_output.setStyleSheet("background-color: #f9f9f9;")
        output_layout.addWidget(self.trans_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        self.trans_process_btn = QPushButton("翻译文本")
        self.trans_process_btn.setIcon(self.icons.get("translate"))
        self.trans_process_btn.clicked.connect(self.start_translate_only)
        
        buttons_layout.addWidget(self.trans_process_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

    def init_tts_only_tab(self):
        layout = QVBoxLayout(self.tts_only_tab)
        layout.setSpacing(10)
        
        # 输入区域
        input_group = QGroupBox("输入文本")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)
        
        # 工具栏 - 导入按钮
        toolbar_layout = QHBoxLayout()
        
        import_btn = QPushButton("导入文本文件")
        import_btn.setIcon(self.icons.get("import"))
        import_btn.clicked.connect(self.import_txt_tts)
        import_btn.setMaximumWidth(150)
        toolbar_layout.addWidget(import_btn)
        
        toolbar_layout.addStretch()
        input_layout.addLayout(toolbar_layout)
        
        # 文本输入
        self.tts_input = QTextEdit()
        self.tts_input.setPlaceholderText("在此输入要朗读的中文文本...")
        input_layout.addWidget(self.tts_input)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # TTS选项
        tts_group = QGroupBox("语音合成选项")
        tts_layout = QFormLayout()
        tts_layout.setSpacing(10)
        
        # 音色选择
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.addItems(["david", "alex", "default"])
        self.tts_voice_combo.setMinimumWidth(120)
        tts_layout.addRow("语音音色:", self.tts_voice_combo)
        
        # 语速和增益控制放在一行
        speed_gain_layout = QHBoxLayout()
        
        # 语速控制
        speed_layout = QVBoxLayout()
        speed_label = QLabel("语速:")
        self.tts_speed = QDoubleSpinBox()
        self.tts_speed.setRange(0.25, 4.0)
        self.tts_speed.setSingleStep(0.1)
        self.tts_speed.setValue(1.0)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.tts_speed)
        speed_gain_layout.addLayout(speed_layout)
        
        # 增益控制
        gain_layout = QVBoxLayout()
        gain_label = QLabel("增益 (dB):")
        self.tts_gain = QDoubleSpinBox()
        self.tts_gain.setRange(-20.0, 20.0)
        self.tts_gain.setSingleStep(0.5)
        self.tts_gain.setValue(0.0)
        self.tts_gain.setSuffix(" dB")
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(self.tts_gain)
        speed_gain_layout.addLayout(gain_layout)
        
        tts_layout.addRow("", speed_gain_layout)
        
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        self.tts_process_btn = QPushButton("合成语音")
        self.tts_process_btn.setIcon(self.icons.get("audio"))
        self.tts_process_btn.clicked.connect(self.start_tts_only)
        
        self.tts_play_btn = QPushButton("播放音频")
        self.tts_play_btn.setIcon(self.icons.get("play"))
        self.tts_play_btn.clicked.connect(self.play_audio)
        self.tts_play_btn.setEnabled(False)
        
        self.tts_export_mp3_btn = QPushButton("导出MP3")
        self.tts_export_mp3_btn.setIcon(self.icons.get("export"))
        self.tts_export_mp3_btn.clicked.connect(self.export_mp3)
        self.tts_export_mp3_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.tts_process_btn)
        buttons_layout.addWidget(self.tts_play_btn)
        buttons_layout.addWidget(self.tts_export_mp3_btn)
        layout.addLayout(buttons_layout)

    def init_audio(self):
        """初始化音频系统"""
        if pygame_available:
            try:
                pygame.mixer.init()
                print("Pygame 音频系统已初始化。")
            except Exception as e:
                self.show_error(f"初始化音频播放器失败 (pygame): {e}")
        else:
            self.show_error("音频播放需要安装 pygame 库。请使用 'pip install pygame' 安装。")

    # --- 文件导入导出功能 ---
    def import_txt_translate_tts(self):
        """为翻译并朗读选项卡导入TXT文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择TXT文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.trans_tts_input.setText(text)
                    self.status_label.setText(f"已从 {os.path.basename(file_path)} 导入文本")
            except Exception as e:
                self.show_error(f"导入文件时出错: {e}")
    
    def import_txt_translate(self):
        """为仅翻译选项卡导入TXT文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择TXT文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.trans_input.setText(text)
                    self.status_label.setText(f"已从 {os.path.basename(file_path)} 导入文本")
            except Exception as e:
                self.show_error(f"导入文件时出错: {e}")
                
    def import_txt_tts(self):
        """为仅朗读选项卡导入TXT文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择TXT文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.tts_input.setText(text)
                    self.status_label.setText(f"已从 {os.path.basename(file_path)} 导入文本")
            except Exception as e:
                self.show_error(f"导入文件时出错: {e}")
    
    def export_txt(self, text_edit):
        """导出文本到TXT文件"""
        text = text_edit.toPlainText()
        if not text:
            self.show_error("没有可导出的文本内容")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存TXT文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(text)
                self.status_label.setText(f"文本已保存到 {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"导出文件时出错: {e}")
    
    def export_mp3(self):
        """导出当前音频到MP3文件"""
        if not self.current_audio_data:
            self.show_error("没有可导出的音频数据")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存MP3文件", "", "MP3音频 (*.mp3);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'wb') as file:
                    file.write(self.current_audio_data)
                self.status_label.setText(f"音频已保存到 {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"导出音频时出错: {e}")
    
    # --- 功能处理 ---
    def start_translate_tts(self):
        """开始翻译并合成语音"""
        api_key = self.api_key_input.text().strip()
        input_text = self.trans_tts_input.toPlainText().strip()
        selected_voice_name = self.trans_tts_voice_combo.currentText()
        selected_speed = self.trans_tts_speed.value()
        selected_gain = self.tts_gain.value()
        
        # 输入校验
        if not api_key:
            self.show_error("请输入您的 SiliconFlow API 密钥")
            return
        if not input_text:
            self.show_error("请输入要处理的文本")
            return
            
        # 准备UI
        self.trans_tts_output.clear()
        self.current_audio_data = None
        self.trans_tts_process_btn.setEnabled(False)
        self.trans_tts_play_btn.setEnabled(False)
        self.trans_tts_export_mp3_btn.setEnabled(False)
        self.status_label.setText("开始处理...")
        self.progress_bar.setValue(10)  # 设置初始进度
        
        # 创建并启动工作线程
        self.worker_thread = threading.Thread(
            target=self.run_translate_tts_worker,
            args=(api_key, input_text, selected_voice_name, selected_speed, selected_gain),
            daemon=True
        )
        self.worker_thread.start()
    
    def start_translate_only(self):
        """只进行翻译"""
        api_key = self.api_key_input.text().strip()
        input_text = self.trans_input.toPlainText().strip()
        
        # 输入校验
        if not api_key:
            self.show_error("请输入您的 SiliconFlow API 密钥")
            return
        if not input_text:
            self.show_error("请输入要翻译的文本")
            return
            
        # 准备UI
        self.trans_output.clear()
        self.trans_process_btn.setEnabled(False)
        self.status_label.setText("正在翻译...")
        self.progress_bar.setValue(10)  # 设置初始进度
        
        # 创建并启动工作线程
        self.worker_thread = threading.Thread(
            target=self.run_translate_worker,
            args=(api_key, input_text),
            daemon=True
        )
        self.worker_thread.start()
    
    def start_tts_only(self):
        """只进行语音合成"""
        api_key = self.api_key_input.text().strip()
        input_text = self.tts_input.toPlainText().strip()
        selected_voice_name = self.tts_voice_combo.currentText()
        selected_speed = self.tts_speed.value()
        selected_gain = self.tts_gain.value()
        
        # 输入校验
        if not api_key:
            self.show_error("请输入您的 SiliconFlow API 密钥")
            return
        if not input_text:
            self.show_error("请输入要朗读的文本")
            return
            
        # 准备UI
        self.current_audio_data = None
        self.tts_process_btn.setEnabled(False)
        self.tts_play_btn.setEnabled(False)
        self.tts_export_mp3_btn.setEnabled(False)
        self.status_label.setText("正在合成语音...")
        self.progress_bar.setValue(10)  # 设置初始进度
        
        # 创建并启动工作线程
        self.worker_thread = threading.Thread(
            target=self.run_tts_worker,
            args=(api_key, input_text, selected_voice_name, selected_speed, selected_gain),
            daemon=True
        )
        self.worker_thread.start()
    
    # --- 后台任务 ---
    def run_translate_tts_worker(self, api_key, input_text, voice_name, speed, gain):
        """运行翻译+TTS工作线程"""
        worker = TranslateAndTTSWorker(api_key, input_text, voice_name, speed, gain)
        worker.signals.status_update.connect(self.update_status)
        worker.signals.translation_ready.connect(lambda text: self.trans_tts_output.setText(text))
        worker.signals.audio_ready.connect(self.handle_audio_data_translate_tts)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: self.on_translate_tts_worker_finished())
        worker.signals.progress_update.connect(self.update_progress)
        worker.run()
    
    def run_translate_worker(self, api_key, input_text):
        """运行仅翻译工作线程"""
        worker = TranslationWorker(api_key, input_text)
        worker.signals.status_update.connect(self.update_status)
        worker.signals.translation_ready.connect(lambda text: self.trans_output.setText(text))
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: self.on_translate_worker_finished())
        worker.signals.progress_update.connect(self.update_progress)
        worker.run()
    
    def run_tts_worker(self, api_key, input_text, voice_name, speed, gain):
        """运行仅TTS工作线程"""
        worker = TTSWorker(api_key, input_text, voice_name, speed, gain)
        worker.signals.status_update.connect(self.update_status)
        worker.signals.audio_ready.connect(self.handle_audio_data_tts)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: self.on_tts_worker_finished())
        worker.signals.progress_update.connect(self.update_progress)
        worker.run()
    
    # --- UI更新处理函数 ---
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.setText(message)
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def handle_audio_data_translate_tts(self, audio_data):
        """处理翻译+TTS选项卡接收到的音频数据"""
        self.current_audio_data = audio_data
        self.play_audio()  # 立即播放
        self.trans_tts_play_btn.setEnabled(True)
        self.trans_tts_export_mp3_btn.setEnabled(True)
    
    def handle_audio_data_tts(self, audio_data):
        """处理TTS选项卡接收到的音频数据"""
        self.current_audio_data = audio_data
        self.play_audio()  # 立即播放
        self.tts_play_btn.setEnabled(True)
        self.tts_export_mp3_btn.setEnabled(True)
    
    def handle_error(self, error_message):
        """处理错误消息"""
        self.show_error(error_message)
        self.status_label.setText(f"错误: {error_message[:50]}...")
        self.progress_bar.setValue(0)  # 重置进度条
        
        # 重置相关按钮状态
        current_tab = self.tabs.currentWidget()
        if current_tab == self.translate_tts_tab:
            self.trans_tts_process_btn.setEnabled(True)
        elif current_tab == self.translate_only_tab:
            self.trans_process_btn.setEnabled(True)
        elif current_tab == self.tts_only_tab:
            self.tts_process_btn.setEnabled(True)
    
    def on_translate_tts_worker_finished(self):
        """当翻译+TTS工作线程完成时"""
        self.trans_tts_process_btn.setEnabled(True)
        self.worker_thread = None
    
    def on_translate_worker_finished(self):
        """当翻译工作线程完成时"""
        self.trans_process_btn.setEnabled(True)
        self.worker_thread = None
    
    def on_tts_worker_finished(self):
        """当TTS工作线程完成时"""
        self.tts_process_btn.setEnabled(True)
        self.worker_thread = None
    
    # --- 工具函数 ---
    def play_audio(self):
        """播放当前音频"""
        if not pygame_available:
            self.show_error("音频播放需要安装 pygame 库")
            return
            
        if self.current_audio_data:
            try:
                self.status_label.setText("正在播放音频...")
                QApplication.processEvents()  # 处理界面事件，避免卡顿
                audio_stream = io.BytesIO(self.current_audio_data)
                pygame.mixer.music.load(audio_stream)
                pygame.mixer.music.play()
                self.status_label.setText("开始播放音频")
            except Exception as e:
                self.show_error(f"播放音频时出错: {e}")
                self.status_label.setText("播放音频出错")
        else:
            self.show_error("无可用音频数据播放")
    
    def show_error(self, message):
        """显示错误消息对话框"""
        error_box = QMessageBox(self)
        error_box.setWindowTitle("错误")
        error_box.setIcon(QMessageBox.Warning)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec()
    
    def closeEvent(self, event):
        """关闭窗口时的处理"""
        if pygame_available:
            pygame.mixer.quit()
            print("Pygame 音频系统已退出")
        super().closeEvent(event)


# --- 主程序入口 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))  # 使用Fusion样式，更现代的外观
    
    # 创建并显示主窗口
    main_window = TranslateAndTTSApp()
    main_window.show()
    
    sys.exit(app.exec())