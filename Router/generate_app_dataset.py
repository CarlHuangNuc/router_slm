"""
基于APP（安卓原生 + 品牌原生 + 装机必备）生成"打开APP"类用户指令数据集
每个APP生成多种自然表达方式，标签均为 0 (本地操作)
每条中文指令后紧跟对应英文翻译，方便中英对比

用法:
    python generate_app_dataset.py                          # 含变体，输出到默认路径
    python generate_app_dataset.py --no-variants            # 不生成ASR变体
    python generate_app_dataset.py -o ./my_output.json      # 指定输出路径

输出: ./dataset/val_data_mobile_instructions.json (默认)
"""

import json
import os
import argparse
import random

from pypinyin.constants import PINYIN_DICT


# 拼音工具：去声调
_TONE_TABLE = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüńňǹ",
    "aaaaeeeeiiiioooouuuuvvvvvnnn",
)


def _strip_tone(py):
    return py.translate(_TONE_TABLE)


def _get_pinyins(ch):
    """返回字符的所有去声调读音列表（含多音字）。"""
    code = ord(ch)
    if code not in PINYIN_DICT:
        return []
    seen = set()
    out = []
    for py in PINYIN_DICT[code].split(","):
        p = _strip_tone(py.strip())
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# 拼音 -> 候选字 的反向映射（在 build_dataset 时按指令字符池构建）
PINYIN_REVERSE_MAP = {}

# ============ Prompt 6 白名单定义 ============
# 只有以下 App 的"打开"指令 label=0 (local)，其余 App 均为 label=1 (cloud)
WHITELIST_APPS = {
    "电话", "信息", "联系人", "相机", "相册","图库", "设置", "时钟", "日历","计算器", 
    "浏览器", "联想浏览器","应用商店", "乐商店", "便笺", "Moto笔记", "录音机", "指南针", "截图", "截图编辑器", 
    "天气", "联想天气",
    "文件管理", "下载",
    
    "实时字幕",
    "Moto", "玩机技巧", "Gametime", "息屏显示", "乐语音", "一键换机", "手机管家", "Ready For", "家庭空间", "杜比音效", 
    "设备帮助", "健康检测", 

    "超级互联", "跨设备协同", "联想负一屏", "联想投屏", 
    "联想服务", "联想账号", "乐云",
    
    "高德地图", "讯飞输入法", "收音机", "联想音乐", "系统视频",
    "微信", "QQ", "抖音", "微博", "淘宝", "支付宝",

    "闹钟", "手电筒", "短信", "备忘录", "下载管理器", "蓝牙", "邮件",
    "屏幕录制", "游戏中心", "视频", "阅读", "投屏", "钱包", "音乐",
    "moto文件", "Moto Gametime", "Moto 显示", "Moto 语音",
    "Moto 迁移", "Moto 安全", "Moto 家庭空间", "Moto 音效", "Moto 导览",
    "微信", "QQ", "抖音", "微博", "淘宝", "支付宝", "哔哩哔哩",
    "京东", "小红书", "美团", "拼多多", "快手", "腾讯视频", "爱奇艺", "网易云音乐",
    "QQ音乐", "百度", "菜鸟裹裹", "12306", "滴滴出行", "饿了么", "优酷视频",
    "番茄免费小说", "今日头条", "迅雷", "WPS", "知乎", "Keep", "番茄ToDo",
    "高德地图", "讯飞输入法", "收音机", "联想音乐", "系统视频"

    # --- 以下 App 在数据集中存在但未加入白名单 (label=1 cloud) ---
    # 【安卓原生-未入选】闹钟, 手电筒, 短信, 备忘录, 下载管理器, 蓝牙, 邮件,
    #   屏幕录制, 游戏中心, 视频, 阅读, 投屏, 钱包, 音乐
    # 【品牌/Moto-未入选】moto文件, Moto Gametime, Moto 显示, Moto 语音,
    #   Moto 迁移, Moto 安全, Moto 家庭空间, Moto 音效, Moto 导览
    # 【第三方热门-未入选】微信, QQ, 抖音, 微博, 淘宝, 支付宝, 哔哩哔哩,
    #   京东, 小红书, 美团, 拼多多, 快手, 腾讯视频, 爱奇艺, 网易云音乐,
    #   QQ音乐, 百度, 菜鸟裹裹, 12306, 滴滴出行, 饿了么, 优酷视频,
    #   番茄免费小说, 今日头条, 迅雷, WPS, 知乎, Keep, 番茄ToDo,
    #   高德地图, 讯飞输入法, 收音机, 联想音乐, 系统视频
}

# local_control 中每个子类的 label 归属
LOCAL_CONTROL_LABELS = {
    "手机拍照": 0, "耳机拍照": 0,
    "上一曲": 0, "下一曲": 0,
    "音量增大": 0, "音量减小": 0,
    "暂停播放": 0, "继续播放": 0,
    "静音": 1, "通话": 1, "进度": 1, "设备开关": 1, "亮度": 1,
    "夜间模式": 1, "锁屏": 1, "定时提醒": 1, "通讯发起": 1,
    "信息查询": 1, "拍摄录制": 1, "系统操作": 1,
}


APP_NAME_EN = {
    "相机": "Camera", "闹钟": "Alarm", "设置": "Settings", "手电筒": "Flashlight",
    "电话": "Phone", "短信": "Messages", "联系人": "Contacts", "浏览器": "Browser",
    "日历": "Calendar", "天气": "Weather", "时钟": "Clock", "计算器": "Calculator",
    "备忘录": "Memo", "文件管理": "FileManager", "图库": "Gallery", "音乐": "Music",
    "收音机": "Radio", "应用商店": "AppStore", "钱包": "Wallet", "录音机": "Recorder",
    "下载管理器": "DownloadManager", "蓝牙": "Bluetooth", "邮件": "Email",
    "屏幕录制": "ScreenRecorder", "手机管家": "PhoneManager", "游戏中心": "GameCenter",
    "视频": "Video", "阅读": "Reading", "指南针": "Compass", "投屏": "ScreenCast",
    "信息": "Messages", "moto文件": "MotoFiles", "下载": "Downloads",
    "联想浏览器": "LenovoBrowser", "乐商店": "LeStore", "便笺": "StickyNotes",
    "Moto笔记": "MotoNote", "截图": "Screenshot", "截图编辑器": "ScreenshotEditor",
    "联想天气": "LenovoWeather",
    "Moto": "Moto", "Moto Gametime": "MotoGametime", "Moto 显示": "MotoDisplay",
    "Moto 语音": "MotoVoice", "Moto 迁移": "MotoTransfer", "Moto 安全": "MotoSecurity",
    "Ready For": "ReadyFor", "Moto 家庭空间": "MotoFamilySpace",
    "Moto 音效": "MotoSound", "Moto 导览": "MotoGuide", "健康检测": "HealthMonitor",
    "实时字幕": "LiveCaption",
    "超级互联": "SuperConnect", "联想负一屏": "LenovoHomeScreen",
    "联想投屏": "LenovoCast", "联想服务": "LenovoService",
    "联想账号": "LenovoAccount", "乐云": "LeCloud",
    "讯飞输入法": "iFlyInput",
    "微信": "WeChat", "QQ": "QQ", "支付宝": "Alipay", "哔哩哔哩": "Bilibili",
    "抖音": "Douyin", "淘宝": "Taobao", "京东": "JD", "高德地图": "Amap",
    "小红书": "Xiaohongshu", "美团": "Meituan", "拼多多": "Pinduoduo",
    "快手": "Kuaishou", "腾讯视频": "TencentVideo", "爱奇艺": "iQIYI",
    "网易云音乐": "NetEaseMusic", "QQ音乐": "QQMusic", "百度": "Baidu",
    "微博": "Weibo", "菜鸟裹裹": "Cainiao", "12306": "12306",
    "滴滴出行": "DiDi", "饿了么": "Eleme", "优酷视频": "Youku",
    "番茄免费小说": "TomatoNovel", "今日头条": "Toutiao", "迅雷": "Xunlei",
    "WPS": "WPS", "知乎": "Zhihu", "Keep": "Keep", "番茄ToDo": "TomatoToDo",
    "手机拍照": "PhonePhoto", "耳机拍照": "EarphonePhoto",
    "上一曲": "PreviousTrack", "下一曲": "NextTrack",
    "音量增大": "VolumeUp", "音量减小": "VolumeDown",
    "暂停播放": "PausePlayback", "继续播放": "ResumePlayback",
    "静音": "Mute", "通话": "Call", "进度": "Progress", "设备开关": "DeviceToggle",
    "亮度": "Brightness", "夜间模式": "NightMode", "锁屏": "LockScreen",
    "定时提醒": "TimerReminder", "通讯发起": "InitiateCall",
    "信息查询": "InfoQuery", "拍摄录制": "VideoRecord", "系统操作": "SystemOp",
    "通用多步": "GeneralMultiStep",
}


def _get_app_label(app_name, app_type, ctrl_name=None):
    """按 Prompt  白名单规则确定 label。

    Args:
        app_name: App 名称
        app_type: 应用类型 (native/brand/moto/lenovo/preinstalled/popular/local_control/cloud_task/multi_step)
        ctrl_name: local_control 子类名（如 "拍照"、"通话"）

    Returns:
        0 (local) 或 1 (cloud)
    """
    if app_type == "cloud_task":
        return 1
    if app_type == "multi_step":
        return 1
    if app_type == "local_control":
        return LOCAL_CONTROL_LABELS.get(ctrl_name, 1)
    # App 类：检查是否在白名单
    return 0 if app_name in WHITELIST_APPS else 1

# 安卓原生自带应用（30款）
# 格式: "APP名": [("中文指令", "英文翻译"), ...]
NATIVE_APPS = {
    "相机": [
        ("打开相机", "Open the camera"),
        ("帮我打开相机", "Help me open the camera"),
        ("启动相机", "Launch the camera"),
        ("我要拍照打开相机", "I want to take a photo, open the camera"),
        ("开一下相机", "Turn on the camera"),
        ("请打开相机", "Please open the camera"),
        ("给我打开相机", "Open the camera for me"),
        ("我要打开相机", "I want to open the camera"),
        ("打开摄像头", "Turn on the camera"),
        ("打开相机应用", "Open the camera app"),
        ("现在打开相机", "Open the camera now"),
        ("马上打开相机", "Open the camera right away"),
        ("帮我现在打开相机", "Help me open the camera now"),
        ("请帮我打开相机", "Please help me open the camera"),
        ("给我现在打开相机", "Open the camera for me now"),
        ("我要马上打开相机", "I want to open the camera right away"),
        ("打开一下相机", "Open the camera for a moment"),
        ("请打开一下相机", "Please open the camera for a moment"),
        ("给我打开一下相机", "Open the camera for me for a moment"),
        ("我要打开一下相机", "I want to open the camera for a moment"),
        ("立刻打开相机", "Open the camera immediately"),
        ("进入相机", "Enter the camera"),
    ],
    "闹钟": [
        ("打开闹钟", "Open the alarm"),
        ("帮我打开闹钟", "Help me open the alarm"),
        ("进入闹钟设置", "Enter alarm settings"),
        ("打开闹钟应用", "Open the alarm app"),
    ],
    "设置": [
        ("打开设置", "Open settings"),
        ("进入设置", "Enter settings"),
        ("帮我打开手机设置", "Help me open phone settings"),
        ("打开系统设置", "Open system settings"),
        ("帮我打开设置", "Help me open settings"),
        ("请打开设置", "Please open settings"),
        ("给我打开设置", "Open settings for me"),
        ("我要打开设置", "I want to open settings"),
        ("打开手机设置", "Open phone settings"),
        ("打开设置应用", "Open the settings app"),
        ("现在打开设置", "Open settings now"),
        ("马上打开设置", "Open settings right away"),
        ("帮我现在打开设置", "Help me open settings now"),
        ("请帮我打开设置", "Please help me open settings"),
        ("给我现在打开设置", "Open settings for me now"),
        ("我要马上打开设置", "I want to open settings right away"),
        ("打开一下设置", "Open settings for a moment"),
        ("请打开一下设置", "Please open settings for a moment"),
        ("给我打开一下设置", "Open settings for me for a moment"),
        ("立刻打开设置", "Open settings immediately"),
    ],
    "手电筒": [
        ("打开手电筒", "Turn on the flashlight"),
        ("帮我开手电筒", "Help me turn on the flashlight"),
        ("把手电筒打开", "Turn the flashlight on"),
        ("开一下手电筒", "Switch on the flashlight"),
    ],
    "电话": [
        ("打开电话", "Open the phone"),
        ("打开拨号界面", "Open the dialer"),
        ("帮我打开电话应用", "Help me open the phone app"),
        ("进入拨号盘", "Enter the dial pad"),
        ("帮我打开电话", "Help me open the phone"),
        ("请打开电话", "Please open the phone"),
        ("给我打开电话", "Open the phone for me"),
        ("我要打开电话", "I want to open the phone"),
        ("打开电话应用", "Open the phone app"),
        ("打开电话软件", "Open the phone software"),
        ("现在打开电话", "Open the phone now"),
        ("马上打开电话", "Open the phone right away"),
        ("帮我现在打开电话", "Help me open the phone now"),
        ("请帮我打开电话", "Please help me open the phone"),
        ("给我现在打开电话", "Open the phone for me now"),
        ("我要马上打开电话", "I want to open the phone right away"),
        ("打开一下电话", "Open the phone for a moment"),
        ("请打开一下电话", "Please open the phone for a moment"),
        ("给我打开一下电话", "Open the phone for me for a moment"),
        ("我要打开一下电话", "I want to open the phone for a moment"),
        ("立刻打开电话", "Open the phone immediately"),
        ("进入电话", "Enter the phone"),
    ],
    "短信": [
        ("打开短信", "Open messages"),
        ("帮我打开短信", "Help me open messages"),
        ("进入短信应用", "Enter the messaging app"),
        ("打开消息", "Open messages"),
    ],
    "联系人": [
        ("打开联系人", "Open contacts"),
        ("帮我打开通讯录", "Help me open the contacts"),
        ("进入联系人列表", "Enter the contacts list"),
        ("打开通讯录", "Open the address book"),
        ("帮我打开联系人", "Help me open contacts"),
        ("请打开联系人", "Please open contacts"),
        ("给我打开联系人", "Open contacts for me"),
        ("我要打开联系人", "I want to open contacts"),
        ("打开联系人应用", "Open the contacts app"),
        ("现在打开联系人", "Open contacts now"),
        ("马上打开联系人", "Open contacts right away"),
        ("帮我现在打开联系人", "Help me open contacts now"),
        ("请帮我打开联系人", "Please help me open contacts"),
        ("给我现在打开联系人", "Open contacts for me now"),
        ("我要马上打开联系人", "I want to open contacts right away"),
        ("打开一下联系人", "Open contacts for a moment"),
        ("请打开一下联系人", "Please open contacts for a moment"),
        ("给我打开一下联系人", "Open contacts for me for a moment"),
        ("我要打开一下联系人", "I want to open contacts for a moment"),
        ("立刻打开联系人", "Open contacts immediately"),
        ("进入联系人", "Enter contacts"),
    ],
    "浏览器": [
        ("打开浏览器", "Open the browser"),
        ("帮我打开浏览器", "Help me open the browser"),
        ("启动自带浏览器", "Launch the built-in browser"),
        ("打开系统浏览器", "Open the system browser"),
        ("请打开浏览器", "Please open the browser"),
        ("给我打开浏览器", "Open the browser for me"),
        ("我要打开浏览器", "I want to open the browser"),
        ("打开浏览器应用", "Open the browser app"),
        ("现在打开浏览器", "Open the browser now"),
        ("马上打开浏览器", "Open the browser right away"),
        ("帮我现在打开浏览器", "Help me open the browser now"),
        ("请帮我打开浏览器", "Please help me open the browser"),
        ("给我现在打开浏览器", "Open the browser for me now"),
        ("我要马上打开浏览器", "I want to open the browser right away"),
        ("打开一下浏览器", "Open the browser for a moment"),
        ("请打开一下浏览器", "Please open the browser for a moment"),
        ("给我打开一下浏览器", "Open the browser for me for a moment"),
        ("我要打开一下浏览器", "I want to open the browser for a moment"),
        ("立刻打开浏览器", "Open the browser immediately"),
        ("进入浏览器", "Enter the browser"),
    ],
    "日历": [
        ("打开日历", "Open the calendar"),
        ("帮我打开日历", "Help me open the calendar"),
        ("进入日历应用", "Enter the calendar app"),
        ("看一下日历", "Check the calendar"),
        ("请打开日历", "Please open the calendar"),
        ("给我打开日历", "Open the calendar for me"),
        ("我要打开日历", "I want to open the calendar"),
        ("打开日历应用", "Open the calendar app"),
        ("打开日程", "Open the schedule"),
        ("现在打开日历", "Open the calendar now"),
        ("马上打开日历", "Open the calendar right away"),
        ("帮我现在打开日历", "Help me open the calendar now"),
        ("请帮我打开日历", "Please help me open the calendar"),
        ("给我现在打开日历", "Open the calendar for me now"),
        ("我要马上打开日历", "I want to open the calendar right away"),
        ("打开一下日历", "Open the calendar for a moment"),
        ("请打开一下日历", "Please open the calendar for a moment"),
        ("给我打开一下日历", "Open the calendar for me for a moment"),
        ("我要打开一下日历", "I want to open the calendar for a moment"),
        ("立刻打开日历", "Open the calendar immediately"),
        ("进入日历", "Enter the calendar"),
    ],
    "天气": [
        ("打开天气", "Open weather"),
        ("帮我打开天气应用", "Help me open the weather app"),
        ("看一下天气APP", "Check the weather app"),
        ("打开天气预报", "Open the weather forecast"),
        ("帮我打开天气", "Help me open weather"),
        ("请打开天气", "Please open weather"),
        ("给我打开天气", "Open weather for me"),
        ("我要打开天气", "I want to open weather"),
        ("现在打开天气", "Open weather now"),
        ("马上打开天气", "Open weather right away"),
        ("帮我现在打开天气", "Help me open weather now"),
        ("请帮我打开天气", "Please help me open weather"),
        ("给我现在打开天气", "Open weather for me now"),
        ("立刻打开天气", "Open weather immediately"),
        ("进入天气", "Enter weather"),
    ],
    "时钟": [
        ("打开时钟", "Open the clock"),
        ("帮我打开时钟", "Help me open the clock"),
        ("进入时钟应用", "Enter the clock app"),
        ("打开世界时钟", "Open the world clock"),
        ("请打开时钟", "Please open the clock"),
        ("给我打开时钟", "Open the clock for me"),
        ("我要打开时钟", "I want to open the clock"),
        ("打开时钟应用", "Open the clock app"),
        ("现在打开时钟", "Open the clock now"),
        ("马上打开时钟", "Open the clock right away"),
        ("帮我现在打开时钟", "Help me open the clock now"),
        ("请帮我打开时钟", "Please help me open the clock"),
        ("给我现在打开时钟", "Open the clock for me now"),
        ("我要马上打开时钟", "I want to open the clock right away"),
        ("打开一下时钟", "Open the clock for a moment"),
        ("请打开一下时钟", "Please open the clock for a moment"),
        ("给我打开一下时钟", "Open the clock for me for a moment"),
        ("我要打开一下时钟", "I want to open the clock for a moment"),
        ("立刻打开时钟", "Open the clock immediately"),
        ("进入时钟", "Enter the clock"),
    ],
    "计算器": [
        ("打开计算器", "Open the calculator"),
        ("帮我打开计算器", "Help me open the calculator"),
        ("启动计算器", "Launch the calculator"),
        ("我要用计算器", "I want to use the calculator"),
        ("请打开计算器", "Please open the calculator"),
        ("给我打开计算器", "Open the calculator for me"),
        ("我要打开计算器", "I want to open the calculator"),
        ("打开计算工具", "Open the calculation tool"),
        ("打开计算器应用", "Open the calculator app"),
        ("现在打开计算器", "Open the calculator now"),
        ("马上打开计算器", "Open the calculator right away"),
        ("帮我现在打开计算器", "Help me open the calculator now"),
        ("请帮我打开计算器", "Please help me open the calculator"),
        ("给我现在打开计算器", "Open the calculator for me now"),
        ("我要马上打开计算器", "I want to open the calculator right away"),
        ("打开一下计算器", "Open the calculator for a moment"),
        ("请打开一下计算器", "Please open the calculator for a moment"),
        ("给我打开一下计算器", "Open the calculator for me for a moment"),
        ("我要打开一下计算器", "I want to open the calculator for a moment"),
        ("立刻打开计算器", "Open the calculator immediately"),
        ("进入计算器", "Enter the calculator"),
    ],
    "备忘录": [
        ("打开备忘录", "Open the memo"),
        ("帮我打开备忘录", "Help me open the memo"),
        ("进入备忘录", "Enter the memo"),
        ("打开笔记本", "Open the notebook"),
    ],
    "文件管理": [
        ("打开文件管理", "Open file manager"),
        ("帮我打开文件管理器", "Help me open the file manager"),
        ("进入文件管理", "Enter file management"),
        ("打开我的文件", "Open my files"),
        ("帮我打开文件管理", "Help me open file management"),
        ("请打开文件管理", "Please open file management"),
        ("给我打开文件管理", "Open file management for me"),
        ("我要打开文件管理", "I want to open file management"),
        ("打开Moto文件", "Open Moto Files"),
        ("打开文件浏览器", "Open the file browser"),
        ("现在打开文件管理", "Open file management now"),
        ("马上打开文件管理", "Open file management right away"),
        ("帮我现在打开文件管理", "Help me open file management now"),
        ("请帮我打开文件管理", "Please help me open file management"),
        ("给我现在打开文件管理", "Open file management for me now"),
        ("我要马上打开文件管理", "I want to open file management right away"),
        ("打开一下文件管理", "Open file management for a moment"),
        ("请打开一下文件管理", "Please open file management for a moment"),
        ("给我打开一下文件管理", "Open file management for me for a moment"),
        ("我要打开一下文件管理", "I want to open file management for a moment"),
        ("立刻打开文件管理", "Open file management immediately"),
    ],
    "图库": [
        ("打开图库", "Open the gallery"),
        ("帮我打开相册", "Help me open the photo album"),
        ("进入图库", "Enter the gallery"),
        ("打开照片", "Open photos"),
        ("打开相册", "Open the photo album"),
        ("请打开相册", "Please open the photo album"),
        ("给我打开相册", "Open the photo album for me"),
        ("我要打开相册", "I want to open the photo album"),
        ("打开相册应用", "Open the photo album app"),
        ("现在打开相册", "Open the photo album now"),
        ("马上打开相册", "Open the photo album right away"),
        ("帮我现在打开相册", "Help me open the photo album now"),
        ("请帮我打开相册", "Please help me open the photo album"),
        ("给我现在打开相册", "Open the photo album for me now"),
        ("我要马上打开相册", "I want to open the photo album right away"),
        ("打开一下相册", "Open the photo album for a moment"),
        ("请打开一下相册", "Please open the photo album for a moment"),
        ("给我打开一下相册", "Open the photo album for me for a moment"),
        ("立刻打开相册", "Open the photo album immediately"),
        ("进入相册", "Enter the photo album"),
    ],
    "音乐": [
        ("打开音乐", "Open music"),
        ("帮我打开音乐播放器", "Help me open the music player"),
        ("启动自带音乐", "Launch the built-in music"),
        ("打开音乐应用", "Open the music app"),
    ],
    "收音机": [
        ("打开收音机", "Open the radio"),
        ("帮我打开收音机", "Help me open the radio"),
        ("启动FM收音机", "Launch FM radio"),
        ("打开调频收音机", "Open the FM radio"),
    ],
    "应用商店": [
        ("打开应用商店", "Open the app store"),
        ("帮我打开应用市场", "Help me open the app market"),
        ("进入应用商店", "Enter the app store"),
        ("打开软件商店", "Open the software store"),
        ("帮我打开应用商店", "Help me open the app store"),
        ("请打开应用商店", "Please open the app store"),
        ("给我打开应用商店", "Open the app store for me"),
        ("我要打开应用商店", "I want to open the app store"),
        ("打开应用市场", "Open the app market"),
        ("现在打开应用商店", "Open the app store now"),
        ("马上打开应用商店", "Open the app store right away"),
        ("帮我现在打开应用商店", "Help me open the app store now"),
        ("请帮我打开应用商店", "Please help me open the app store"),
        ("给我现在打开应用商店", "Open the app store for me now"),
        ("我要马上打开应用商店", "I want to open the app store right away"),
        ("打开一下应用商店", "Open the app store for a moment"),
        ("请打开一下应用商店", "Please open the app store for a moment"),
        ("给我打开一下应用商店", "Open the app store for me for a moment"),
        ("我要打开一下应用商店", "I want to open the app store for a moment"),
        ("立刻打开应用商店", "Open the app store immediately"),
    ],
    "钱包": [
        ("打开钱包", "Open the wallet"),
        ("帮我打开手机钱包", "Help me open the phone wallet"),
        ("进入钱包应用", "Enter the wallet app"),
        ("启动钱包", "Launch the wallet"),
    ],
    "录音机": [
        ("打开录音机", "Open the recorder"),
        ("帮我打开录音", "Help me open the recorder"),
        ("启动录音机", "Launch the recorder"),
        ("我要录音打开录音机", "I want to record, open the recorder"),
        ("帮我打开录音机", "Help me open the recorder"),
        ("请打开录音机", "Please open the recorder"),
        ("给我打开录音机", "Open the recorder for me"),
        ("我要打开录音机", "I want to open the recorder"),
        ("打开录音", "Open recording"),
        ("打开录音工具", "Open the recording tool"),
        ("现在打开录音机", "Open the recorder now"),
        ("马上打开录音机", "Open the recorder right away"),
        ("帮我现在打开录音机", "Help me open the recorder now"),
        ("请帮我打开录音机", "Please help me open the recorder"),
        ("给我现在打开录音机", "Open the recorder for me now"),
        ("我要马上打开录音机", "I want to open the recorder right away"),
        ("打开一下录音机", "Open the recorder for a moment"),
        ("请打开一下录音机", "Please open the recorder for a moment"),
        ("给我打开一下录音机", "Open the recorder for me for a moment"),
        ("我要打开一下录音机", "I want to open the recorder for a moment"),
        ("立刻打开录音机", "Open the recorder immediately"),
        ("进入录音机", "Enter the recorder"),
    ],
    "下载管理器": [
        ("打开下载管理器", "Open the download manager"),
        ("帮我打开下载管理", "Help me open download management"),
        ("进入下载列表", "Enter the download list"),
        ("查看下载进度", "Check download progress"),
    ],
    "蓝牙": [
        ("打开蓝牙", "Turn on Bluetooth"),
        ("帮我打开蓝牙设置", "Help me open Bluetooth settings"),
        ("进入蓝牙管理", "Enter Bluetooth management"),
        ("开一下蓝牙", "Switch on Bluetooth"),
    ],
    "邮件": [
        ("打开邮件", "Open email"),
        ("帮我打开邮箱", "Help me open the mailbox"),
        ("进入邮件应用", "Enter the email app"),
        ("查看邮件", "Check emails"),
    ],
    "屏幕录制": [
        ("打开屏幕录制", "Open screen recording"),
        ("帮我开始录屏", "Help me start screen recording"),
        ("启动屏幕录制", "Launch screen recording"),
        ("打开录屏功能", "Open the screen recording feature"),
    ],
    "手机管家": [
        ("打开手机管家", "Open phone manager"),
        ("帮我打开手机管家", "Help me open the phone manager"),
        ("进入手机管家", "Enter the phone manager"),
        ("启动安全管家", "Launch the security manager"),
    ],
    "游戏中心": [
        ("打开游戏中心", "Open the game center"),
        ("帮我打开游戏中心", "Help me open the game center"),
        ("进入游戏大厅", "Enter the game lobby"),
        ("打开游戏商店", "Open the game store"),
    ],
    "视频": [
        ("打开视频", "Open video"),
        ("帮我打开视频播放器", "Help me open the video player"),
        ("启动视频应用", "Launch the video app"),
        ("打开本地视频", "Open local videos"),
    ],
    "阅读": [
        ("打开阅读", "Open reading"),
        ("帮我打开阅读应用", "Help me open the reading app"),
        ("进入阅读器", "Enter the reader"),
        ("启动阅读模式", "Launch reading mode"),
    ],
    "指南针": [
        ("打开指南针", "Open the compass"),
        ("帮我打开指南针", "Help me open the compass"),
        ("启动指南针", "Launch the compass"),
        ("看一下方向打开指南针", "Check the direction, open the compass"),
        ("请打开指南针", "Please open the compass"),
        ("给我打开指南针", "Open the compass for me"),
        ("我要打开指南针", "I want to open the compass"),
        ("打开指南针应用", "Open the compass app"),
        ("打开方向工具", "Open the direction tool"),
        ("现在打开指南针", "Open the compass now"),
        ("马上打开指南针", "Open the compass right away"),
        ("帮我现在打开指南针", "Help me open the compass now"),
        ("请帮我打开指南针", "Please help me open the compass"),
        ("给我现在打开指南针", "Open the compass for me now"),
        ("我要马上打开指南针", "I want to open the compass right away"),
        ("打开一下指南针", "Open the compass for a moment"),
        ("请打开一下指南针", "Please open the compass for a moment"),
        ("给我打开一下指南针", "Open the compass for me for a moment"),
        ("我要打开一下指南针", "I want to open the compass for a moment"),
        ("立刻打开指南针", "Open the compass immediately"),
        ("进入指南针", "Enter the compass"),
    ],
    "投屏": [
        ("打开投屏", "Open screen casting"),
        ("帮我打开投屏功能", "Help me open the screen casting feature"),
        ("启动无线投屏", "Launch wireless screen casting"),
        ("连接投屏", "Connect screen casting"),
    ],
}

# 品牌原生应用（联想/Moto）
BRAND_APPS = {
    "信息": [
        ("打开信息", "Open Messages"),
        ("帮我打开信息", "Help me open Messages"),
        ("进入信息应用", "Enter the Messages app"),
        ("启动信息", "Launch Messages"),
        ("请打开信息", "Please open Messages"),
        ("给我打开信息", "Open Messages for me"),
        ("我要打开信息", "I want to open Messages"),
        ("打开信息应用", "Open the Messages app"),
        ("现在打开信息", "Open Messages now"),
        ("马上打开信息", "Open Messages right away"),
        ("帮我现在打开信息", "Help me open Messages now"),
        ("请帮我打开信息", "Please help me open Messages"),
        ("给我现在打开信息", "Open Messages for me now"),
        ("我要马上打开信息", "I want to open Messages right away"),
        ("打开一下信息", "Open Messages for a moment"),
        ("请打开一下信息", "Please open Messages for a moment"),
        ("给我打开一下信息", "Open Messages for me for a moment"),
        ("立刻打开信息", "Open Messages immediately"),
        ("进入信息", "Enter Messages"),
    ],
    "moto文件": [
        ("打开moto文件", "Open Moto Files"),
        ("帮我打开moto文件", "Help me open Moto Files"),
        ("进入moto文件管理", "Enter Moto Files management"),
        ("启动moto文件", "Launch Moto Files"),
    ],
    "下载": [
        ("打开下载", "Open Downloads"),
        ("帮我打开下载", "Help me open Downloads"),
        ("进入下载", "Enter Downloads"),
        ("查看下载内容", "Check downloaded content"),
        ("请打开下载", "Please open Downloads"),
        ("给我打开下载", "Open Downloads for me"),
        ("我要打开下载", "I want to open Downloads"),
        ("打开下载管理", "Open Download Management"),
        ("打开下载中心", "Open the Download Center"),
        ("现在打开下载", "Open Downloads now"),
        ("马上打开下载", "Open Downloads right away"),
        ("帮我现在打开下载", "Help me open Downloads now"),
        ("请帮我打开下载", "Please help me open Downloads"),
        ("给我现在打开下载", "Open Downloads for me now"),
        ("我要马上打开下载", "I want to open Downloads right away"),
        ("打开一下下载", "Open Downloads for a moment"),
        ("请打开一下下载", "Please open Downloads for a moment"),
        ("给我打开一下下载", "Open Downloads for me for a moment"),
        ("我要打开一下下载", "I want to open Downloads for a moment"),
        ("立刻打开下载", "Open Downloads immediately"),
    ],
    "联想浏览器": [
        ("打开联想浏览器", "Open Lenovo Browser"),
        ("帮我打开联想浏览器", "Help me open Lenovo Browser"),
        ("启动联想浏览器", "Launch Lenovo Browser"),
        ("进入联想浏览器", "Enter Lenovo Browser"),
    ],
    "乐商店": [
        ("打开乐商店", "Open Le Store"),
        ("帮我打开乐商店", "Help me open Le Store"),
        ("进入乐商店", "Enter Le Store"),
        ("启动乐商店", "Launch Le Store"),
    ],
    "便笺": [
        ("打开便笺", "Open Sticky Notes"),
        ("帮我打开便笺", "Help me open Sticky Notes"),
        ("进入便笺", "Enter Sticky Notes"),
        ("启动便笺", "Launch Sticky Notes"),
        ("打开笔记", "Open Notes"),
        ("帮我打开笔记", "Help me open Notes"),
        ("请打开笔记", "Please open Notes"),
        ("给我打开笔记", "Open Notes for me"),
        ("我要打开笔记", "I want to open Notes"),
        ("打开便签", "Open Sticky Notes"),
        ("现在打开笔记", "Open Notes now"),
        ("马上打开笔记", "Open Notes right away"),
        ("帮我现在打开笔记", "Help me open Notes now"),
        ("请帮我打开笔记", "Please help me open Notes"),
        ("给我现在打开笔记", "Open Notes for me now"),
        ("我要马上打开笔记", "I want to open Notes right away"),
        ("打开一下笔记", "Open Notes for a moment"),
        ("请打开一下笔记", "Please open Notes for a moment"),
        ("给我打开一下笔记", "Open Notes for me for a moment"),
        ("我要打开一下笔记", "I want to open Notes for a moment"),
        ("立刻打开笔记", "Open Notes immediately"),
        ("进入笔记", "Enter Notes"),
        ("进入便签里记录一下", "Enter Sticky Notes to make a note"),
        ("打开笔记记一下", "Open Notes to write something down"),
        ("帮我记录", "Help me take a note"),
        ("记一下", "Take a note"),
    ],
    "Moto笔记": [
        ("打开Moto笔记", "Open Moto Note"),
        ("帮我打开Moto笔记", "Help me open Moto Note"),
        ("进入Moto笔记", "Enter Moto Note"),
        ("启动Moto笔记", "Launch Moto Note"),
    ],
    "截图": [
        ("打开截图", "Open Screenshot"),
        ("帮我打开截图", "Help me open Screenshot"),
        ("进入截图应用", "Enter the Screenshot app"),
        ("启动截图", "Launch Screenshot"),
    ],
    "截图编辑器": [
        ("打开截图编辑器", "Open Screenshot Editor"),
        ("帮我打开截图编辑器", "Help me open Screenshot Editor"),
        ("进入截图编辑器", "Enter Screenshot Editor"),
        ("启动截图编辑器", "Launch Screenshot Editor"),
    ],
    "联想天气": [
        ("打开联想天气", "Open Lenovo Weather"),
        ("帮我打开联想天气", "Help me open Lenovo Weather"),
        ("启动联想天气", "Launch Lenovo Weather"),
        ("查看联想天气", "Check Lenovo Weather"),
    ],
}

# Moto官方预装应用（13款）
MOTO_APPS = {
    "Moto": [
        ("打开Moto", "Open Moto"),
        ("帮我打开Moto", "Help me open Moto"),
        ("进入玩机技巧", "Enter Moto tips"),
        ("启动Moto应用", "Launch Moto app"),
    ],
    "Moto Gametime": [
        ("打开Moto Gametime", "Open Moto Gametime"),
        ("帮我打开游戏模式", "Help me open game mode"),
        ("启动游戏模式", "Launch game mode"),
        ("进入游戏中心", "Enter game center"),
    ],
    "Moto 显示": [
        ("打开Moto显示", "Open Moto Display"),
        ("帮我打开息屏显示", "Help me open Always-on Display"),
        ("启动息屏显示", "Launch Always-on Display"),
        ("进入Moto显示设置", "Enter Moto Display settings"),
    ],
    "Moto 语音": [
        ("打开Moto语音", "Open Moto Voice"),
        ("帮我打开语音助手", "Help me open voice assistant"),
        ("启动语音助手", "Launch voice assistant"),
        ("打开乐语音", "Open Le Voice"),
    ],
    "Moto 迁移": [
        ("打开Moto迁移", "Open Moto Transfer"),
        ("帮我打开一键换机", "Help me open one-click phone transfer"),
        ("启动一键换机", "Launch phone transfer"),
        ("进入数据迁移", "Enter data migration"),
    ],
    "Moto 安全": [
        ("打开Moto安全", "Open Moto Security"),
        ("帮我打开手机管家", "Help me open phone manager"),
        ("进入手机管家", "Enter phone manager"),
        ("启动安全中心", "Launch security center"),
    ],
    "Ready For": [
        ("打开Ready For", "Open Ready For"),
        ("帮我打开桌面模式", "Help me open desktop mode"),
        ("启动桌面模式", "Launch desktop mode"),
        ("进入投屏模式", "Enter casting mode"),
    ],
    "Moto 家庭空间": [
        ("打开Moto家庭空间", "Open Moto Family Space"),
        ("帮我打开家庭空间", "Help me open Family Space"),
        ("进入家庭空间", "Enter Family Space"),
        ("启动家庭管理", "Launch family management"),
    ],
    "Moto 音效": [
        ("打开Moto音效", "Open Moto Sound"),
        ("帮我打开音效设置", "Help me open sound settings"),
        ("启动杜比全景声", "Launch Dolby Atmos"),
        ("进入音效设置", "Enter sound settings"),
    ],
    "Moto 导览": [
        ("打开Moto导览", "Open Moto Guide"),
        ("帮我打开设备帮助", "Help me open device help"),
        ("进入设备帮助", "Enter device help"),
        ("启动新手引导", "Launch beginner guide"),
    ],
    "健康检测": [
        ("打开健康检测", "Open Health Monitoring"),
        ("帮我打开健康监测", "Help me open health monitoring"),
        ("启动健康检测", "Launch health monitoring"),
        ("查看健康数据", "Check health data"),
    ],
    "实时字幕": [
        ("打开实时字幕", "Open Live Caption"),
        ("帮我打开实时字幕", "Help me open Live Caption"),
        ("启动实时字幕", "Launch Live Caption"),
        ("开启字幕显示", "Turn on caption display"),
    ],
}

# 联想生态应用（6款）
LENOVO_APPS = {
    "超级互联": [
        ("打开超级互联", "Open Super Connect"),
        ("帮我打开超级互联", "Help me open Super Connect"),
        ("启动跨设备协同", "Launch cross-device sync"),
        ("进入设备互联", "Enter device connection"),
    ],
    "联想负一屏": [
        ("打开联想负一屏", "Open Lenovo Home Screen"),
        ("帮我打开负一屏", "Help me open the minus screen"),
        ("进入联想负一屏", "Enter Lenovo Home Screen"),
        ("启动智能助手", "Launch smart assistant"),
    ],
    "联想投屏": [
        ("打开联想投屏", "Open Lenovo Cast"),
        ("帮我打开联想投屏", "Help me open Lenovo Cast"),
        ("启动投屏功能", "Launch screen casting"),
        ("进入无线投屏", "Enter wireless casting"),
    ],
    "联想服务": [
        ("打开联想服务", "Open Lenovo Services"),
        ("帮我打开联想服务", "Help me open Lenovo Services"),
        ("进入售后服务", "Enter after-sales service"),
        ("启动联想客服", "Launch Lenovo customer service"),
    ],
    "联想账号": [
        ("打开联想账号", "Open Lenovo Account"),
        ("帮我打开联想账号", "Help me open Lenovo Account"),
        ("进入账号设置", "Enter account settings"),
        ("登录联想账号", "Log in to Lenovo Account"),
    ],
    "乐云": [
        ("打开乐云", "Open Le Cloud"),
        ("帮我打开乐云", "Help me open Le Cloud"),
        ("进入云服务", "Enter cloud services"),
        ("启动云同步", "Launch cloud sync"),
    ],
}

# 第三方预装应用（仅新增，去重已存在的）
THIRD_PARTY_PREINSTALLED = {
    "讯飞输入法": [
        ("打开讯飞输入法", "Open iFlytek Input"),
        ("帮我打开讯飞输入法", "Help me open iFlytek Input"),
        ("启动讯飞输入法", "Launch iFlytek Input"),
        ("打开讯飞", "Open iFlytek"),
    ],
}

# 大众装机必备APP（30款）
POPULAR_APPS = {
    "微信": [
        ("打开微信", "Open WeChat"),
        ("帮我打开微信", "Help me open WeChat"),
        ("启动微信", "Launch WeChat"),
        ("我要用微信", "I want to use WeChat"),
        ("进入微信", "Enter WeChat"),
    ],
    "QQ": [
        ("打开QQ", "Open QQ"),
        ("帮我打开QQ", "Help me open QQ"),
        ("启动QQ", "Launch QQ"),
        ("进入QQ聊天", "Enter QQ chat"),
    ],
    "支付宝": [
        ("打开支付宝", "Open Alipay"),
        ("帮我打开支付宝", "Help me open Alipay"),
        ("启动支付宝", "Launch Alipay"),
        ("进入支付宝", "Enter Alipay"),
    ],
    "哔哩哔哩": [
        ("打开哔哩哔哩", "Open Bilibili"),
        ("帮我打开B站", "Help me open Bilibili"),
        ("启动bilibili", "Launch Bilibili"),
        ("打开B站", "Open Bilibili"),
        ("进入哔哩哔哩", "Enter Bilibili"),
    ],
    "抖音": [
        ("打开抖音", "Open Douyin"),
        ("帮我打开抖音", "Help me open Douyin"),
        ("启动抖音", "Launch Douyin"),
        ("我要刷抖音", "I want to scroll Douyin"),
        ("进入抖音", "Enter Douyin"),
    ],
    "淘宝": [
        ("打开淘宝", "Open Taobao"),
        ("帮我打开淘宝", "Help me open Taobao"),
        ("启动淘宝", "Launch Taobao"),
        ("进入淘宝", "Enter Taobao"),
        ("我要逛淘宝", "I want to browse Taobao"),
    ],
    "京东": [
        ("打开京东", "Open JD.com"),
        ("帮我打开京东", "Help me open JD.com"),
        ("启动京东", "Launch JD.com"),
        ("进入京东商城", "Enter JD Mall"),
    ],
    "高德地图": [
        ("打开高德地图", "Open Amap"),
        ("帮我打开高德", "Help me open Amap"),
        ("启动高德导航", "Launch Amap navigation"),
        ("进入高德地图", "Enter Amap"),
    ],
    "小红书": [
        ("打开小红书", "Open Xiaohongshu"),
        ("帮我打开小红书", "Help me open Xiaohongshu"),
        ("启动小红书", "Launch Xiaohongshu"),
        ("进入小红书", "Enter Xiaohongshu"),
    ],
    "美团": [
        ("打开美团", "Open Meituan"),
        ("帮我打开美团", "Help me open Meituan"),
        ("启动美团", "Launch Meituan"),
        ("进入美团外卖", "Enter Meituan delivery"),
        ("打开美团APP", "Open Meituan app"),
    ],
    "拼多多": [
        ("打开拼多多", "Open Pinduoduo"),
        ("帮我打开拼多多", "Help me open Pinduoduo"),
        ("启动拼多多", "Launch Pinduoduo"),
        ("进入拼多多", "Enter Pinduoduo"),
    ],
    "快手": [
        ("打开快手", "Open Kuaishou"),
        ("帮我打开快手", "Help me open Kuaishou"),
        ("启动快手", "Launch Kuaishou"),
        ("我想刷快手", "I want to scroll Kuaishou"),
    ],
    "腾讯视频": [
        ("打开腾讯视频", "Open Tencent Video"),
        ("帮我打开腾讯视频", "Help me open Tencent Video"),
        ("启动腾讯视频", "Launch Tencent Video"),
        ("进入腾讯视频", "Enter Tencent Video"),
    ],
    "爱奇艺": [
        ("打开爱奇艺", "Open iQIYI"),
        ("帮我打开爱奇艺", "Help me open iQIYI"),
        ("启动爱奇艺", "Launch iQIYI"),
        ("进入爱奇艺", "Enter iQIYI"),
    ],
    "网易云音乐": [
        ("打开网易云音乐", "Open NetEase Music"),
        ("帮我打开网易云", "Help me open NetEase Music"),
        ("启动网易云音乐", "Launch NetEase Music"),
        ("进入网易云", "Enter NetEase Music"),
    ],
    "QQ音乐": [
        ("打开QQ音乐", "Open QQ Music"),
        ("帮我打开QQ音乐", "Help me open QQ Music"),
        ("启动QQ音乐", "Launch QQ Music"),
        ("进入QQ音乐", "Enter QQ Music"),
    ],
    "百度": [
        ("打开百度", "Open Baidu"),
        ("帮我打开百度", "Help me open Baidu"),
        ("启动百度APP", "Launch Baidu app"),
        ("进入百度搜索", "Enter Baidu Search"),
    ],
    "微博": [
        ("打开微博", "Open Weibo"),
        ("帮我打开微博", "Help me open Weibo"),
        ("启动微博", "Launch Weibo"),
        ("进入新浪微博", "Enter Sina Weibo"),
        ("我要刷微博", "I want to scroll Weibo"),
    ],
    "菜鸟裹裹": [
        ("打开菜鸟裹裹", "Open Cainiao"),
        ("帮我打开菜鸟", "Help me open Cainiao"),
        ("启动菜鸟裹裹", "Launch Cainiao"),
        ("查看快递打开菜鸟", "Check delivery, open Cainiao"),
    ],
    "12306": [
        ("打开12306", "Open 12306"),
        ("帮我打开铁路12306", "Help me open Railway 12306"),
        ("启动12306", "Launch 12306"),
        ("进入12306买票", "Enter 12306 to buy tickets"),
    ],
    "滴滴出行": [
        ("打开滴滴出行", "Open DiDi"),
        ("帮我打开滴滴", "Help me open DiDi"),
        ("启动滴滴", "Launch DiDi"),
        ("进入滴滴打车", "Enter DiDi for a ride"),
    ],
    "饿了么": [
        ("打开饿了么", "Open Ele.me"),
        ("帮我打开饿了么", "Help me open Ele.me"),
        ("启动饿了么", "Launch Ele.me"),
        ("进入饿了么点餐", "Enter Ele.me to order food"),
    ],
    "优酷视频": [
        ("打开优酷", "Open Youku"),
        ("帮我打开优酷视频", "Help me open Youku Video"),
        ("启动优酷", "Launch Youku"),
        ("进入优酷", "Enter Youku"),
    ],
    "番茄免费小说": [
        ("打开番茄小说", "Open Tomato Novel"),
        ("帮我打开番茄免费小说", "Help me open Tomato Free Novel"),
        ("启动番茄小说", "Launch Tomato Novel"),
        ("进入番茄看书", "Enter Tomato to read books"),
    ],
    "今日头条": [
        ("打开今日头条", "Open Toutiao"),
        ("帮我打开头条", "Help me open Toutiao"),
        ("启动今日头条", "Launch Toutiao"),
        ("进入今日头条看新闻", "Enter Toutiao to read news"),
    ],
    "迅雷": [
        ("打开迅雷", "Open Xunlei"),
        ("帮我打开迅雷", "Help me open Xunlei"),
        ("启动迅雷", "Launch Xunlei"),
        ("进入迅雷下载", "Enter Xunlei to download"),
    ],
    "WPS": [
        ("打开WPS", "Open WPS"),
        ("帮我打开WPS", "Help me open WPS"),
        ("启动WPS Office", "Launch WPS Office"),
        ("进入WPS办公", "Enter WPS Office"),
    ],
    "知乎": [
        ("打开知乎", "Open Zhihu"),
        ("帮我打开知乎", "Help me open Zhihu"),
        ("启动知乎", "Launch Zhihu"),
        ("进入知乎看看", "Enter Zhihu to browse"),
    ],
    "Keep": [
        ("打开Keep", "Open Keep"),
        ("帮我打开Keep", "Help me open Keep"),
        ("启动Keep运动", "Launch Keep fitness"),
        ("进入Keep健身", "Enter Keep workout"),
    ],
    "番茄ToDo": [
        ("打开番茄ToDo", "Open Tomato ToDo"),
        ("帮我打开番茄钟", "Help me open the Pomodoro timer"),
        ("启动番茄ToDo", "Launch Tomato ToDo"),
        ("进入番茄计时", "Enter Tomato timer"),
    ],
}


# 本地手机控制指令（label=0）：设备级本地操作，非打开APP
# 格式: "类别名": [("中文指令", "英文翻译", 0), ...]
LOCAL_CONTROLS = {
    "手机拍照": [
        ("帮我拍照", "Take a photo for me"),
        ("拍照", "Take a photo"),
        ("帮我拍张照", "Help me take a picture"),
        ("帮我手机拍照", "Take a photo with my phone"),
        ("手机拍照", "Take photos with the phone"),
        ("帮我用手机拍张照", "Help me take a picture with my phone"),
        ("给我拍张照", "Take a picture for me"),
        ("帮我拍一张照片", "Help me take a photo"),
        ("拍一张照片", "Take a photo"),
        ("帮我拍个照片", "Help me take a picture"),
        ("给我拍个照", "Take a picture for me"),
        ("帮忙拍照", "Help take a photo"),
        ("我要拍照", "I want to take a photo"),
        ("现在拍照", "Take a photo now"),
        ("立刻拍照", "Take a photo immediately"),
        ("马上拍照", "Take a photo right away"),
        ("帮我拍一张", "Help me take one"),
        ("帮我拍照片", "Help me take a picture"),
        ("拍一张吧", "Take one please"),
        ("拍一下", "Take a quick photo"),
        ("帮我来张照片", "Take a photo for me"),
        ("帮我照张相", "Help me take a picture"),
        ("手机拍一张", "Take one with the phone"),
        ("帮我用手机拍照", "Help me take a photo with the phone"),
        ("手机来一张", "Take one with the phone"),
        ("打开相机拍照", "Open the camera to take a photo"),
        ("快拍照", "Take a photo quickly"),
        ("帮我拍一下", "Help me take a quick photo"),
        ("拍个照片吧", "Take a photo please"),
        ("给我拍个照片吧", "Take a photo for me please"),
    ],
    "耳机拍照": [
        ("帮我耳机拍照", "Take a photo using the earphones"),
        ("耳机拍照", "Take photos via earphones"),
        ("帮我用耳机拍张照", "Help me take a picture with my earphones"),
        ("用耳机拍照", "Take a photo with earphones"),
        ("帮我用耳机拍照", "Help me take a photo with earphones"),
        ("耳机控制拍照", "Control photo capture with earphones"),
        ("用耳机帮我拍照", "Use earphones to take a photo for me"),
        ("耳机拍一张", "Take one with earphones"),
        ("耳机拍一张照片", "Take a photo with earphones"),
        ("耳机帮我拍张照", "Have earphones take a picture for me"),
        ("用耳机拍个照", "Take a picture with earphones"),
        ("耳机拍照片", "Take pictures with earphones"),
        ("耳机触发拍照", "Trigger photo capture via earphones"),
        ("耳机按键拍照", "Take a photo via earphone button"),
        ("用耳机进行拍照", "Use earphones to take a photo"),
        ("帮我用耳机拍一张", "Help me take one with earphones"),
        ("耳机帮我拍照片", "Use earphones to take pictures for me"),
        ("耳机拍照一下", "Take a quick photo with earphones"),
        ("耳机快速拍照", "Quickly take a photo with earphones"),
        ("耳机拍一张照片吧", "Take a photo with earphones please"),
        ("用耳机按下拍照", "Press earphones to take a photo"),
        ("耳机拍个照片吧", "Take a picture with earphones please"),
        ("耳机拍照功能", "Earphone photo function"),
        ("启动耳机拍照", "Activate earphone photo capture"),
        ("耳机控制相机拍照", "Use earphones to control the camera"),
        ("用耳机点击拍照", "Tap earphones to take a photo"),
        ("耳机执行拍照", "Execute photo capture via earphones"),
        ("耳机辅助拍照", "Earphone-assisted photo capture"),
    ],
    
    "上一曲": [
        ("上一曲", "Previous track"),
        ("上一首", "Previous song"),
        ("切换上一首", "Switch to the previous song"),
        ("放上一首", "Play the previous song"),
        ("返回上一曲", "Return to the previous track"),
        ("播放上一首歌", "Play the previous song"),
        ("上一首音乐", "Previous music"),
        ("上一首播放", "Play the previous one"),
        ("切到上一首", "Skip to the previous song"),
        ("换上一首", "Change to the previous song"),
        ("上一首歌曲", "Previous song"),
        ("切换到上一首", "Switch to the previous song"),
        ("回到上一曲", "Go back to the previous track"),
        ("上一首继续", "Continue with the previous song"),
        ("上一个歌曲", "Previous song"),
        ("往前一首", "Go back one song"),
        ("上一首音频", "Previous audio"),
        ("播放上一首", "Play the previous one"),
        ("切回上一首", "Switch back to the previous song"),
        ("回上一曲", "Back to the previous track"),
        ("上一首播放音乐", "Play previous music"),
        ("播放上一个", "Play the previous one"),
        ("上一个音乐", "Previous music"),
        ("上一首继续播放", "Resume from the previous song"),
        ("上一曲播放", "Play the previous track"),
        ("切换回上一个", "Switch back to the previous one"),
    ],
    "下一曲": [
        ("下一曲", "Next track"),
        ("下一首", "Next song"),
        ("切歌", "Switch to the next song"),
        ("跳过这首歌", "Skip this song"),
        ("切换下一首", "Switch to the next song"),
        ("放下一首", "Play the next song"),
        ("跳到下一首", "Jump to the next song"),
        ("播放下一首歌", "Play the next song"),
        ("下一首音乐", "Next music"),
        ("下一首播放", "Play the next one"),
        ("切到下一首", "Skip to the next song"),
        ("换下一首", "Change to the next song"),
        ("下一首歌曲", "Next song"),
        ("切换到下一首", "Switch to the next song"),
        ("继续下一曲", "Continue to the next track"),
        ("下一个歌曲", "Next song"),
        ("往后一首", "Go forward one song"),
        ("下一首音频", "Next audio"),
        ("播放下一首", "Play the next one"),
        ("跳到下一曲", "Jump to the next track"),
        ("下一首继续", "Continue with the next song"),
        ("下一首开始", "Start the next song"),
        ("切换下一曲播放", "Switch and play the next track"),
        ("下一个音乐", "Next music"),
        ("下一首继续播放", "Continue playing the next song"),
        ("下一曲播放", "Play the next track"),
        ("下一首歌播放", "Play the next song"),
        ("继续下一首", "Continue to the next song"),
    ],
    "音量增大": [
        ("音量增大", "Increase the volume"),
        ("调大音量", "Turn up the volume"),
        ("声音大一点", "Make it louder"),
        ("音量调大", "Turn up the volume"),
        ("把声音调大", "Turn the sound up"),
        ("音量加大", "Increase the volume"),
        ("提高音量", "Raise the volume"),
        ("声音放大", "Amplify the sound"),
        ("音量变大", "Make the volume louder"),
        ("声音调高", "Raise the sound"),
        ("把音量调高", "Turn up the volume"),
        ("音量往上调", "Adjust the volume up"),
        ("声音加大一点", "Make the sound a bit louder"),
        ("调高音量", "Raise the volume"),
        ("音量变高", "Make the volume higher"),
        ("增加音量", "Increase the volume"),
        ("音量大一点", "A bit louder please"),
        ("音量提升", "Boost the volume"),
        ("调大声音", "Turn up the sound"),
        ("声音再大一点", "A little louder please"),
        ("再大一点音量", "Volume a bit higher"),
        ("声音放大一点", "Amplify the sound a bit"),
        ("提高声音", "Raise the sound"),
        ("音量往上调一点", "Adjust the volume up a bit"),
        ("音量增加一点", "Increase the volume a bit"),
        ("声音加强", "Strengthen the sound"),
        ("调大一点音量", "Turn the volume up a bit"),
    ],
    "音量减小": [
        ("音量减小", "Decrease the volume"),
        ("调小音量", "Turn down the volume"),
        ("声音小一点", "Make it quieter"),
        ("音量调小", "Turn down the volume"),
        ("把声音调小", "Turn the sound down"),
        ("音量降低", "Lower the volume"),
        ("降低音量", "Reduce the volume"),
        ("声音变小", "Make the sound quieter"),
        ("音量变小", "Make the volume lower"),
        ("声音调低", "Lower the sound"),
        ("把音量调低", "Turn down the volume"),
        ("音量往下调", "Adjust the volume down"),
        ("声音减小一点", "Reduce the sound a bit"),
        ("调低音量", "Lower the volume"),
        ("音量减少", "Decrease the volume"),
        ("降低声音", "Lower the sound"),
        ("音量小一点", "A bit quieter please"),
        ("音量下降", "Volume down"),
        ("调小声音", "Turn down the sound"),
        ("声音再小一点", "A little quieter please"),
        ("再小一点音量", "Volume a bit lower"),
        ("声音弱一点", "Soften the sound a bit"),
        ("减小音量", "Reduce the volume"),
        ("音量往下调一点", "Adjust the volume down a bit"),
        ("音量减少一点", "Decrease the volume a bit"),
        ("声音降低", "Lower the sound"),
        ("调小一点音量", "Turn the volume down a bit"),
    ],
    "暂停播放": [
        ("暂停播放", "Pause playback"),
        ("停止播放", "Stop playback"),
        ("暂停", "Pause"),
        ("停止", "Stop"),
        ("先暂停", "Pause first"),
        ("暂停一下", "Pause for a moment"),
        ("把音乐暂停", "Pause the music"),
        ("暂停音乐", "Pause music"),
        ("先停一下", "Stop for a moment"),
        ("暂停当前播放", "Pause current playback"),
        ("暂停一下播放", "Pause playback for a moment"),
        ("音乐暂停", "Music paused"),
        ("停止一下", "Stop for a moment"),
        ("先停一下播放", "Stop playback for a moment"),
        ("暂停当前音乐", "Pause the current music"),
        ("暂停歌曲", "Pause the song"),
        ("暂停音频", "Pause audio"),
        ("停止音乐播放", "Stop music playback"),
        ("暂停音频播放", "Pause audio playback"),
        ("暂停当前歌曲", "Pause the current song"),
        ("暂停声音", "Pause the sound"),
        ("暂停这首歌", "Pause this song"),
        ("先暂停音乐", "Pause the music first"),
        ("暂停一下音乐", "Pause the music for a moment"),
        ("暂停一下歌曲", "Pause the song for a moment"),
        ("停止一下音乐", "Stop the music for a moment"),
        ("音乐先暂停", "Pause the music first"),
        ("先暂停播放", "Pause playback first"),
    ],    
    "继续播放": [
        ("播放音乐", "Play music"),
        ("继续播放", "Resume playback"),
        ("播放", "Play"),
        ("继续", "Resume"),
        ("恢复播放", "Resume playback"),
        ("继续播放音乐", "Continue playing music"),
        ("接着播放", "Continue playing"),
        ("继续放", "Keep playing"),
        ("恢复音乐播放", "Resume music playback"),
        ("继续播放歌曲", "Continue playing the song"),
        ("接着播", "Keep playing"),
        ("继续当前播放", "Continue the current playback"),
        ("开始播放", "Start playing"),
        ("播放歌曲", "Play the song"),
        ("播放当前歌曲", "Play the current song"),
        ("播放一下", "Play it"),
        ("直接播放", "Play directly"),
        ("立刻播放", "Play immediately"),
        ("马上播放", "Play right away"),
        ("开启播放", "Start playback"),
        ("启动播放", "Launch playback"),
        ("点播放", "Tap play"),
        ("按下播放", "Press play"),
        ("打开播放", "Turn on playback"),
        ("切到播放", "Switch to play"),
        ("播放模式", "Play mode"),
        ("开始听歌", "Start listening to music"),
    ],

    "静音": [
        ("静音", "Mute"),
        ("取消静音", "Unmute"),
    ],
    "通话": [
        ("接电话", "Answer the phone"),
        ("接听", "Answer the call"),
        ("挂电话", "Hang up the phone"),
        ("挂断", "Hang up"),
        ("拒接", "Reject the call"),
    ],
    "进度": [
        ("快进一下", "Fast forward a bit"),
        ("快退一下", "Rewind a bit"),
        ("快进", "Fast forward"),
        ("快退", "Rewind"),
        ("倒退一点", "Go back a bit"),
    ],
    "设备开关": [
        ("打开WiFi", "Turn on WiFi"),
        ("关闭WiFi", "Turn off WiFi"),
        ("开启飞行模式", "Turn on airplane mode"),
        ("关闭飞行模式", "Turn off airplane mode"),
        ("打开数据流量", "Turn on mobile data"),
        ("关掉数据流量", "Turn off mobile data"),
        ("打开个人热点", "Turn on personal hotspot"),
        ("关闭热点", "Turn off hotspot"),
        ("打开定位", "Turn on location"),
        ("关闭位置信息", "Turn off location services"),
        ("打开NFC", "Turn on NFC"),
        ("关闭NFC", "Turn off NFC"),
        ("开启勿扰模式", "Turn on do not disturb"),
        ("关闭免打扰", "Turn off do not disturb"),
        ("开启省电模式", "Turn on power saving mode"),
        ("关闭省电模式", "Turn off power saving mode"),
        ("打开自动旋转", "Turn on auto rotate"),
        ("锁定竖屏", "Lock portrait orientation"),
    ],
    "亮度": [
        ("亮度调高", "Increase brightness"),
        ("亮度调低", "Decrease brightness"),
        ("屏幕亮一点", "Make the screen brighter"),
        ("屏幕暗一点", "Make the screen dimmer"),
        ("亮度调到最高", "Set brightness to maximum"),
        ("亮度调到最低", "Set brightness to minimum"),
    ],
    "夜间模式": [
        ("打开护眼模式", "Turn on eye protection mode"),
        ("关闭护眼模式", "Turn off eye protection mode"),
        ("开启暗色模式", "Turn on dark mode"),
        ("关闭暗色模式", "Turn off dark mode"),
        ("打开夜间模式", "Turn on night mode"),
        ("关闭夜间模式", "Turn off night mode"),
    ],
    "锁屏": [
        ("锁屏", "Lock the screen"),
        ("锁定手机", "Lock the phone"),
        ("关闭屏幕", "Turn off the screen"),
        ("息屏", "Turn off display"),
    ],
    "定时提醒": [
        ("设一个闹钟", "Set an alarm"),
        ("明早七点叫我", "Wake me up at 7 tomorrow morning"),
        ("设一个30分钟的闹钟", "Set a 30 minute alarm"),
        ("倒计时5分钟", "Set a 5 minute countdown"),
        ("开始计时", "Start the timer"),
        ("提醒我下午三点开会", "Remind me of a meeting at 3 PM"),
        ("设置一个提醒", "Set a reminder"),
    ],
    "通讯发起": [
        ("打电话给张三", "Call Zhang San"),
        ("拨打10086", "Dial 10086"),
        ("给妈妈打电话", "Call mom"),
        ("给李四发条短信", "Send a text to Li Si"),
        ("发消息给妈妈说我到了", "Send a message to mom saying I arrived"),
        ("给张三发短信", "Send a text to Zhang San"),
    ],
    "信息查询": [
        ("现在几点了", "What time is it now"),
        ("今天星期几", "What day is it today"),
        ("还剩多少电", "How much battery is left"),
        ("查看电量", "Check battery level"),
        ("手机内存还有多少", "How much storage is left"),
        ("查看存储空间", "Check storage space"),
    ],
    "拍摄录制": [
        ("帮我录像", "Record a video for me"),
        ("开始录视频", "Start recording video"),
        ("录一段视频", "Record a video clip"),
        ("帮我录一段", "Record something for me"),
        ("截个屏", "Take a screenshot"),
        ("屏幕截图", "Screenshot"),
        ("截屏", "Capture the screen"),
    ],
    "系统操作": [
        ("重启手机", "Restart the phone"),
        ("关机重启", "Shut down and restart"),
        ("关机", "Shut down"),
        ("回到桌面", "Go to home screen"),
        ("返回上一页", "Go back to previous page"),
        ("返回主页", "Return to home"),
    ],
}

# 多步指令
muti_step_instruction = {
    "微信": [
        ("打开微信，发送谢谢给Bob", "Open WeChat and send \"Thank you\" to Bob."),
        ("打开微信发个红包给张三", "Open WeChat and send a red envelope to Zhang San."),
        ("进入微信给妈妈发条语音", "Enter WeChat and send a voice message to Mom."),
        ("打开微信视频通话给李四", "Open WeChat and video call Li Si."),
        ("打开微信扫一扫付款", "Open WeChat and scan to pay."),
    ],
    "支付宝": [
        ("打开支付宝扫码付款", "Open Alipay and scan to pay."),
        ("打开支付宝转账给张三200块", "Open Alipay and transfer 200 yuan to Zhang San."),
        ("进入支付宝查看余额", "Enter Alipay and check the balance."),
        ("打开支付宝缴纳水电费", "Open Alipay and pay utility bills."),
    ],
    "淘宝": [
        ("打开淘宝搜索蓝牙耳机", "Open Taobao and search for Bluetooth earphones."),
        ("进入淘宝看看购物车", "Enter Taobao and check the shopping cart."),
        ("打开淘宝领个优惠券", "Open Taobao and claim a coupon."),
    ],
    "高德地图": [
        ("打开高德地图导航到公司", "Open Amap and navigate to the office."),
        ("进入高德搜索附近加油站", "Enter Amap and search for nearby gas stations."),
        ("打开高德地图查看实时路况", "Open Amap and check real-time traffic."),
    ],
    "抖音": [
        ("打开抖音搜索做菜教程", "Open Douyin and search for cooking tutorials."),
        ("进入抖音发一条视频", "Enter Douyin and post a video."),
        ("打开抖音看直播", "Open Douyin and watch a livestream."),
    ],
    "相机": [
        ("打开相机录一段视频", "Open the camera and record a video."),
        ("打开相机切换到前置拍个自拍", "Open the camera, switch to front and take a selfie."),
        ("打开相机扫描这个二维码", "Open the camera and scan this QR code."),
    ],
    "设置": [
        ("打开设置连接WiFi", "Open settings and connect to WiFi."),
        ("进入设置修改密码", "Enter settings and change the password."),
        ("打开设置把蓝牙关掉", "Open settings and turn off Bluetooth."),
        ("打开设置调大字体", "Open settings and increase the font size."),
    ],
    "电话": [
        ("打开电话拨打10086", "Open the phone and dial 10086."),
        ("打开电话给妈妈打过去", "Open the phone and call Mom."),
    ],
    "信息": [
        ("打开信息给李四发条短信说我到了", "Open Messages and text Li Si \"I've arrived\"."),
        ("进入信息查看验证码", "Enter Messages and check the verification code."),
    ],
    "日历": [
        ("打开日历添加明天的会议", "Open the calendar and add tomorrow's meeting."),
        ("进入日历设置一个生日提醒", "Enter the calendar and set a birthday reminder."),
    ],
    "浏览器": [
        ("打开浏览器搜索天气预报", "Open the browser and search for the weather forecast."),
        ("打开浏览器查一下附近餐厅", "Open the browser and look up nearby restaurants."),
    ],
    "QQ": [
        ("打开QQ给班长发消息", "Open QQ and send a message to the class monitor."),
        ("进入QQ看看空间动态", "Enter QQ and check Qzone updates."),
    ],
    "百度": [
        ("打开百度搜索感冒怎么办", "Open Baidu and search how to treat a cold."),
        ("打开百度查一下高铁时刻表", "Open Baidu and check the train schedule."),
    ],
    "美团": [
        ("打开美团点一份外卖", "Open Meituan and order a delivery."),
        ("进入美团订一个酒店", "Enter Meituan and book a hotel."),
    ],
    "京东": [
        ("打开京东搜索笔记本电脑", "Open JD and search for laptops."),
        ("打开京东查看物流进度", "Open JD and check the delivery status."),
    ],
    "应用商店": [
        ("去应用商店下载app", "Download the app from the App Store"),
    ],
    "录音机": [
        ("开始录音", "Start recording"),
    ],
    "通用多步": [
        ("帮我打开音乐播放周杰伦的歌", "Help me open music and play Jay Chou's songs."),
        ("打开录音机开始录音", "Open the recorder and start recording."),
        ("打开相册删除昨天的照片", "Open the gallery and delete yesterday's photos."),
        ("打开计算器帮我算一下", "Open the calculator and help me calculate."),
        ("打开文件管理找到下载的文件", "Open file manager and find the downloaded file."),
        ("打开应用商店下载微信", "Open the app store and download WeChat."),
        ("打开时钟设一个七点的闹钟", "Open the clock and set an alarm for 7 AM."),
        ("进入设置看看还有多少存储空间", "Enter settings and check remaining storage."),
    ],
}

# 云端操作指令（label=1）：非本地操作、纯任务式、需联网
# 格式: [("中文指令", "英文翻译"), ...]
CLOUD_INSTRUCTIONS = [
    # --- 原始20条 ---
    ("发送一条线上文字消息给好友", "Send a text message to a friend online"),
    ("在线扫码完成线下付款", "Scan a code online to complete an offline payment"),
    ("搜索并在线播放网络视频", "Search and play online videos"),
    ("浏览线上短视频推荐内容", "Browse recommended short video content online"),
    ("查询两地出行路线并实时导航", "Query travel routes between two places and navigate in real time"),
    ("在线下单订购外卖餐饮", "Order food delivery online"),
    ("线上搜索商品并加入购物车", "Search for products online and add to cart"),
    ("查询快递实时物流运输状态", "Check real-time logistics status of a package"),
    ("在线浏览生活攻略与种草笔记", "Browse lifestyle guides and recommendation notes online"),
    ("在线查询并预订出行车票", "Query and book travel tickets online"),
    ("查看周边商家线上优惠活动", "Check online promotions from nearby stores"),
    ("在线追剧观影、观看网络剧集", "Watch dramas and movies online"),
    ("在线搜索并播放网络音乐歌曲", "Search and play music online"),
    ("查看全网实时热点热搜榜单", "Check real-time trending topics across the internet"),
    ("绑定快递单号查询物流轨迹", "Bind a tracking number and check logistics"),
    ("全网搜索问题答案和生活资讯", "Search the internet for answers and lifestyle information"),
    ("进入线上直播间观看并发表评论", "Enter an online live stream room to watch and comment"),
    ("搜索专业问题浏览网友解答", "Search professional questions and browse answers"),
    ("浏览全网实时新闻热点资讯", "Browse real-time news and hot topics"),
    ("登录网页端收发电子邮箱邮件", "Log in to webmail to send and receive emails"),
    # --- 社交通讯 ---
    ("在群聊里发一条语音消息", "Send a voice message in a group chat"),
    ("给朋友发送线上红包", "Send a digital red envelope to a friend"),
    ("在线发布一条朋友圈动态", "Post a moment on social media online"),
    ("查看好友最新发布的状态", "Check friends' latest status updates"),
    ("在线视频通话联系远方亲人", "Make a video call to distant family members"),
    ("在社交平台上关注一位博主", "Follow a blogger on social media"),
    ("给好友的动态点赞并评论", "Like and comment on a friend's post"),
    ("在线创建群聊邀请多人加入", "Create a group chat and invite people online"),
    ("分享一篇文章到社交群组", "Share an article to a social group"),
    ("在线查看并回复未读消息", "Check and reply to unread messages online"),
    # --- 购物电商 ---
    ("搜索对比多家店铺商品价格", "Search and compare product prices from multiple stores"),
    ("在线领取购物优惠券", "Claim shopping coupons online"),
    ("查看已购商品的售后进度", "Check after-sales progress of purchased items"),
    ("在线申请商品退货退款", "Apply for product return and refund online"),
    ("浏览线上限时秒杀活动商品", "Browse flash sale products online"),
    ("线上参与拼团购买优惠商品", "Join a group buying deal online"),
    ("在线查看购物车已选商品总价", "Check total price of items in the shopping cart online"),
    ("搜索附近超市线上购物配送", "Search nearby supermarkets for online shopping delivery"),
    ("在线咨询商家客服了解商品详情", "Consult seller customer service for product details online"),
    ("查看线上购物节满减活动规则", "Check online shopping festival discount rules"),
    ("在线比价寻找最优购买渠道", "Compare prices online to find the best purchase channel"),
    ("搜索并收藏感兴趣的商品", "Search and bookmark products of interest"),
    ("查看已关注店铺的上新通知", "Check new product notifications from followed stores"),
    ("在线查看商品用户评价和晒图", "View product reviews and user photos online"),
    ("提交订单并选择线上支付方式", "Submit an order and choose an online payment method"),
    # --- 出行交通 ---
    ("查询实时公交到站信息", "Query real-time bus arrival information"),
    ("在线预约网约车出行", "Book a ride-hailing trip online"),
    ("查看附近共享单车可用数量", "Check available shared bikes nearby"),
    ("在线规划自驾出行最佳路线", "Plan the best driving route online"),
    ("查询地铁线路换乘方案", "Query subway transfer options"),
    ("在线预订酒店住宿房间", "Book a hotel room online"),
    ("查看航班实时动态和延误信息", "Check real-time flight status and delay information"),
    ("在线购买长途汽车票", "Buy long-distance bus tickets online"),
    ("查询高速公路实时路况拥堵情况", "Check real-time highway traffic conditions"),
    ("在线办理机票值机选座", "Check in and select seats for a flight online"),
    ("搜索目的地旅游攻略和景点推荐", "Search for travel guides and attraction recommendations"),
    ("在线查看停车场空余车位", "Check available parking spaces online"),
    ("查询火车实时晚点信息", "Check real-time train delay information"),
    ("在线预约顺风车拼车出行", "Book a carpool ride online"),
    ("搜索附近加油站位置和油价", "Search nearby gas stations and fuel prices"),
    # --- 餐饮美食 ---
    ("搜索附近评分最高的餐厅", "Search for the highest-rated restaurants nearby"),
    ("在线浏览餐厅菜单和价格", "Browse restaurant menus and prices online"),
    ("在线预订今晚的餐厅座位", "Reserve a restaurant table for tonight online"),
    ("查看外卖配送员实时位置", "Check the delivery driver's real-time location"),
    ("在线点一杯奶茶外送到家", "Order a milk tea for home delivery online"),
    ("搜索附近可配送的早餐店铺", "Search nearby breakfast shops that deliver"),
    ("在线查看餐厅排队等位情况", "Check restaurant queue status online"),
    ("搜索适合聚餐的大桌餐厅", "Search for restaurants with large tables for gatherings"),
    ("在线搜索健康减脂餐推荐", "Search for healthy diet meal recommendations online"),
    ("查看本地美食榜单热门推荐", "Check local food ranking recommendations"),
    ("在线查询食品安全检测报告", "Check food safety inspection reports online"),
    ("搜索深夜还在营业的外卖店铺", "Search for late-night delivery restaurants"),
    # --- 生活服务 ---
    ("在线缴纳本月水电燃气费", "Pay monthly utility bills online"),
    ("线上预约家政保洁上门服务", "Book housekeeping services online"),
    ("在线查询社保公积金缴纳记录", "Check social insurance and housing fund records online"),
    ("在线预约医院门诊挂号", "Make a hospital appointment online"),
    ("查询附近药店库存和药品价格", "Check nearby pharmacy inventory and drug prices"),
    ("在线办理手机号充值话费", "Top up phone credit online"),
    ("搜索附近宠物医院和评价", "Search for nearby pet hospitals and reviews"),
    ("在线预约汽车保养维修服务", "Book car maintenance service online"),
    ("查询当地政务办事大厅预约号", "Check local government service appointment numbers"),
    ("在线查看小区物业公告通知", "Check community property announcements online"),
    ("搜索附近可用的充电桩位置", "Search for available charging stations nearby"),
    ("在线查询个人信用报告", "Check personal credit report online"),
    ("线上办理银行卡挂失冻结", "Report and freeze a bank card online"),
    ("在线查询违章记录并缴纳罚款", "Check traffic violations and pay fines online"),
    ("搜索附近快递驿站取件码", "Search for pickup codes at nearby courier stations"),
    # --- 娱乐影音 ---
    ("在线搜索本周新上映电影", "Search for new movies released this week"),
    ("搜索并在线收听有声书小说", "Search and listen to audiobooks online"),
    ("在线观看综艺节目最新一期", "Watch the latest variety show episode online"),
    ("搜索歌手演唱会线上购票渠道", "Search for concert ticket purchasing channels online"),
    ("在线创建并分享个人歌单", "Create and share a personal playlist online"),
    ("搜索热门播客节目在线收听", "Search for popular podcasts to listen online"),
    ("在线观看体育赛事直播", "Watch live sports events online"),
    ("搜索并订阅喜欢的网络小说", "Search and subscribe to favorite web novels"),
    ("在线查看电影评分和影评", "Check movie ratings and reviews online"),
    ("搜索线上KTV并在线唱歌", "Search for online KTV and sing online"),
    ("在线收听电台节目和脱口秀", "Listen to radio shows and talk shows online"),
    ("搜索最近热播剧集的更新进度", "Search for update progress of trending TV series"),
    # --- 学习教育 ---
    ("在线搜索免费学习课程资源", "Search for free learning course resources online"),
    ("在线报名参加技能培训班", "Sign up for a skills training class online"),
    ("搜索并观看线上公开课视频", "Search and watch online open course videos"),
    ("在线提交作业并查看批改结果", "Submit homework and check grading results online"),
    ("搜索外语在线翻译和例句", "Search for online translations and example sentences"),
    ("在线查看考试成绩和排名", "Check exam scores and rankings online"),
    ("搜索历年考试真题和答案解析", "Search for past exam papers and answer analysis"),
    ("在线参加线上模拟考试测验", "Take online mock exams"),
    ("搜索在线编程练习平台刷题", "Search for online coding practice platforms"),
    ("在线查看学校课程表和通知", "Check school schedule and notifications online"),
    ("搜索留学申请流程和材料清单", "Search for study abroad application process and materials"),
    ("在线咨询教育顾问规划学习路径", "Consult an education advisor for study planning online"),
    # --- 金融理财 ---
    ("在线查看股票实时行情走势", "Check real-time stock market trends online"),
    ("在线转账给他人银行账户", "Transfer money to another bank account online"),
    ("查看基金净值和收益变动", "Check fund net value and earnings changes"),
    ("在线申请个人消费贷款", "Apply for a personal consumer loan online"),
    ("查看信用卡本月账单明细", "Check this month's credit card statement"),
    ("在线购买定期理财产品", "Buy fixed-term financial products online"),
    ("搜索对比各银行存款利率", "Search and compare deposit rates across banks"),
    ("在线设置自动还款提醒", "Set up automatic repayment reminders online"),
    ("查看外汇实时汇率并换算", "Check real-time exchange rates and convert"),
    ("在线开通证券账户进行投资", "Open a securities account for investment online"),
    ("查询本月消费支出分类统计", "Query this month's spending breakdown"),
    ("在线申请提高信用卡额度", "Apply to increase credit card limit online"),
    # --- 健康医疗 ---
    ("在线问诊咨询医生健康问题", "Consult a doctor about health issues online"),
    ("搜索附近核酸检测点位置", "Search for nearby nucleic acid testing locations"),
    ("在线预约体检套餐并缴费", "Book and pay for a health checkup package online"),
    ("查看线上健康报告和体检结果", "Check online health reports and checkup results"),
    ("在线购买药品并配送到家", "Buy medicine online with home delivery"),
    ("搜索疾病症状和治疗方案", "Search for disease symptoms and treatment plans"),
    ("在线记录每日饮食和热量摄入", "Record daily diet and calorie intake online"),
    ("搜索附近可预约的牙科诊所", "Search for dental clinics nearby that accept appointments"),
    ("在线查看医保余额和使用记录", "Check medical insurance balance and usage records online"),
    ("搜索在线心理咨询师进行预约", "Search for online psychologists and make an appointment"),
    # --- 工作办公 ---
    ("在线编辑共享文档并协同办公", "Edit shared documents and collaborate online"),
    ("搜索并下载工作需要的模板", "Search and download work templates"),
    ("在线发起视频会议邀请同事", "Start a video meeting and invite colleagues online"),
    ("查看项目管理看板任务进度", "Check project management board task progress"),
    ("在线提交请假审批申请", "Submit a leave request for approval online"),
    ("搜索线上招聘信息并投递简历", "Search for job postings and submit resumes online"),
    ("在线查看工资条和薪资明细", "Check payslips and salary details online"),
    ("在线签署电子合同文件", "Sign electronic contracts online"),
    ("搜索行业报告和市场分析数据", "Search for industry reports and market analysis data"),
    ("在线预订会议室并发送通知", "Book a meeting room and send notifications online"),
    ("查看团队共享网盘中的文件", "Check files in the team's shared drive"),
    ("在线填写并提交工作日报周报", "Fill in and submit daily/weekly work reports online"),
    # --- 信息查询 ---
    ("搜索今日黄金价格实时走势", "Search for today's real-time gold price trends"),
    ("在线查询天气预报和穿衣建议", "Check weather forecast and clothing advice online"),
    ("搜索当地今日限行尾号信息", "Search for today's local traffic restriction info"),
    ("在线查询快递派送时效和费用", "Check delivery time and costs online"),
    ("搜索最新手机发布信息和参数", "Search for latest phone release info and specs"),
    ("在线查看本地招聘会时间地点", "Check local job fair schedule and location online"),
    ("搜索周边停车场收费标准", "Search for parking lot rates nearby"),
    ("在线查询电影放映场次和座位", "Check movie showtimes and seating online"),
    ("搜索当季热门旅游目的地推荐", "Search for popular travel destinations this season"),
    ("在线查看高考分数线和录取结果", "Check college entrance exam scores and admissions online"),
    ("搜索本地家装建材市场地址", "Search for local home renovation material markets"),
    ("在线查询航班准点率和机票价格", "Check flight punctuality and ticket prices online"),
    ("搜索附近健身房的会员价格", "Search for gym membership prices nearby"),
    ("在线查看二手房源信息和价格", "Check second-hand housing listings and prices online"),
    ("搜索当地美食节活动时间安排", "Search for local food festival schedules"),
    # --- 内容创作与互动 ---
    ("在线发布一条图文动态到社区", "Post a photo and text update to the community online"),
    ("编写一篇线上博客文章并发布", "Write and publish a blog post online"),
    ("在线上传短视频并添加配乐字幕", "Upload a short video with music and subtitles online"),
    ("给线上帖子撰写评论并发表", "Write and post a comment on an online thread"),
    ("在线参与话题讨论发表观点", "Participate in topic discussions and share opinions online"),
    ("搜索热门话题标签并参与互动", "Search for trending hashtags and engage online"),
    ("在线发起投票问卷收集意见", "Create an online poll to collect opinions"),
    ("给线上创作者打赏支持", "Tip and support an online creator"),
    ("在线分享个人运动健身成果", "Share personal fitness achievements online"),
    ("在线预约拍摄写真并查看样片", "Book a photo shoot and view samples online"),
    # --- 政务民生 ---
    ("在线预约办理身份证补换业务", "Book an ID card replacement appointment online"),
    ("线上申请居住证办理和续签", "Apply for residence permit online"),
    ("在线查询个人社保缴纳年限", "Check personal social insurance contribution years online"),
    ("线上办理出入境证件预约", "Book an exit-entry document appointment online"),
    ("在线申报个人所得税年度汇算", "File annual individual income tax online"),
    ("查询线上政务服务办事指南", "Check online government service guides"),
    ("在线提交营业执照注册申请", "Submit a business license registration online"),
    ("线上查询不动产登记信息", "Check real estate registration information online"),
    ("在线预约婚姻登记办理", "Book a marriage registration appointment online"),
    ("查询本地人才引进补贴政策", "Check local talent introduction subsidy policies"),
    # --- 租房搬家 ---
    ("搜索附近合适的租房房源信息", "Search for suitable rental listings nearby"),
    ("在线预约搬家公司上门服务", "Book a moving company service online"),
    ("查看线上房屋出租合同模板", "Check online rental contract templates"),
    ("搜索短租民宿并在线预订", "Search for short-term rentals and book online"),
    ("在线联系房东预约看房时间", "Contact the landlord to schedule a viewing online"),
    ("查询租房补贴申请条件和流程", "Check rental subsidy application conditions and process"),
    ("搜索对比多个平台租房价格", "Compare rental prices across multiple platforms"),
    ("在线签署电子租房合同", "Sign an electronic rental contract online"),
    ("查看合租室友匹配信息", "Check roommate matching information"),
    ("在线评估房屋租金市场行情", "Evaluate rental market trends online"),
    # --- 社区互助 ---
    ("在线发帖询问邻居推荐家电维修", "Post online asking neighbors for appliance repair recommendations"),
    ("搜索本地二手物品转让信息", "Search for local second-hand item listings"),
    ("在线发布闲置物品出售帖子", "Post idle items for sale online"),
    ("查看社区组织的线上活动报名", "Check community online event registrations"),
    ("在线查询附近便民服务网点", "Search for nearby convenience service locations online"),
    ("搜索本地家教辅导老师并预约", "Search for local tutors and make an appointment"),
    ("在线寻找同城拼车搭伴出行", "Find same-city carpooling partners online"),
    ("搜索附近可以遛狗的公园", "Search for nearby dog-friendly parks"),
    ("在线报名社区志愿者服务活动", "Sign up for community volunteer activities online"),
    ("搜索本地兴趣社团并申请加入", "Search for local hobby clubs and apply to join"),
]


# ============ ASR 语音变体映射 ============
# 常见语音识别(ASR)混淆字符映射，用于生成变体指令，增强测试集泛化性
# 每类错误: 原始字符 -> [可能的误识别字符]
PHONETIC_MAPS = {
    # --- 1. 平翘舌不分 (s/sh, z/zh, c/ch) ---
    "s_sh": {
        "设": ["色"], "是": ["四"], "说": ["梭"], "收": ["搜"],
        "生": ["僧"], "师": ["司"], "少": ["扫"], "深": ["森"],
        "时": ["司"], "什": ["丝"], "手": ["叟"], "身": ["森"],
        "声": ["僧"], "识": ["丝"], "实": ["词"], "使": ["死"],
        "试": ["四"], "市": ["四"], "室": ["寺"], "示": ["四"],
        "刷": ["撒"], "双": ["桑"], "帅": ["塞"], "睡": ["碎"],
        "顺": ["孙"], "书": ["苏"], "数": ["素"], "事": ["四"],
        "上": ["桑"], "水": ["嘴"],
    },
    "z_zh": {
        "这": ["则"], "只": ["子"], "找": ["早"], "站": ["赞"],
        "主": ["组"], "装": ["庄"], "中": ["宗"], "转": ["钻"],
        "真": ["针"], "知": ["资"], "重": ["众"], "住": ["租"],
        "照": ["造"], "正": ["赠"], "整": ["赠"], "指": ["子"],
        "准": ["尊"], "专": ["钻"], "张": ["脏"], "章": ["脏"],
        "助": ["租"], "注": ["租"], "证": ["赠"], "支": ["资"],
    },
    "c_ch": {
        "出": ["粗"], "吃": ["词"], "常": ["藏"], "成": ["层"],
        "传": ["窜"], "从": ["丛"], "窗": ["仓"], "唱": ["藏"],
        "超": ["操"], "查": ["擦"], "车": ["册"], "城": ["层"],
        "程": ["层"], "冲": ["聪"], "抽": ["凑"], "穿": ["窜"],
        "创": ["仓"], "存": ["层"],
    },

    # --- 2. n/l 不分 ---
    "n_l": {
        "你": ["里"], "那": ["辣"], "哪": ["拉"], "脑": ["老"],
        "男": ["兰"], "难": ["兰"], "年": ["连"], "弄": ["龙"],
        "暖": ["卵"], "浓": ["龙"], "牛": ["刘"], "奶": ["来"],
        "闹": ["涝"], "能": ["棱"], "拿": ["拉"], "内": ["类"],
        "南": ["兰"], "怒": ["路"], "泥": ["梨"], "念": ["恋"],
        "女": ["旅"], "尿": ["料"], "嫩": ["论"], "宁": ["灵"],
        "农": ["龙"], "诺": ["洛"],
    },

    # --- 3. 前后鼻音不分 (an/ang, en/eng, in/ing) ---
    "an_ang": {
        "进": ["近"], "心": ["星"], "新": ["星"], "信": ["性"],
        "今": ["京"], "金": ["京"], "紧": ["景"], "尽": ["境"],
        "清": ["轻"], "请": ["轻"], "情": ["晴"], "经": ["京"],
        "静": ["境"], "定": ["订"], "明": ["名"], "平": ["凭"],
        "名": ["民"], "领": ["林"], "听": ["廷"], "行": ["衡"],
        "影": ["隐"], "应": ["印"], "英": ["因"], "迎": ["银"],
        "兴": ["新"], "停": ["庭"], "令": ["林"], "零": ["林"],
    },

    # --- 4. 常见同音字 ---
    "homophone": {
        "打": ["大"], "机": ["鸡"], "启": ["起"], "帮": ["邦"],
        "用": ["佣"], "看": ["刊"], "想": ["响"], "到": ["道"],
        "了": ["乐"], "吗": ["麻"], "呢": ["尼"], "吧": ["八"],
        "啊": ["阿"], "过": ["锅"], "一": ["衣"], "个": ["各"],
        "下": ["夏"], "上": ["尚"], "开": ["揩"], "关": ["官"],
        "设": ["色"], "置": ["智"], "文": ["纹"], "件": ["见"],
        "管": ["馆"], "网": ["往"], "云": ["匀"], "盘": ["爬"],
        "台": ["抬"], "图": ["途"], "本": ["奔"], "夹": ["家"],
    },

    # --- 5. f/h 混淆 (南方方言) ---
    "f_h": {
        "发": ["花"], "飞": ["灰"], "分": ["昏"], "风": ["烘"],
        "方": ["荒"], "非": ["黑"], "房": ["黄"], "放": ["晃"],
        "翻": ["欢"], "饭": ["患"], "粉": ["混"], "富": ["护"],
        "服": ["湖"], "浮": ["湖"], "福": ["胡"], "辅": ["湖"],
        "封": ["烘"], "费": ["汇"], "返": ["缓"], "范": ["换"],
    },
}


def generate_phonetic_variants(text):
    """
    为中文指令生成 ASR 语音变体。
    对每个 PHONETIC_MAPS 类别，找到文本中可替换的字符，
    每次只替换一个字符（单字符变异），生成贴近真实 ASR 错误的变体。

    Returns: list of (variant_text, variant_type, source_text)
    """
    variants = []
    seen = set()

    for category, char_map in PHONETIC_MAPS.items():
        for i, ch in enumerate(text):
            if ch in char_map:
                for replacement in char_map[ch]:
                    variant_text = text[:i] + replacement + text[i + 1:]
                    key = (variant_text, category)
                    if variant_text != text and key not in seen:
                        seen.add(key)
                        variants.append((variant_text, category, text))

    return variants


def build_pinyin_reverse_map(char_pool):
    """
    根据字符池构建 拼音(去声调) -> {同音字} 反向映射。
    限定候选只来自数据集出现过的字符，避免引入生僻字。
    """
    rev = {}
    for ch in char_pool:
        for py in _get_pinyins(ch):
            rev.setdefault(py, set()).add(ch)
    return rev


# 每个字最多生成多少个拼音同音变体
MAX_PINYIN_CANDIDATES_PER_CHAR = 3


def generate_pinyin_variants(text, reverse_map, max_per_char=MAX_PINYIN_CANDIDATES_PER_CHAR):
    """
    通过拼音回译生成同音/多音字变体（模拟语音 -> 文字的 ASR 错误）。
    流程: 文本 -> 每字读音(含多音字) -> 同读音候选字 -> 替换单字
    候选范围限定在 reverse_map(数据集字符池)，避免生僻字。

    Returns: list of (variant_text, variant_type, source_text)
    """
    variants = []
    seen = set()

    for i, ch in enumerate(text):
        pys = _get_pinyins(ch)
        if not pys:
            continue
        # 收集所有同音(含多音)候选字
        candidates = set()
        for py in pys:
            candidates.update(reverse_map.get(py, set()))
        candidates.discard(ch)
        if not candidates:
            continue
        # 排序后取前 N 个，结果稳定
        sorted_cands = sorted(candidates)
        for replacement in sorted_cands[:max_per_char]:
            variant_text = text[:i] + replacement + text[i + 1:]
            if variant_text == text:
                continue
            key = variant_text
            if key in seen:
                continue
            seen.add(key)
            variants.append((variant_text, "pinyin", text))

    return variants


def build_dataset(with_variants=True):
    dataset = []
    counter = {"idx": 0}
    seen_text_keys = set()  # 去重: (text, app, lang)

    # 1. 收集所有本地中文指令出现过的汉字，构建拼音反向映射
    char_pool = set()
    for app_dict in (NATIVE_APPS, BRAND_APPS, MOTO_APPS, LENOVO_APPS,
                     THIRD_PARTY_PREINSTALLED, POPULAR_APPS):
        for instructions in app_dict.values():
            for item in instructions:
                cn = item[0]
                for ch in cn:
                    if "\u4e00" <= ch <= "\u9fff":
                        char_pool.add(ch)
    pinyin_rev_map = build_pinyin_reverse_map(char_pool) if with_variants else {}

    def add_record(text, label, label_name, app, app_type, lang, description,
                   function_name, variant_type=None, source_text=None):
        """添加一条记录，自动去重并分配 id。"""
        key = (text, app, lang)
        if key in seen_text_keys:
            return
        seen_text_keys.add(key)
        counter["idx"] += 1
        record = {
            "id": counter["idx"],
            "text": text,
            "label": label,
            "label_name": label_name,
            "function_name": function_name,
            "app": app,
            "app_en": APP_NAME_EN.get(app, app) if app else None,
            "app_type": app_type,
            "lang": lang,
            "description": description,
        }
        if variant_type is not None:
            record["variant_type"] = variant_type
            record["source_text"] = source_text
        dataset.append(record)

    def emit_app_records(app_dict, app_type, description):
        """对一组APP指令: 发出原始中英文记录，并对中文生成语音变体。
        指令格式: (cn, en) 或 (cn, en, label)。
        label 解析顺序: 先按白名单规则取默认值，若 item[2] 显式给出(非 None) 则覆盖。"""
        for app_name, instructions in app_dict.items():
            for item in instructions:
                cn, en = item[0], item[1]
                # 1) 白名单默认 label
                label = _get_app_label(app_name, app_type)
                # 2) 硬编码覆盖（默认空，存在且非 None 才覆盖）
                if len(item) > 2 and item[2] is not None:
                    label = item[2]
                label_name = "local" if label == 0 else "cloud"
                add_record(cn, label, label_name, app_name, app_type, "zh", description,
                           function_name="open_app")
                add_record(en, label, label_name, app_name, app_type, "en", description,
                           function_name="open_app")
                if with_variants:
                    for var_text, var_type, src in generate_phonetic_variants(cn):
                        add_record(var_text, label, label_name, app_name, app_type, "zh",
                                   description, function_name="open_app",
                                   variant_type=var_type, source_text=src)
                    for var_text, var_type, src in generate_pinyin_variants(cn, pinyin_rev_map):
                        add_record(var_text, label, label_name, app_name, app_type, "zh",
                                   description, function_name="open_app",
                                   variant_type=var_type, source_text=src)

    # 本地操作 (打开APP/手机功能) - 含语音变体
    emit_app_records(NATIVE_APPS, "native", "安卓原生自带应用")
    emit_app_records(BRAND_APPS, "brand", "品牌原生应用(联想/Moto)")
    emit_app_records(MOTO_APPS, "moto", "Moto官方预装应用")
    emit_app_records(LENOVO_APPS, "lenovo", "联想生态应用")
    emit_app_records(THIRD_PARTY_PREINSTALLED, "preinstalled", "第三方预装应用")
    emit_app_records(POPULAR_APPS, "popular", "大众装机必备应用")

    # 本地手机控制指令 - 先按白名单规则取默认值，硬编码可覆盖
    for ctrl_name, instructions in LOCAL_CONTROLS.items():
        ctrl_desc = f"本地手机控制指令-{ctrl_name}"
        for item in instructions:
            cn, en = item[0], item[1]
            # 1) 白名单默认 label
            label = _get_app_label(ctrl_name, "local_control", ctrl_name=ctrl_name)
            # 2) 硬编码覆盖（默认空）
            if len(item) > 2 and item[2] is not None:
                label = item[2]
            label_name = "local" if label == 0 else "cloud"
            add_record(cn, label, label_name, ctrl_name, "local_control", "zh", ctrl_desc,
                       function_name="keywords_operation")
            add_record(en, label, label_name, ctrl_name, "local_control", "en", ctrl_desc,
                       function_name="keywords_operation")
            if with_variants:
                for var_text, var_type, src in generate_phonetic_variants(cn):
                    add_record(var_text, label, label_name, ctrl_name, "local_control", "zh",
                               ctrl_desc, function_name="keywords_operation",
                               variant_type=var_type, source_text=src)
                for var_text, var_type, src in generate_pinyin_variants(cn, pinyin_rev_map):
                    add_record(var_text, label, label_name, ctrl_name, "local_control", "zh",
                               ctrl_desc, function_name="keywords_operation",
                               variant_type=var_type, source_text=src)

    # 云端操作 - 不生成变体（专注分类边界，避免噪声）
    cloud_desc = "云端操作-非本地指令/纯任务式/需联网"
    for item in CLOUD_INSTRUCTIONS:
        cn, en = item[0], item[1]
        # 1) 白名单默认 label (cloud_task -> 1)
        label = _get_app_label(None, "cloud_task")
        # 2) 硬编码覆盖（默认空）
        if len(item) > 2 and item[2] is not None:
            label = item[2]
        label_name = "local" if label == 0 else "cloud"
        add_record(cn, label, label_name, None, "cloud_task", "zh", cloud_desc,
                   function_name="online_chat")
        add_record(en, label, label_name, None, "cloud_task", "en", cloud_desc,
                   function_name="online_chat")

    # 多步指令 - 默认 cloud (label=1)，硬编码可覆盖
    multi_desc = "多步指令-打开APP后执行操作"
    for app_name, instructions in muti_step_instruction.items():
        for item in instructions:
            cn, en = item[0], item[1]
            # 1) 白名单默认 label (multi_step -> 1)
            label = _get_app_label(app_name, "multi_step")
            # 2) 硬编码覆盖（默认空）
            if len(item) > 2 and item[2] is not None:
                label = item[2]
            label_name = "local" if label == 0 else "cloud"
            add_record(cn, label, label_name, app_name, "multi_step", "zh", multi_desc,
                       function_name="online_chat")
            add_record(en, label, label_name, app_name, "multi_step", "en", multi_desc,
                       function_name="online_chat")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成手机语音指令分类数据集")
    parser.add_argument("--no-variants", action="store_true",
                        help="不生成ASR语音变体，仅保留原始指令")
    parser.add_argument("-o", "--output", default="./dataset/val_data_mobile_instructions.json",
                        help="输出JSON文件路径 (默认: ./dataset/val_data_mobile_instructions.json)")
    args = parser.parse_args()

    with_variants = not args.no_variants
    output_path = args.output

    dataset = build_dataset(with_variants=with_variants)

    # 统计 - 按 app_type
    native_count = sum(1 for d in dataset if d["app_type"] == "native")
    brand_count = sum(1 for d in dataset if d["app_type"] == "brand")
    moto_count = sum(1 for d in dataset if d["app_type"] == "moto")
    lenovo_count = sum(1 for d in dataset if d["app_type"] == "lenovo")
    preinstalled_count = sum(1 for d in dataset if d["app_type"] == "preinstalled")
    popular_count = sum(1 for d in dataset if d["app_type"] == "popular")
    cloud_count = sum(1 for d in dataset if d["app_type"] == "cloud_task")
    local_ctrl_count = sum(1 for d in dataset if d["app_type"] == "local_control")
    local_ctrl_local = sum(1 for d in dataset if d["app_type"] == "local_control" and d["label"] == 0)
    local_ctrl_cloud = sum(1 for d in dataset if d["app_type"] == "local_control" and d["label"] == 1)
    zh_count = sum(1 for d in dataset if d["lang"] == "zh")
    en_count = sum(1 for d in dataset if d["lang"] == "en")
    native_apps = len(NATIVE_APPS)
    brand_apps = len(BRAND_APPS)
    moto_apps = len(MOTO_APPS)
    lenovo_apps = len(LENOVO_APPS)
    preinstalled_apps = len(THIRD_PARTY_PREINSTALLED)
    popular_apps = len(POPULAR_APPS)

    local_count = sum(1 for d in dataset if d["label"] == 0)
    total_cloud_count = sum(1 for d in dataset if d["label"] == 1)

    # 统计 - 语音变体
    variant_count = sum(1 for d in dataset if d.get("variant_type"))
    original_count = len(dataset) - variant_count
    variant_per_category = {}
    for cat in list(PHONETIC_MAPS.keys()) + ["pinyin"]:
        variant_per_category[cat] = sum(1 for d in dataset if d.get("variant_type") == cat)

    print(f"数据集生成完成:")
    print(f"  ASR变体: {'开启' if with_variants else '关闭'}")
    print(f"  安卓原生应用: {native_apps} 款, {native_count} 条指令 (local)")
    print(f"  品牌原生应用: {brand_apps} 款, {brand_count} 条指令 (local)")
    print(f"  Moto预装应用: {moto_apps} 款, {moto_count} 条指令 (local)")
    print(f"  联想生态应用: {lenovo_apps} 款, {lenovo_count} 条指令 (local)")
    print(f"  第三方预装:   {preinstalled_apps} 款, {preinstalled_count} 条指令 (local)")
    print(f"  装机必备应用: {popular_apps} 款, {popular_count} 条指令 (local)")
    print(f"  本地控制指令: {local_ctrl_count} 条 (label=0:{local_ctrl_local}, label=1:{local_ctrl_cloud})")
    print(f"  云端操作指令: {cloud_count} 条 (cloud)")
    print(f"  中文: {zh_count} 条, 英文: {en_count} 条")
    print(f"  原始样本: {original_count} 条, 语音变体: {variant_count} 条")
    if with_variants:
        print(f"  语音变体分类:")
        for cat, cnt in variant_per_category.items():
            print(f"    {cat:12s}: {cnt} 条")
    print(f"  总计: {len(dataset)} 条指令 (label=0本地:{local_count} / label=1云端:{total_cloud_count})")

    # 自动创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "task": "用户意图二分类 - 打开APP指令 vs 云端操作",
                "with_variants": with_variants,
                "total_apps": native_apps + brand_apps + moto_apps + lenovo_apps + preinstalled_apps + popular_apps,
                "total_samples": len(dataset),
                "original_samples": original_count,
                "variant_samples": variant_count,
                "variant_per_category": variant_per_category,
                "native_apps": native_apps,
                "native_samples": native_count,
                "brand_apps": brand_apps,
                "brand_samples": brand_count,
                "moto_apps": moto_apps,
                "moto_samples": moto_count,
                "lenovo_apps": lenovo_apps,
                "lenovo_samples": lenovo_count,
                "preinstalled_apps": preinstalled_apps,
                "preinstalled_samples": preinstalled_count,
                "popular_apps": popular_apps,
                "popular_samples": popular_count,
                "local_control_samples": local_ctrl_count,
                "local_control_local": local_ctrl_local,
                "local_control_cloud": local_ctrl_cloud,
                "cloud_samples": cloud_count,
                "zh_samples": zh_count,
                "en_samples": en_count,
                "label_definition": {
                    "0": "local - 仅限白名单App(电话/信息/相机/时钟/微信/QQ/抖音等48个)的打开指令 + 白名单操作(拍照/切歌/音量/播放)",
                    "1": "cloud - 非白名单App的打开指令 + 非白名单设备操作(锁屏/WiFi/闹钟等) + 云端任务"
                },
                "variant_type_definition": {
                    "s_sh": "平翘舌不分 (s/sh)",
                    "z_zh": "平翘舌不分 (z/zh)",
                    "c_ch": "平翘舌不分 (c/ch)",
                    "n_l": "n/l 不分",
                    "an_ang": "前后鼻音不分 (an/ang, en/eng, in/ing)",
                    "homophone": "常见同音字混淆 (人工映射)",
                    "f_h": "f/h 混淆 (南方方言)",
                    "pinyin": "拼音回译同音/多音字 (基于 pypinyin 自动生成)"
                }
            },
            "data": dataset
        }, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {output_path}")

    # 展示部分样例
    print("\n原始样例预览 (前10条):")
    for s in [d for d in dataset if not d.get("variant_type")][:10]:
        app_str = s['app'] or "-"
        print(f"  [{s['app_type']:12s}] [{s['lang']}] {app_str:10s} | {s['text']}")

    if with_variants:
        print("\n语音变体样例预览 (每类2条):")
        for cat in list(PHONETIC_MAPS.keys()) + ["pinyin"]:
            samples = [d for d in dataset if d.get("variant_type") == cat][:2]
            for s in samples:
                print(f"  [{cat:10s}] {s['source_text']:20s} -> {s['text']}")
