# Router Model - 本地/云端语音指令分类

## 概述

本模块用于生成移动端语音指令的二分类训练数据，将用户语音指令分为：
- **A类 (label=0, 本地)**：可在本地执行的指令（打开白名单内 App、本地设备控制）
- **B类 (label=1, 云端)**：需要云端处理的指令（非白名单 App、云端任务、多步操作）

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_app_dataset.py` | 数据集生成脚本，基于白名单规则生成带标签的训练样本 |
| `dataset/val_data_whitelist_20260529.json` | 生成的训练数据（1939 条，label=0: 1248 / label=1: 691） |

## 数据分类规则

### 本地指令 (label=0)
- 单步打开 App 指令（如打开相机、启动音乐等）
- 8 类本地控制操作（手机拍照、耳机拍照、上/下一曲、音量增减、暂停/继续播放）

### 云端指令 (label=1)
- 打开非白名单 App
- 设备控制操作（静音、通话、亮度、锁屏、定时提醒等）
- 云端任务（搜索、购物、聊天、查天气、导航等）
- 多步复合操作（打开XX并执行YY）

## 数据字段

每条记录包含：
- `text`: 指令文本
- `label`: 0(本地) / 1(云端)
- `label_name`: local / cloud
- `function_name`: open_app / keywords_operation / online_chat
- `app`: 关联的 App 或控制名称
- `app_en`: App 英文名称
- `app_type`: native / brand / preinstalled / popular / local_control / cloud_task / multi_step
- `lang`: zh / en

## 使用方法

```bash
# 生成数据（无语音变体）
python generate_app_dataset.py --no-variants -o ./dataset/val_data_whitelist_20260529.json

# 生成数据（含语音变体，用于鲁棒性训练）
python generate_app_dataset.py -o ./dataset/val_data_with_variants.json
```

