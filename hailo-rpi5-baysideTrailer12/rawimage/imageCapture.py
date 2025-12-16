import cv2
import time
import os

# 你的 RTSP 地址
# 注意：102 通常是子码流(低清)，如果你想要高清大图，通常应该改为 101
RTSP_URL = "rtsp://admin:31415926Pi@172.16.109.101:554/Streaming/Channels/101"


def capture_one_frame():
    print(f"尝试连接到: {RTSP_URL} ...")

    # 技巧：强制 FFMPEG 使用 TCP 传输，比默认的 UDP 更稳定，能防止花屏
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    # 初始化视频捕获
    cap = cv2.VideoCapture(RTSP_URL)

    # 检查是否成功打开
    if not cap.isOpened():
        print("错误：无法打开 RTSP 视频流。请检查 IP、密码或网络连接。")
        return

    # 有时候第一帧是灰色的或者正在调整曝光，建议丢弃前几帧
    # 这里我们读取并丢弃前 5 帧，取第 6 帧保存
    for i in range(5):
        cap.read()

    # 正式读取一帧
    ret, frame = cap.read()

    if ret:
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"成功！图片已保存为: {filename}")
        print(f"图片分辨率: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("错误：无法读取帧数据。")

    # 释放资源
    cap.release()


if __name__ == "__main__":
    capture_one_frame()
