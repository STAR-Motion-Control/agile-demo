import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


SERIAL_NUMBER = "419222302306"

WIDTH = 640
HEIGHT = 480
FPS = 30

SAVE_DIR = Path("./captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # 指定 D455 序列号
    config.enable_device(SERIAL_NUMBER)

    # 采集 RGB 彩色图像
    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.rgb8,
        FPS
    )

    try:
        pipeline.start(config)

        # 预热，避免刚启动时曝光、白平衡未稳定
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames(5000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("未获取到 RGB 图像")

        # RealSense RGB -> OpenCV BGR
        rgb_image = np.asanyarray(color_frame.get_data())
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        save_name = f"rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = SAVE_DIR / save_name

        success = cv2.imwrite(str(save_path), bgr_image)

        if not success:
            raise RuntimeError("图像保存失败")

        print(f"图像已保存：{save_path.resolve()}")
        print(f"图像尺寸：{bgr_image.shape[1]} x {bgr_image.shape[0]}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()