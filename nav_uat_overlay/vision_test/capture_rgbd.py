import argparse
from datetime import datetime
from pathlib import Path


CAMERA_SERIAL_NUMBERS = {
    "d455": "419222302306",
    "d435i": "406122070550",
}
DEFAULT_CAMERA = "d455"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one RGB image and one depth image from the RealSense RGBD camera."
    )
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_SERIAL_NUMBERS),
        default=DEFAULT_CAMERA,
        help="Known RealSense camera preset.",
    )
    parser.add_argument(
        "--serial-number",
        default=None,
        help="Override the RealSense serial number selected by --camera.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before saving, for exposure/depth stabilization.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Timeout for the frame that will be saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "captures",
        help="Directory for saved RGB and depth images.",
    )
    return parser.parse_args()


def resolve_serial_number(args):
    if args.serial_number:
        return args.serial_number
    return CAMERA_SERIAL_NUMBERS[args.camera]


def configure_pipeline(args):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    serial_number = resolve_serial_number(args)
    if serial_number:
        config.enable_device(serial_number)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )
    return pipeline, config, serial_number


def save_rgbd_images(output_dir, camera_name, color_image, depth_m):
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{camera_name}_{timestamp}"

    rgb_path = output_dir / f"rgb_{prefix}.jpg"
    depth_npy_path = output_dir / f"depth_m_{prefix}.npy"
    depth_mm_path = output_dir / f"depth_mm_{prefix}.png"
    depth_vis_path = output_dir / f"depth_vis_{prefix}.jpg"

    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    valid_depth = depth_m[depth_m > 0]
    if valid_depth.size:
        max_depth = float(np.percentile(valid_depth, 95))
    else:
        max_depth = 1.0
    max_depth = max(max_depth, 1e-6)
    depth_vis = np.clip(depth_m / max_depth * 255.0, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    if not cv2.imwrite(str(rgb_path), color_image):
        raise RuntimeError(f"Failed to save RGB image: {rgb_path}")
    if not cv2.imwrite(str(depth_mm_path), depth_mm):
        raise RuntimeError(f"Failed to save depth PNG: {depth_mm_path}")
    if not cv2.imwrite(str(depth_vis_path), depth_vis):
        raise RuntimeError(f"Failed to save depth visualization: {depth_vis_path}")
    np.save(depth_npy_path, depth_m)

    return {
        "rgb": rgb_path,
        "depth_m_npy": depth_npy_path,
        "depth_mm_png": depth_mm_path,
        "depth_visualization": depth_vis_path,
    }


def main():
    import numpy as np

    args = parse_args()
    pipeline, config, serial_number = configure_pipeline(args)
    pipeline_started = False

    try:
        print(f"Using camera={args.camera}, serial_number={serial_number}")
        pipeline.start(config)
        pipeline_started = True

        for _ in range(max(0, args.warmup_frames)):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            raise RuntimeError("Failed to get RGB frame.")
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame.")

        color_image = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
        saved_paths = save_rgbd_images(args.output_dir, args.camera, color_image, depth_m)

        print(f"RGB shape: {color_image.shape[1]} x {color_image.shape[0]}")
        print(f"Depth shape: {depth_m.shape[1]} x {depth_m.shape[0]}")
        for name, path in saved_paths.items():
            print(f"{name}: {path.resolve()}")
    finally:
        if pipeline_started:
            pipeline.stop()


if __name__ == "__main__":
    main()
