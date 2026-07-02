#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 引导模块：将机器人头部相机图像发送给千问/Qwen VLM，
获取箱子位置和移动建议。
"""

import base64
import json
import re

import cv2
import numpy as np
from openai import OpenAI

_SYSTEM_PROMPT = """\
You are a robot vision system guiding a humanoid robot to approach a cardboard box.
The robot has a head-mounted camera. You will receive TWO images:

1. RGB image - the regular color view from the robot's head camera
2. Depth colormap - JET colormap visualization of depth (red = close, blue = far)
   The depth values are in millimeters. Use this to estimate actual distances.

Your task:
1. Is there a box/cardboard box visible in the RGB image?
2. If yes, is the box fully visible or partially cut off at the image edge?
   - "complete": the whole box is visible in the frame
   - "left_cut": the box is cut off on the LEFT side of the image
   - "right_cut": the box is cut off on the RIGHT side of the image
   - "both_cut": boxes are cut off on BOTH sides of the image
   - "top_cut": the box is cut off on the TOP side of the image (robot is too close)
3. If yes, use the depth image to estimate how far the box is, then determine
   how long the robot should walk to get within arm's reach (arms reach ~0.3-0.5m).
   The robot walks at about 0.2 m/s.

Movement conventions (from the robot's perspective):
- move_x_ms: positive ms = walk FORWARD, negative ms = walk BACKWARD
- move_y_ms: positive ms = walk LEFT (strafe left), negative ms = walk RIGHT (strafe right)

Respond in valid JSON only, no markdown, no extra text:
{"box_visible": true/false, "box_in_frame": "complete"/"left_cut"/"right_cut"/"both_cut"/"top_cut",
 "move_x_ms": number, "move_y_ms": number,
 "confidence": "high"/"medium"/"low", "description": "brief explanation in English"}"""


class VLMGuide:
    """调用千问 VLM API 获取箱子引导信息。"""

    def __init__(self, endpoint: str, api_key: str, model: str = "qwen-vl-max"):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=endpoint)

    def query(self, image_bgr: np.ndarray, depth_u16: np.ndarray = None) -> dict:
        """
        发送 RGB 图像和深度图给 VLM，返回移动建议。

        Args:
            image_bgr: BGR 格式的 numpy 图像数组 (H, W, 3)
            depth_u16: 深度图 (H, W) uint16 毫米，可选

        Returns:
            dict: {"box_visible": bool, "move_x_ms": float, "move_y_ms": float,
                   "confidence": str, "description": str}
            出错时返回 {"box_visible": False, "error": str}
        """
        h, w = image_bgr.shape[:2]

        # RGB: 直接编码原图（"low" detail 会自动缩放，预缩只会损失细节）
        ok, rgb_jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return {"box_visible": False, "error": "RGB JPEG encode failed"}
        rgb_uri = f"data:image/jpeg;base64,{base64.b64encode(rgb_jpg.tobytes()).decode('utf-8')}"

        content = [
            {"type": "text", "text": "Where is the box? Give movement directions in JSON."},
            {"type": "image_url", "image_url": {"url": rgb_uri, "detail": "auto"}},
        ]

        # Depth: 色映射可视化 + JPEG 编码
        if depth_u16 is not None:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_u16, alpha=0.05), cv2.COLORMAP_JET)
            ok, depth_jpg = cv2.imencode(".jpg", depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                depth_uri = f"data:image/jpeg;base64,{base64.b64encode(depth_jpg.tobytes()).decode('utf-8')}"
                content.append(
                    {"type": "image_url", "image_url": {"url": depth_uri, "detail": "auto"}})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            print(f"  [VLM 原始返回] {raw[:500]}")
        except IndexError:
            return {"box_visible": False, "error": "Empty response from VLM (no choices)"}
        except Exception as e:
            print(f"  [VLM 调用失败] {e}")
            return {"box_visible": False, "error": f"API call failed: {e}"}

        parsed = self._parse_response(raw)
        print(f"  [VLM 解析结果] {parsed}")
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """从 VLM 返回文本中提取 JSON。"""
        # 去除可能的 markdown 代码块标记
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"box_visible": False, "error": f"No JSON found in: {raw[:200]}"}

        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"box_visible": False, "error": f"JSON parse error: {e}"}

        move_x_ms = float(data.get("move_x_ms", 0))
        move_y_ms = float(data.get("move_y_ms", 0))

        return {
            "box_visible": bool(data.get("box_visible", False)),
            "box_in_frame": str(data.get("box_in_frame", "complete")),
            "move_x_ms": move_x_ms,
            "move_y_ms": move_y_ms,
            "confidence": str(data.get("confidence", "low")),
            "description": str(data.get("description", "")),
        }
