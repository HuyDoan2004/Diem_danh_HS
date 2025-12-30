import pyrealsense2 as rs
import numpy as np

class D435i:
    def __init__(self, width=848, height=480, fps=30):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None, None, None
        color_np = np.asanyarray(color.get_data())
        depth_np = np.asanyarray(depth.get_data())  # đơn vị: mm (uint16)
        return color_np, depth_np, depth

    def stop(self):
        self.pipeline.stop()
