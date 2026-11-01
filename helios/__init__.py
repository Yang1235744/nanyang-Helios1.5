"""nanyang-Helios 1.5 —— 中等规模视频/图片生成模型（含音频 + 南阳 GPT 捆绑）。

升级点（相对 1.2 低等版）：
  - 高清图片（最高 1024px）
  - 长视频（默认 15 秒 @ 12fps = 180 帧，时空解耦注意力）
  - 音频生成（AudioVAE + AudioDiT，与视频同步出音轨）
  - 捆绑南阳 GPT（对话大模型 + 工具调用生成图片/视频），形成统一中等模型
"""

__version__ = "1.5.0a0"
MODEL_NAME = "nanyang-Helios1.5"
