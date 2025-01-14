from moviepy.editor import VideoFileClip

# 加载视频文件
video = VideoFileClip("/nas_data/data/Fei/DIP-Teaching/Assignments/04_3DGS/data/chair/checkpoints/debug_rendering.mp4")

# 可以选择剪辑视频（如只转换前10秒的部分）
video = video.subclip(0, 10)  # 从 0 秒到 10 秒

# 将视频转换为 GIF
video.write_gif("./output/chair.gif", fps = 4 )

video = VideoFileClip("/nas_data/data/Fei/DIP-Teaching/Assignments/04_3DGS/data/lego/checkpoints/debug_rendering.mp4")

video = video.subclip(0, 10)  # 从 0 秒到 10 秒

# 将视频转换为 GIF
video.write_gif("./output/lego.gif", fps = 4 )
