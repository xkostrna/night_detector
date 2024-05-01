import supervision as sv

frames_generator = sv.get_video_frames_generator('nyc_night.mp4')
fps_monitor = sv.FPSMonitor()

for frame in frames_generator:
    fps_monitor.tick()
    fps = fps_monitor()

print(f'Average fps: {fps}')
