import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd

def analyze_video(video_path, width_cm, height_cm, interval, display_joints):

    def find_contours(mask):
        contours_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        return contours
    
        #从视频第一帧读取关节颜色值
    def get_hsv_from_frames(cap):
        hsv_colors = {joint: [] for joint in ["hip", "knee", "ankle", "foot"]}
        
        # 读取视频的第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            return hsv_colors

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for joint in hsv_colors.keys():
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Select {joint} point")
            points = plt.ginput(1)
            x, y = int(points[0][0]), int(points[0][1])
            hsv = hsv_frame[y, x]
            hsv_colors[joint] = hsv
            plt.close()
        return hsv_colors
    
        #计算跑步带速度平移
    def adjust_positions(positions, speed, fps):
        adjusted_positions = []
        displacement_per_frame = speed / (fps*8)   #iphone 慢动作是240帧，输出视频是30帧，这里fps需要*8
        for i, (x, y) in enumerate(positions):
            adjusted_x = x - i * displacement_per_frame
            adjusted_positions.append((adjusted_x, y))
        return np.array(adjusted_positions)
    
    def smooth_positions(positions, window_size=10):
        smoothed_positions = np.convolve(positions, np.ones(window_size)/window_size, mode='valid')
        return smoothed_positions

    def calculate_velocity(positions):
        velocity = np.diff(positions, axis=0)
        return velocity

    # 用于确认选择的函数
    def confirm_selection(event):
        if event.key == 'enter':
        # 在控制台打印选择的区域
           print("Selected regions:")
           for i, region in enumerate(selected_regions):
               print(f"Region {i+1}: ({region[0]}, {region[1]})")

    # 定义选区时的回调函数
    def onselect(vmin, vmax):
        selected_regions.append((vmin, vmax))
        print(f"Selected region: ({vmin}, {vmax})")
        # 更新图形，将选中区域标记出来
        for region in selected_regions:
          ax1.axvspan(region[0], region[1], color='red', alpha=0.3)
        fig.canvas.draw_idle()
    
    def update_phases_based_on_selection(phases, selected_regions):
        for vmin, vmax in selected_regions:
           for i in range(int(vmin), int(vmax)):
              phases[i] = 'Swing'
        return phases

    # 加载视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    hsv_colors = get_hsv_from_frames(cap)

    # 获取视频的帧率和大小
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {width}x{height} at {fps} fps")

    # 定义合理的HSV范围
    green_lower = np.array(hsv_colors["hip"]) - np.array([10, 50, 50])
    green_upper = np.array(hsv_colors["hip"]) + np.array([10, 255, 255])
    blue_lower = np.array(hsv_colors["knee"]) - np.array([10, 50, 50])
    blue_upper = np.array(hsv_colors["knee"]) + np.array([10, 255, 255])
    yellow_lower = np.array(hsv_colors["ankle"]) - np.array([10, 50, 50])
    yellow_upper = np.array(hsv_colors["ankle"]) + np.array([10, 255, 255])
    red_lower = np.array(hsv_colors["foot"]) - np.array([10, 50, 50])
    red_upper = np.array(hsv_colors["foot"]) + np.array([10, 255, 255])

    # 初始化列表来存储关节位置数据
    hip_positions = []
    knee_positions = []
    ankle_positions = []
    foot_positions = []

    # 定义输出视频路径
    output_path_with_lines = video_path.replace('.mp4', '_output_with_lines.mp4')
    output_path_lines_only = video_path.replace('.mp4', '_output_lines_only.mp4')

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_with_lines = cv2.VideoWriter(output_path_with_lines, fourcc, fps, (width, height))
    output_video_lines_only = cv2.VideoWriter(output_path_lines_only, fourcc, fps, (width, height))

    # 初始化帧计数器和写入间隔
    frame_count = 0
    write_interval = 2
    processed_frames = 0

    # 定义像素到厘米的转换比例
    pixel_to_cm_x = width_cm / width
    pixel_to_cm_y = height_cm / height
    # 像素转化为厘米
    def convert_to_cm(x, y):
        cm_x = x * pixel_to_cm_x
        cm_y = y * pixel_to_cm_y
        return cm_x, cm_y

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 使用cv2.inRange()函数进行颜色阈值分割
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # 初始化关节坐标
        joint_coords = {"hip": None, "knee": None, "ankle": None, "foot": None}
        
        # 使用一个字典存储颜色信息
        colors_info = {
            "hip": (green_mask, (0, 255, 0)),
            "knee": (blue_mask, (255, 0, 0)),
            "ankle": (yellow_mask, (0, 255, 255)),
            "foot": (red_mask, (0, 0, 255))
        }

        # 遍历每种颜色，找到轮廓并计算中心
        for joint, (mask, color) in colors_info.items():
            contours = find_contours(mask)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                    joint_coords[joint] = convert_to_cm(x, y)
            print(f"{joint} contours: {len(contours)}, coords: {joint_coords[joint]}")

        # 检查是否所有关节都被检测到
        if all(joint_coords.values()):
            hip_positions.append(joint_coords["hip"])
            knee_positions.append(joint_coords["knee"])
            ankle_positions.append(joint_coords["ankle"])
            foot_positions.append(joint_coords["foot"])

            # 只处理每隔一定数量的帧
            if frame_count % write_interval == 0:
                # 在原始帧上绘制关节和连线
                for joint, (cm_x, cm_y) in joint_coords.items():
                    x, y = int(cm_x / pixel_to_cm_x), int(cm_y / pixel_to_cm_y)
                    color = colors_info[joint][1]
                    cv2.circle(frame, (x, y), 5, color, -1)
                hip = joint_coords["hip"]
                knee = joint_coords["knee"]
                ankle = joint_coords["ankle"]
                foot = joint_coords["foot"]
                cv2.line(frame, (int(hip[0] / pixel_to_cm_x), int(hip[1] / pixel_to_cm_y)),
                                (int(knee[0] / pixel_to_cm_x), int(knee[1] / pixel_to_cm_y)), (255, 255, 255), 2)
                cv2.line(frame, (int(knee[0] / pixel_to_cm_x), int(knee[1] / pixel_to_cm_y)),
                                (int(ankle[0] / pixel_to_cm_x), int(ankle[1] / pixel_to_cm_y)), (255, 255, 255), 2)
                cv2.line(frame, (int(ankle[0] / pixel_to_cm_x), int(ankle[1] / pixel_to_cm_y)),
                                (int(foot[0] / pixel_to_cm_x), int(foot[1] / pixel_to_cm_y)), (255, 255, 255), 2)

                # 创建一个空白帧用于绘制关节连线
                blank_frame = np.zeros_like(frame)
                for joint, (cm_x, cm_y) in joint_coords.items():
                    x, y = int(cm_x / pixel_to_cm_x), int(cm_y / pixel_to_cm_y)
                    color = colors_info[joint][1]
                    cv2.circle(blank_frame, (x, y), 5, color, -1)
                cv2.line(blank_frame, (int(hip[0] / pixel_to_cm_x), int(hip[1] / pixel_to_cm_y)),
                                (int(knee[0] / pixel_to_cm_x), int(knee[1] / pixel_to_cm_y)), (255, 255, 255), 2)
                cv2.line(blank_frame, (int(knee[0] / pixel_to_cm_x), int(knee[1] / pixel_to_cm_y)),
                                (int(ankle[0] / pixel_to_cm_x), int(ankle[1] / pixel_to_cm_y)), (255, 255, 255), 2)
                cv2.line(blank_frame, (int(ankle[0] / pixel_to_cm_x), int(ankle[1] / pixel_to_cm_y)),
                                (int(foot[0] / pixel_to_cm_x), int(foot[1] / pixel_to_cm_y)), (255, 255, 255), 2)

                # 将处理后的帧写入视频
                output_video_with_lines.write(frame)
                output_video_lines_only.write(blank_frame)
                processed_frames += 1

        frame_count += 1

    print(f"Total frames processed: {processed_frames}")

    # 释放资源
    cap.release()
    output_video_with_lines.release()
    output_video_lines_only.release()

    # 将关节位置数据转换为NumPy数组
    hip_positions = np.array(hip_positions)
    knee_positions = np.array(knee_positions)
    ankle_positions = np.array(ankle_positions)
    foot_positions = np.array(foot_positions)

    # 输出关节位置的形状
    print("Shapes of joint position arrays:")
    print(f"Hip positions: {hip_positions.shape}")
    print(f"Knee positions: {knee_positions.shape}")
    print(f"Ankle positions: {ankle_positions.shape}")
    print(f"Foot positions: {foot_positions.shape}")

    # 对齐所有关节位置数据的长度
    min_len = min(len(hip_positions), len(knee_positions), len(ankle_positions), len(foot_positions))
    hip_positions = hip_positions[:min_len]
    knee_positions = knee_positions[:min_len]
    ankle_positions = ankle_positions[:min_len]
    foot_positions = foot_positions[:min_len]

    # 获取跑步带的速度
    treadmill_speed = float(input("Enter treadmill speed in cm/s: "))

    # 调整关节位置数据
    if treadmill_speed != 0:
        hip_positions = adjust_positions(hip_positions, treadmill_speed, fps)
        knee_positions = adjust_positions(knee_positions, treadmill_speed, fps)
        ankle_positions = adjust_positions(ankle_positions, treadmill_speed, fps)
        foot_positions = adjust_positions(foot_positions, treadmill_speed, fps)

    # 导出关节位置数据到CSV文件
    data = {
        "Frame": np.arange(min_len),
        "Hip_X_cm": hip_positions[:, 0], "Hip_Y_cm": hip_positions[:, 1],
        "Knee_X_cm": knee_positions[:, 0], "Knee_Y_cm": knee_positions[:, 1],
        "Ankle_X_cm": ankle_positions[:, 0], "Ankle_Y_cm": ankle_positions[:, 1],
        "Foot_X_cm": foot_positions[:, 0], "Foot_Y_cm": foot_positions[:, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(video_path.replace('.mp4', '_joint_positions.csv'), index=False)

    # 计算步幅长度
    # 找到最低点
    lowest_points = [i for i in range(1, len(foot_positions) - 1) if foot_positions[i][1] < foot_positions[i-1][1] and foot_positions[i][1] < foot_positions[i+1][1]]

    strides = []
    for i in range(1, len(lowest_points)):
        stride_length = np.abs(foot_positions[lowest_points[i], 0] - foot_positions[lowest_points[i - 1], 0])
        strides.append(stride_length)
    average_stride_length = np.mean(strides)
    print(f"Average stride length: {average_stride_length:.2f} cm")


    smoothed_foot_positions = smooth_positions(foot_positions[:, 1])

    # 计算速度和加速度
    velocity = np.abs(calculate_velocity(smoothed_foot_positions))

    # 补齐长度以匹配原始数据长度
    velocity = np.concatenate(([0], velocity))

    # 初始所有帧都定义为支撑相
    phases = ['Stance'] * len(smoothed_foot_positions)

     # 创建图形和子图
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.plot(velocity, label='Velocity', color='blue')
    ax1.set_title('Velocity over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Velocity (cm/frame)')
    ax1.legend()
    ax1.grid(True)
    # 存储选择的区域
    selected_regions = []
   
    # 添加SpanSelector
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,rectprops=dict(alpha=0.5, facecolor='red'))

    # 添加按键事件处理程序
    fig.canvas.mpl_connect('key_press_event', confirm_selection)
    plt.show()

    # 更新相位
    phases = update_phases_based_on_selection(phases, selected_regions)

    # 绘制关节运动图
    plt.figure(figsize=(10, 5))
    # 绘制每个时间点的关节连线

    step = interval if interval > 0 else 1
    for i in range(0, len(smoothed_foot_positions), step):
        if 'hip' in display_joints:
            plt.plot(hip_positions[i, 0], hip_positions[i, 1])
        if 'knee' in display_joints:
            plt.plot(knee_positions[i, 0], knee_positions[i, 1])
        if 'ankle' in display_joints:
            plt.plot(ankle_positions[i, 0], ankle_positions[i, 1])
        if 'foot' in display_joints:
            plt.plot(foot_positions[i, 0], foot_positions[i, 1])

        color = 'black' if phases[i] == 'Stance' else 'grey'
        if 'hip' in display_joints and 'knee' in display_joints:
            plt.plot([hip_positions[i, 0], knee_positions[i, 0]], [hip_positions[i, 1], knee_positions[i, 1]], color=color)
        if 'knee' in display_joints and 'ankle' in display_joints:
            plt.plot([knee_positions[i, 0], ankle_positions[i, 0]],[knee_positions[i, 1], ankle_positions[i, 1]], color=color)
        if 'ankle' in display_joints and 'foot' in display_joints:
            plt.plot([ankle_positions[i, 0], foot_positions[i, 0]], [ankle_positions[i, 1], foot_positions[i, 1]], color=color)
    # 设置图表标签和标题
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title('Hind Limb Motion of the Mouse')
    plt.grid(True)
    # Save SVG without axes and grid
    plt.axis('off')  # Turn off axes
    plt.grid(False)  # Turn off grid

    # Save SVG file
    plt.savefig(video_path.replace('.mp4','output.svg'),format='svg')
    plt.show()

if __name__ == "__main__":
    video_path = input("Enter the path to the video:")
    width_cm = float(input("Enter the width of the video in cm:"))
    height_cm = float(input("Enter the height of the video in cm:"))
    interval = int(input("Enter interval vaule(0 for no interval):"))
    display_joints = input("Please enter the joints to be displayed(hip,knee,ankle,foot) split by: ,").split(',')
    analyze_video(video_path, width_cm, height_cm, interval,display_joints)
