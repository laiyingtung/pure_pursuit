import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import matplotlib as mpl
import rospy
from nav_msgs.msg import Path

# 初始化全局變數
PathData = []  # 用來存儲路徑點

# 初始化變數
currentPos = [0, 0]
currentHeading = 90  # 一開始朝向的方位(用度表示)
lastFoundIndex = 0
lookAheadDis = 0.8
linearVel = 150
xs, ys = [], []

# 回調函數來處理接收到的路徑數據
def path_callback(data):
    global PathData
    PathData = []
    for pose_stamped in data.poses:
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        PathData.append([x, y])
    PathData = np.array(PathData)  # 將接收到的數據轉換為 2D 路徑數據
    if len(PathData) > 0:
        print("接收到新的路徑: ", PathData)
    else:
        print("接收到的路徑數據為空")

# pure pursuit 計算
def pt_to_pt_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

def sgn(num):
    return 1 if num >= 0 else -1

def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
    currentX = currentPos[0]
    currentY = currentPos[1]
    lastFoundIndex = LFindex
    goalPt = [0, 0]
    end = False

    for i in range(lastFoundIndex, len(path) - 1):
        if i >= len(path) - 20:
            end = True  # 到達路徑的終點
        x1, y1 = path[i][0] - currentX, path[i][1] - currentY
        x2, y2 = path[i + 1][0] - currentX, path[i + 1][1] - currentY
        dx, dy = x2 - x1, y2 - y1
        dr = math.sqrt(dx ** 2 + dy ** 2)
        D = x1 * y2 - x2 * y1
        discriminant = (lookAheadDis * 2) * (dr * 2) - D ** 2
        if discriminant >= 0:
            sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
            sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
            sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
            sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2

            sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
            minX = min(path[i][0], path[i + 1][0])
            minY = min(path[i][1], path[i + 1][1])
            maxX = max(path[i][0], path[i + 1][0])
            maxY = max(path[i][1], path[i + 1][1])

            if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or \
                    ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and \
                        ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                    if pt_to_pt_distance(sol_pt1, path[i + 1]) < pt_to_pt_distance(sol_pt2, path[i + 1]):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2
                else:
                    if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2

                if pt_to_pt_distance(goalPt, path[i + 1]) < pt_to_pt_distance([currentX, currentY], path[i + 1]):
                    lastFoundIndex = i
                    break
                else:
                    lastFoundIndex = i + 1

    Kp = 3
    absTargetAngle = math.atan2(goalPt[1] - currentY, goalPt[0] - currentX) * 180 / math.pi
    if absTargetAngle < 0:
        absTargetAngle += 360

    turnError = absTargetAngle - currentHeading
    if turnError > 180 or turnError < -180:
        turnError = -1 * sgn(turnError) * (360 - abs(turnError))

    turnVel = Kp * turnError
    return goalPt, lastFoundIndex, turnVel, end

# 動畫
fig, ax = plt.subplots()
ax.set_aspect('equal')
pose, = ax.plot([], [], 'o', color='black', markersize=10)
trajectory_line, = ax.plot([], [], '-', color='orange', linewidth=4)
heading_line, = ax.plot([], [], '-', color='red')
connection_line, = ax.plot([], [], '-', color='green')

ax.set_xlim(-10, 50)
ax.set_ylim(-10, 50)

cycle = 0

def pure_pursuit_animation(frames):
    global currentPos, currentHeading, lastFoundIndex, PathData, xs, ys
    if len(PathData) == 0:
        return  # 如果路徑尚未接收到，什麼也不做
    
    if lastFoundIndex >= len(PathData) - 2:
        lastFoundIndex = 0
    
    goalPt, lastFoundIndex, turnVel, end = pure_pursuit_step(PathData, currentPos, currentHeading, lookAheadDis, lastFoundIndex)
    
    maxLinVelfeet = 200 / 60 * np.pi * 4 / 12
    maxTurnVelDeg = 200 / 60 * np.pi * 4 / 9 * 180 / np.pi

    stepDis = linearVel / 100 * maxLinVelfeet * 50 / 1000
    currentPos[0] += stepDis * np.cos(currentHeading * np.pi / 180)
    currentPos[1] += stepDis * np.sin(currentHeading * np.pi / 180)

    heading_line.set_data([currentPos[0], currentPos[0] + 0.5 * np.cos(currentHeading / 180 * np.pi)], 
                          [currentPos[1], currentPos[1] + 0.5 * np.sin(currentHeading / 180 * np.pi)])
    connection_line.set_data([currentPos[0], goalPt[0]], [currentPos[1], goalPt[1]])

    currentHeading += turnVel / 100 * maxTurnVelDeg * 50 / 1000
    if currentHeading < 0:
        currentHeading += 360
    currentHeading = currentHeading % 360

    xs.append(currentPos[0])
    ys.append(currentPos[1])

    pose.set_data([currentPos[0]], [currentPos[1]])

    trajectory_line.set_data(xs, ys)

    #移除自動退出條件，車輛達到目標後繼續運行
    if end:
        print("Reached end, but continuing...")
        lastFoundIndex = 0
        currentPos = [PathData[0][0], PathData[0][1]]  # 重置到路徑的起點
        xs, ys = [], []  # 清空之前的軌跡

def main():
    rospy.init_node('pure_pursuit_path_follower')
    rospy.Subscriber("/calculated_path", Path, path_callback)
    
    anim = animation.FuncAnimation(fig, pure_pursuit_animation, frames=500, interval=83)
    plt.show()

if __name__ == '__main__':
    main()