# 重新绘制图像：使用连续的线条，并将颜色映射到线段上

from matplotlib.collections import LineCollection


fig, ax = plt.subplots(figsize=(10, 6))

for robot_id in df['robot_id'].unique():
    robot_data = df[df['robot_id'] == robot_id]
    points = np.array([robot_data['x'], robot_data['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speeds = robot_data['speed'].values
    # 为 segments 对应的颜色（长度比 segments 少1）
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(speeds.min(), speeds.max()))
    lc.set_array(speeds[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)

cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('Speed')

ax.set_title('Robot Trajectories Colored by Speed (Continuous Lines)')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend([f'Robot {i}' for i in df['robot_id'].unique()])
ax.grid(True)

plt.tight_layout()
plt.show()
