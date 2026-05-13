# 自动运行生成的代码，以及统计分析

1. 文件名: experiment_runner.py
2. 参数：
    - env_config_path: 环境配置文件路径
    - workspace_path: 需要运行的代码路径，例如 layer/cross 表示运行layer/cross 文件夹下的所有的代码
    - experiment_duration: 实验持续时间，单个代码超时则终止
    - run_mode: 运行模式，Options: ['rerun','continue'，'analyze'], 'rerun'表示重新运行，'continue'表示继续上次没有运行完的继续运行，'analyze'表示分析结果,不运行
    - max_speed: 最大速度，用于调节速度，默认的速度是0.2，可以通过这个参数调节速度，manager会根据这个速度来调节小车的速度(归一化+*max_speed)。
    - tolerance: 容忍度，用于判断是否到达目标位置 或者相互重叠多少为碰撞，默认是0.05。
3. 输出：
   - 代码的运行过程数据，以及运行结果(是否到达目标位置，是否发生碰撞) 保存在workspace_path下的result.json 中
   - 分析阶段会读取result.json文件，生成统计数据，绘图保存在workspace_path下
4. 注意事项：
   - 由于需要检测碰撞，所以在物理引擎处要屏蔽碰撞
   - 目前只支持cross一个任务的自动检测，自动判断是否到达目标位置，是否发生碰撞
   - 可能存在进程没有完全退出的情况，需要手动kill掉，具体操作如下：
     - top或者htop查看是否有CPU占用异常
     - 查看进程指令：ps aux | grep run.py
     - kill 指令：pgrep -f run.py | xargs kill -9
