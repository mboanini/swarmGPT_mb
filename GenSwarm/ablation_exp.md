# 消融实验说明

该文档用于说明如何进行后续的消融实验，用于明确实验内容

1. 实验任务：cross，偶数个机器人均匀分布在一个圆形上，穿越一片障碍物后到达对面的位置。

2. 基本环境配置

   1. [ ] ROS调整为dt=0.01， 100Hz的频率发布速度

3. 基本框架组成：

   - [ ] 约束池 analyze constraints
   - [ ] 函数分析 analyze request
   - [ ] 函数设计 design functions

   - [ ] 多个函数的layer、parallel生成方式 write functions

   - [ ] 自代码检查 functions review
   - [ ] 错误反馈 bug

   - [ ] 视觉反馈 video critic

   - [ ] 人工反馈 human feedback

4. 指标说明

   碰撞的判定标准为：当两个机器人的距离为5cm的交接时，认定为碰撞。

   到达目标点的判定标准：如果机器人距离目标点的距离在容忍度=0.1m内，则认为机器人到达了目标点

   - [ ] "collision_frequency": 公式为

   - ```
     collision_count / (num_target_entities * num_timesteps)
     ```

     表示所有机器人的总步长中，有多少步是发生碰撞的。

     范围在0~1之间，越大说明避碰效果越差

   - [ ] "collision_severity_sum": 公式为

   - ```
     overlap = info1["size"] + info2["size"] - distance
     collision_severity_sum += overlap / (info1["size"] + info2["size"])
     collision_severity_mean = collision_severity_sum / collision_count
     ```

     表示两个圆形机器人的在发生碰撞后，碰撞的程度如何，通过计算两个圆形机器人的相交的程度来计算

   - [ ] distance_ratio_average: 公式为

   - ```
     distance_ratio = final_distance / initial_distance
     total_distance_ratio += distance_ratio
     average_distance_ratio = total_distance_ratio / num_targets
     ```

     通过计算机器人最终时刻距离目标和最初距离目标的距离的比值，来说明到达目标点的程度，最后对所有机器人求平均，得到本算法的路径分配效率。

     范围一般是在0~1之间，越大说明机器人距离目标点越远。

   - [ ] target_achievement_ratio: 公式为

   - ```
     if final_distance <= tolerance:
         achieved_targets += 1
     target_achieve_ratio = achieved_targets / num_targets
     ```

     如果机器人的最终时刻距离目标距离小于容忍度，则认为机器人到达了目标点。

     范围在0~1之间，数值越大则说明到达目标点的机器人越多

   - [ ] steps_ratio_average: 公式为

   - ```
     if final_distance <= tolerance:
         total_steps_ratio += steps_to_target / len(info["trajectory"])
     average_steps_ratio = total_steps_ratio / achieved_targets
     ```

     统计到达目标点的机器人所花费的平均步数。

     数值大于0，数值越大说明机器人到达目标点花费的步数越多，则算法效率越低

***

### 进行方式

由于human feedback特殊，先不在自动测试中加入。

0. 检查cross任务的提示词**（cmy）**

1. 先自动化测试以下环节的消融对结果的影响，每次只去除其中一个部分**（jwk）**

   > - [ ] **约束池 analyze constraints （由于该部分较为特殊，需要单独写一个分支）**
   > - [ ] 函数分析 analyze request (不做消融)
   > - [ ] 函数设计 design functions (不做消融)
   >
   > - [ ] 多个函数的layer、parallel生成方式 write functions (已做)
   >
   > - [ ] 自代码检查 functions review (要做消融，代码更改不大，只要修改一下generate functions 部分的action即可)
   > - [ ] 错误反馈 bug （不做消融）
   >
   > - [ ] 视觉反馈 video critic (和Human feedback一起做)

2. 统计失败的类型，然后每种找5个例子，用一个固定的反馈意见对这5个进行human feedback，然后评估反馈后的效果**（cmy）**
