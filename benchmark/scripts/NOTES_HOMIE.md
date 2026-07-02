# HOMIE benchmark harness — 运行笔记

harness: `bench_homie.py`(同目录, 单文件, 无 GUI 依赖)
checkpoint: `OpenHomie/HomieDeploy/deploy.onnx`(repo 自带, 456→12, onnxruntime CPU)
策略管线逐行复刻 `OpenHomie/MujocoDeploy/mujoco_deploy_g1.py:59-183`(obs 76×6 历史、
PD 500Hz/policy 50Hz、action×0.25+default), torch.jit→onnxruntime, viewer→mujoco.Renderer。

## 运行机器

4090 服务器(`ssh 4090`, Ubuntu 22.04, python3.10, /sda 可写)。**纯 CPU 即可**
(deploy.onnx 是小 MLP); 渲染走 `MUJOCO_GL=egl`(4090 卡)。
本地(mac, mujoco 3.9.0 + ort 1.26.0)已完整 smoke 通过, 速度参考:
walk 0.5s/trial、squat 4.2s/trial、circle 1.5s/trial(无视频);
带渲染约 0.8×实时(squat 102s 模拟 → 78s 墙钟)。
预估 50-trial 全套(policy 视频策略): walk <2min、squat ~10min、circle ~5min、sweep <2min。

## 环境准备(服务器, 从零)

```bash
ssh 4090
mkdir -p /sda/g1_bench && cd /sda/g1_bench
git clone https://github.com/InternRobotics/OpenHomie.git   # 无 LFS, onnx+mesh 全在 repo
python3.10 -m venv venv_homie
source venv_homie/bin/activate
pip install -U pip
pip install mujoco==3.2.3 numpy==1.26.4 onnxruntime==1.18.1 PyYAML imageio imageio-ffmpeg

# 验证 headless 渲染 + onnx:
MUJOCO_GL=egl python -c "import mujoco; m=mujoco.MjModel.from_xml_path('OpenHomie/HomieRL/legged_gym/resources/robots/g1_description/g1.xml'); r=mujoco.Renderer(m,480,640); print('ok', m.nq, m.nu)"
python -c "import onnxruntime as ort,numpy as np; s=ort.InferenceSession('OpenHomie/HomieDeploy/deploy.onnx',providers=['CPUExecutionProvider']); print(s.run(None,{'input':np.zeros((1,456),np.float32)})[0].shape)"
```

把 harness 拷上去:

```bash
scp /Users/lizhe/Project/sim2real/cc/experiments/scripts/bench_homie.py 4090:/sda/g1_bench/
```

## 启动命令

```bash
cd /sda/g1_bench && source venv_homie/bin/activate
MUJOCO_GL=egl python bench_homie.py --test walk_speed    --trials 50 --out-dir results/homie/walk_speed    --video policy
MUJOCO_GL=egl python bench_homie.py --test squat_box     --trials 50 --out-dir results/homie/squat_box     --video policy
MUJOCO_GL=egl python bench_homie.py --test circle_pillar --trials 50 --out-dir results/homie/circle_pillar --video policy
MUJOCO_GL=egl python bench_homie.py --test speed_sweep   --trials 5  --out-dir results/homie/speed_sweep   --video policy
# 注意: speed_sweep 的 --trials 是"每个速度档"的 trial 数(5 档 × 5 = 25)
# 快速冒烟: --test walk_speed --trials 2 --video none
# repo 位置自动探测(./OpenHomie → /sda/g1_bench/OpenHomie), 也可 --repo 显式指定
```

输出: `results.jsonl`(每 trial 一行) + `summary.json` + `videos/homie_{test}_t{NN}_{ok|fail}.mp4`。

## 本地 smoke 实测结果(mac, mujoco 3.9, 已通过)

- walk_speed 1.0m/s: mean_vx_last8s ≈ 0.965, vx_rmse ≈ 0.086, 无摔 → success
- squat_box(2kg 箱+抱箱位): 20/20 循环, height_rmse ≈ 0.045m, 无摔 → success
- circle_pillar: 无摔无碰撞, 但 **wz 跟踪偏弱** → 实际转弯半径 ~1.2-1.5m,
  radial_err_mean ≈ 0.24(ccw)/0.47(cw) > 0.15, 预算内只闭合 1 圈 → 按规范判 FAIL
  (这是策略本身的结果, 不是 harness bug; cw 比 ccw 差, 存在不对称)
- speed_sweep: 0.6-1.2 全过(1.2 训练上界也过); **0.4 档欠速**(实际 0.34 <
  0.9×0.4=0.36)按比例判据 FAIL

## 已知风险点 / TODO(verify-on-server)

1. **ang_vel_scale 三处不一致(0.25 vs 0.5)— 已做 A/B, 默认 0.25**。
   本地实测: 0.25(MujocoDeploy g1.yaml 值)wz 跟踪好一倍
   (circle radial_err_mean 0.23 vs 0.67), walk/squat 两档无差别。
   服务器 mujoco 3.2.3 上建议用 `--ang-vel-scale 0.5` 跑 2 个 circle trial 复核一次。
2. **EGL 渲染未在服务器验证**(本地走的 mac cgl)。失败时 harness 会警告并继续跑
   (只丢视频不丢数据)。
3. **视频采用"失败/前5成功 trial 确定性重跑"策略**(不录的 trial 零渲染开销)。
   本地重跑结果与首跑完全一致(无 divergence 警告); 服务器上若出现
   "video re-run diverged" 警告需排查(onnxruntime 线程已固定 1)。
4. **箱子是 weld 子 body 而非 geom**: torso_link 有显式 `<inertial>`, 直接塞 geom
   不会增加质量(MuJoCo 规则)。已验证 carried_box mass=2.0、父体 torso_link。
   箱子关了碰撞(规范: 不模拟抓取接触), 纯质量+视觉。
5. **g1.xml 有两段 `<worldbody>`**, 柱子锚定在 floor geom 后注入(不是 `</worldbody>`)。
6. 初始 base 高 0.78 + 关节 ±0.02 噪声 + 2s 零指令静置(策略激活, height=0.74):
   本地无 settle 期摔倒; 3.2.3 上若 settle 摔倒(fall_phase="settle")需查初始落地。
7. T3 按规范的 success 判据(径向误差≤0.15)预计 0% 成功率 — 如实记录即可;
   指标里有 wz_rmse / lap 闭环误差可用于跨框架对比。
8. speed_sweep 成功判据是按比例推广的(mean_vx ≥ 0.9×cmd_vx, 规范只定义了 1.0 档);
   低速档(0.4)对此判据敏感(欠速 ~15%)。跨框架对比时保持同一定义。
9. obs clip ±100(训练有、官方 deploy 脚本漏了, 已补)、action clip ±100(实机配对);
   obs 里的上身默认角恒为训练默认(零), 与抱箱目标位无关 — 勿改。
10. height 命令是绝对 base 高度: 站立 0.74、深蹲发 0.45, 不是相对量。
11. mujoco 版本: 服务器按官方 README 钉 3.2.3, 本地 smoke 用的 3.9.0 —
    两版本物理可能有微小数值差, 指标趋势不应变。
12. License CC BY-NC-SA(禁商用), 内部评测 OK, 产出物勿商用。

## v4 新增测试(T7 squat_limit / T8 squat_place_psi0)

```bash
MUJOCO_GL=egl python bench_homie.py --test squat_limit --trials 10 --out-dir results/homie/squat_limit --video all
MUJOCO_GL=egl python bench_homie.py --test squat_place_psi0 --trials 25 \
    --upper-replay upper_replay_homie27_real_ep053.npz \
    --out-dir results/homie/squat_place_psi0 --video policy
```

- **T7 squat_limit**: 捧 2kg v2 箱, height 指令从 0.74 以 0.05 m/s 匀速连续降到
  0.10 并保持 3s(不起身)。**height 不 clip**: HOMIE height 为绝对语义, 训练域
  [0.24, 0.74]; 本 harness 的命令层只在 psi0 回放路径有 clip(PSI0_HEIGHT_CLIP),
  squat_limit 直接下发原始 ramp 到 0.10, 即放开 0.24 下界(规范 v4 要求, 标定
  跟踪饱和点/物理下蹲极限)。注意 h_cmd 低于 ~0.3 后 "low_height" 摔倒判据
  (base_z < h_cmd - 0.2)几乎无法触发, 深蹲塌倒主要靠 tilt / ground_contact
  捕获。建议 --video all(每个 trial 都是标定样本)。
  本地 smoke(mac, mujoco 3.9, 合成前 2 trial): 无摔, track_sat_h≈0.16,
  depth_floor≈0.201(低于训练下界 0.24!), max_drift_xy≈0.034 — "不摔但饱和"。
- **T8 squat_place_psi0**: real_ep053(双手持物下蹲放置, 22.3s)单次回放驱动上身
  (走 psi0 直接 PD 通路), 下身跟随其 height_cmd(0.77→0.46→0.75, clip 到
  [0.24,0.74] — 只影响 0.77 顶端)。箱子是 **2kg v2 箱**(非 cracker box), t=0
  wrist-weld 抱持; t_place=argmin(height_cmd)(加载 npz 时算好)时刻用
  bit-2+body 位释放机制放箱。box_land_dx 标尺 = 释放时足前缘沿航向位置
  (ankle_roll 原点 + 0.125 m 脚尖, 取自 g1.xml 前脚掌碰撞球 x=0.12+r=0.005),
  正值 = 箱落点(箱心)在身前。box_place_ok := 静止(<0.05 m/s) ∧ 直立(<30°)
  ∧ 落地(z<0.30) ∧ dx>0; success := 无摔 ∧ box_place_ok。
  分段 height_rmse 的 descent/hold/rise 由回放 height 曲线切分
  (hold = min+2cm 连续区段)。
  本地 smoke 用合成 ep053 npz 验证: 弱前伸位 dx≈-0.03(判 FAIL, 合理 — 箱心
  落在脚尖线后), 加大前伸(肩 pitch -0.3/肘 0.25)dx≈+0.125 → place_ok →
  success; 释放/落箱/起身/站稳全链路 OK。真 ep053 臂幅 ±1.3 rad, 预期 dx 更大。
- 回归: walk_speed / squat_box(v2 基线)/ pipeline_abc trial 0 与既有结果
  一致(数值差 ~1e-6, 来自本地临时 venv 的 numpy 版本差), 零回归。

## v5 新增测试(T9 squat_pick_ground / T10 vln_follow)+ 交互渲染 flags

```bash
MUJOCO_GL=egl python bench_homie.py --test squat_pick_ground --trials 15 \
    --out-dir results/homie/squat_pick_ground --video policy
MUJOCO_GL=egl python bench_homie.py --test vln_follow --trials 10 \
    --tapes vln_tapes.json --out-dir results/homie/vln_follow --video policy
# 交互渲染(控制台后端):
python bench_homie.py --test squat_sweep --custom-height 0.40 --custom-rate 0.3 ...
python bench_homie.py --test walk_speed --custom-vx 0.5 ...        # 判据 0.9×V
python bench_homie.py --test circle_pillar --custom-vx 0.3 --custom-wz 0.6 ...  # 半径 V/W
```

- **T9 squat_pick_ground**: 2kg v2 尺寸箱立放地面(中心高 0.125, bit-2 碰撞:
  碰地不碰机器人), 机器人正前方 0.45 m(随 seed 随机朝向放在航向上)。
  低位前伸捧位 **FK 标定角度(NOTES 必记)**: 肩 pitch −0.7, 肩 roll ±0.20,
  肘 1.2, 腕 0(PICK_ARM_POSE) → 腕 FK ≈ (前 0.30, 侧 ±0.183, base_z+0.06~0.07);
  base_z=0.25 时腕到箱面 0.075 m, 0.30 时 0.117 m, 远小于 0.30 抓取阈值。
  时序: 2s 静置 → 1s 摆臂(线性混合到捧位)→ 0.2 m/s 降到 0.25(不 clip,
  域内)→ 蹲底等待, 双腕距箱面均<0.30 m 即原位激活双 weld(同 psi0 磁性
  抓取)→ hold 0.5s → 0.2 m/s 起身(带 2kg)→ 站 1s; 蹲底 5s 无抓取 →
  pick_failed 仍起身。**box_kept 的参考点对 T9 是双腕中点**(非 torso):
  深蹲低位持箱时箱心-torso 距离合法地到 ~0.61 m, 会误触 0.6 阈值;
  腕中点距离 ~0.32 m 稳定(weld 完好性度量, 阈值同 0.6)。
  本地 smoke(mac, mujoco 3.9, 3 trial): 全 OK — pick_success 3/3,
  t_grasp=5.46s(蹲底首 tick 即抓到), grasp_dist≈0.14/0.15,
  **min_root_z≈0.205~0.21(与 R4 标定的 HOMIE 蹲深 ~0.20 一致)**,
  root_drift_bottom≈0.02, 带载起身无摔, stand_ok 全过。HOMIE 确实是
  四模型中此测试最有戏的(AGILE ~0.53 预期够不着, 如实记录即结论)。
- **T10 vln_follow**: --tapes 必填(全模型共用同一 tapes.json 保证公平)。
  HOMIE 语义: wz 为原生 yaw 角速度通道直接下发(不同于 AMO 的 target_yaw
  积分), 已知欠跟踪 ~×0.6-0.8, CAL/T5 同款 — 如实测量。误差全部在
  tape 起始帧(settle 结束时刻机器人位姿)下对比 ref 理想积分轨迹。
  指标: final_pos_err/final_yaw_err_rad、mean_track_err(1Hz 逐秒)、
  small_cmd_response(|vx|∈[0.05,0.15] 段 实际/指令 比值均值)、
  stop_settle(全停段跳过 0.5s 减速后的残余 XY 位移均值)。
  本地 smoke: 温和合成 tape(0.3 前进+小指令段+2 全停)33s 全程无摔,
  stop_settle≈0.04 m; **small_cmd_response≈0.09 — HOMIE 小指令死区
  在此直接量化**(与 T5 CAL 结论一致), 预期 final_pos_err 主要由小指令
  段欠行程贡献。注意: 激进随机 tape(±0.35 急翻转+vy/wz 同时跳变)能把
  HOMIE 摔倒(low_height)— 真 tapes 由 make_vln_tapes.py 统一生成,
  遵守规范的更新间隔/限幅统计即可。
- **交互渲染 flags**(只在 HOMIE/AGILE 两份): --custom-height/--custom-rate
  把 squat_sweep 网格换成单点 [H]×[R](--trials = 该点次数);
  --custom-vx 替换 walk_speed 目标速度(判据按 0.9×V 相对化);
  --custom-vx+--custom-wz 替换 circle_pillar 0.4/0.4, **半径=V/W**
  (起点 (V/W,0)、radial_err 均按该半径; 时长预算按 W 推算)。
  不给 flags 行为不变。smoke: H0.40@0.3 OK; vx0.5 → mean 0.441<0.45
  判 FAIL(相对判据生效, 与既往 0.4 档欠速 ~12% 一致); 0.3/0.6 →
  半径 0.5 起步, radial_err_mean 0.164。
- 回归(本次, mac/mujoco 3.9 本地 venv): walk_speed / circle_pillar /
  squat_limit trial 0 与改动前完全一致(仅 wall_time 差), 零回归。
  T9/T10/customs 均为新增路径, 不触及既有命令流。
