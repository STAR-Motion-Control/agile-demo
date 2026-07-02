# G1 Benchmark 实验控制台（4090 常驻）

静态站 + 渲染 API 一体，由纯标准库 `server.py` 同源服务（无 CORS 问题，无 pip 依赖）。
站点根：`4090:/sda/lizhe/g1bench/site/`，端口 **8017**。

## 访问方式

- 校园网可达 4090 内网时：直接开 `http://<4090内网IP>:8017`
  （当前内网 IP：`10.200.7.6`，以 `ssh 4090 hostname -I` 第一项为准）。
- 不可达时走 SSH 隧道（mac 上跑）：

  ```bash
  ssh -L 8017:127.0.0.1:8017 4090
  # 然后浏览器开 http://localhost:8017
  ```

## 重启 / 状态

```bash
ssh 4090
pkill -f 'server.py --port 8017' || true        # 停旧实例（只匹配本服务）
cd /sda/lizhe/g1bench/console
nohup python3 server.py --port 8017 --root /sda/lizhe/g1bench/site \
    > server.log 2>&1 &
curl -s 127.0.0.1:8017/api/health               # {"ok": true, ...}
tail -f server.log
```

任务队列为单工作线程串行执行，单任务超时 600 s（`server.py` 顶部 `CONFIG` 可改，
含 harness / conda env / AGILE repo+mjcf 路径）。任务历史持久化在 `site/jobs.json`，
重启不丢；渲染产物在 `site/videos/custom/`，工作目录与日志在 `site/jobs/<job_id>/`。

## 新增实验结果同步流程（无需重启 server）

在 mac 的 `cc/experiments/` 下：

```bash
# 1) 重新生成 manifest（site 模式：视频路径 videos/<model>/<file>）
python3 console/make_manifest.py --site

# 2) 同步新视频 + 控制台文件到 4090
rsync -a --progress results/videos_r2plus/ 4090:/sda/lizhe/g1bench/site/videos/
rsync -a console/index.html console/manifest.json console/server.py \
    console/build_site.sh PROTOCOL.md 4090:/sda/lizhe/g1bench/console/

# 3) 4090 上组装站点（拷 index/manifest 到 site/ 并校验视频齐全）
ssh 4090 'bash /sda/lizhe/g1bench/console/build_site.sh'
```

刷新浏览器即可——manifest.json/index.html 都带 `no-cache`，server 不用动。
（若改了 `server.py` 本身才需要按上面的重启流程重启。）

## 渲染 API（前端"交互渲染台"即调它）

```bash
# 提交（参数范围: height∈[0.2,0.7] rate∈[0.05,0.8] vx∈[0.2,1.2] wz∈[0.1,0.6]）
curl -s -X POST http://127.0.0.1:8017/api/render -H 'Content-Type: application/json' \
  -d '{"model":"homie","kind":"squat","height":0.4,"rate":0.2}'
# => {"job_id": "j20260612_...", "status": "queued"}

curl -s http://127.0.0.1:8017/api/job/<job_id>   # 轮询: queued/running/done/failed + log_tail
curl -s http://127.0.0.1:8017/api/jobs           # 全部任务，最新在前
```

kind → harness 调用：`squat` = `--test squat_sweep --custom-height H --custom-rate R`；
`walk` = `--test walk_speed --custom-vx V`；`circle` = `--test circle_pillar --custom-vx V --custom-wz W`
（均 `--trials 1 --video all`，跑完 `caption_videos.py` 烧字幕落 `videos/custom/`）。
`--custom-*` 参数由 `scripts/bench_homie.py` / `bench_agile.py` 新增支持，4090 与本地已同步。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `console/server.py` | 静态服务（带 Range，Safari 可播）+ `/api/render` `/api/job` `/api/jobs`，纯 stdlib |
| `console/index.html` | 前端单页：目标/流程/记录表/视频库/交互渲染台，全相对路径，无 CDN |
| `console/make_manifest.py` | 扫 results/ 生成 manifest.json（`--site` 输出站点路径） |
| `console/build_site.sh` | 4090 上组装 site/（拷文件 + 归一化 manifest 路径 + 校验视频） |
| `scripts/bench_{homie,agile}.py` | 新增 `--custom-height/--custom-rate/--custom-vx/--custom-wz` 单点渲染 |
| `scripts/caption_videos.py` | 字幕烧录；walk/circle 字幕改为读 metrics 实际指令值 |
