> 本次独立训练环境请先阅读 [START_HERE.md](START_HERE.md)。以下为保留的上游 demo 说明。

# PiPER Interactive Demo

输入一句话，点击 **执行指令**：腕部相机观察 → 本地千问理解 → RGB-D 定位 → IK 抓取与搬运 → 放置 → 仿真结果检查。

操作台提供三个独立演示入口，切换入口会生成对应的随机场景：

| 入口 | 自动执行内容 | 开发回合限时 |
|---|---|---|
| L1 方块分类 | CPU 颜色识别，连续抓放红、绿、蓝三个方块至同色托盘 | 180 秒仿真时间 |
| L2 果蔬分拣 | 从八类扫描果蔬中随机取六类；千问看腕部图像，逐个选物并按配置分类投放 | 300 秒仿真时间 |
| L3 自然语言 | 输入自己的指令，完成一次视觉选物与抓放 | 单次交互预算；不是官方五指令回合 |

主界面在常见桌面分辨率下一屏显示双相机、任务进度、结果和输入框，详细日志与模型设置在弹窗中。**随机重置**不需要输入种子；详情保留布局编号用于复现。L3 失败或停止后可以直接提交新指令，先从当前姿态重新观察，保留物体和机械臂状态。L1/L2 的单个物体失败后继续后续尝试；整轮结束后仍可继续尝试，只有点击重置或切换入口才重新生成场景。

分类映射来自 `config/competition/scene.yaml` 的 `tasks.l1/l2.mapping`，是演示预设，**不是官方已确认规则**。L1/L2 按独立仿真评估器记录不同物体的分类与稳定放置结果；不伪造完成数或官方分数。L2 默认最多八次抓放尝试，尚不保证所有随机布局都能全部完成。

这是在本地运行的 MuJoCo 交互 demo，GitHub 托管源码。不是连接真机的控制程序，也不是完整比赛系统。

## 快速开始（Ubuntu / Python 3.12）

```bash
git clone https://github.com/shiyilong11/piper-interactive-demo.git
cd piper-interactive-demo
sudo apt-get install libosmesa6 git python3.12-venv
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/setup_robot.py
bash scripts/run_interactive_demo.sh
```

打开 **http://127.0.0.1:8770/**。页面打开时不会执行任何动作。

没有 GPU 时，先将“理解方式”改为 **CPU 规则 · 颜色方块**，输入 `把红色方块放到蓝色托盘`，点击执行。该模式只接受明确的颜色方块指令，不会调用或冒充千问。

支持暂停、继续、停止、重置布局和释放模型显存。停止冻结当前仿真；失败或停止后先重置。模型推理正在进行时，暂停和停止会在返回后阻止后续动作，不会终止正在运行的 GPU kernel。

## 从另一台电脑通过外网访问

支持 HTTPS 隧道和访问口令，计算仍在 GPU 主机运行。见 [远程访问部署说明](docs/remote-access.md)。

## 本地千问（可选）

另建模型环境，避免影响原有训练环境：

```bash
python3.12 -m venv .local/qwen-runtime
.local/qwen-runtime/bin/pip install -e .
.local/qwen-runtime/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.local/qwen-runtime/bin/pip install -r requirements-model.txt
.local/qwen-runtime/bin/python scripts/download_model.py
bash scripts/run_interactive_demo.sh
```

模型：`Qwen/Qwen3-VL-4B-Instruct`，固定 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。BF16 权重约 8.9 GB，不在仓库中。下载需联网，运行使用本地文件，不调用云 API。模型在第一次执行千问指令时加载；启动空闲显存门槛为 12 GiB，可在 `config/vlm/qwen.yaml` 修改。实测 RTX 5090 单次请求约 8.5 GiB PyTorch 显存，不代表所有配置的上限。其他显卡需匹配 PyTorch/CUDA/驱动。

默认使用 `.local/qwen-runtime/bin/python`，可通过 `QWEN_PYTHON` 指向已有模型环境；`DEMO_PYTHON` 可指定仿真环境。模型与仿真分进程，仿真采用 CPU OSMesa 渲染。

## 输入、输出与能力边界

| 阶段 | 输入 | 输出 |
|---|---|---|
| 指令理解 | 用户文本 + 当前腕部 RGB | 状态、目标二维点、目标托盘颜色 |
| 几何定位 | 目标点 + RGB-D + 内参 + 手眼外参 + 桌面高度 | 可见目标连通区域、三维中心估计、尺寸与方向 |
| 抓放 | 目标几何 + 已配置托盘坐标 + 关节状态 | 限速关节目标与夹爪开口 |
| 评估 | MuJoCo 接触、物体位置与速度 | 抓起持续时间、入托盘并稳定放置结果 |

- **千问只看腕部 RGB**。全景用于人类监看，不送给模型。目标三维位置由深度计算，不读取仿真物体真值来规划。
- 托盘坐标、桌面高度和拾取区域是已知配置；不是通过有限画幅重建整个桌面。左右按当前图像解释，尚不支持全桌面“最左边”或跨视角关系推理。
- 每个低层技能完成一个物体的抓放；L1/L2 将它串成连续演示。千问输出点可按深度区域做最多 18 px 的邻近修正；目标被截断时，用可见深度区域和机械臂运动学主动调整腕部相机，让目标居中，再重新拍照、识别和定位。主动调整最多三次，无法改善时切换预设观察位；用尽视角仍无法确定时结束本次尝试。深度缺失仍优先退远观察，抓空进行有界重试。
- 低层技能是传统 RGB-D + IK + 位置控制，不是训练好的 BC/RL 策略，也不是端到端 VLA。当前网页任务按真实观测逐段恢复：直接路径未通过先搬至前方灵活区，放下后再计算下一段；必要时采用向目标推进的长短段。没有第二段预测否决或固定准备次数放弃，按实际时间/安全边界收尾。路径有有限笛卡尔和关节空间绕行，不能证明全局不可达。
- 方块完成过真实接触抓放验证；水果为 YCB 扫描外观和近似碰撞形状，抓取仍为实验性。相邻物体连在一起、遮挡、细长/不规则物体和新布局均可能失败。
- 物体没有焊接到夹爪或按轨迹瞬移。仿真真值只供独立评估使用。L3 显示“成功”表示**所选物体**完成物理抓放，不证明模型理解符合原指令；请核对标注图和原始输出。L1/L2 另外按配置映射核对实际类别和托盘。
- 不是官方计分器。原有 L1 默认颜色映射分数与用户任意指定托盘可以不同；界面按本次所选目标和目的托盘评估。

## 可修改配置与扩展接口

| 文件 | 内容 |
|---|---|
| `config/eye_in_hand.yaml` | 相机内参、深度、`hand_eye.T_end_camera` 光学坐标→link6，平移单位米；真实标定后替换 |
| `config/vlm/wrist_camera.yaml` | 观察关节姿态、引用的手眼和模型配置 |
| `config/competition/scene.yaml` | 桌面、托盘、任务物体、尺寸颜色变化、物理和开发判定阈值 |
| `config/competition/objects_scanned.yaml` | 物体资产、质量和碰撞形状 |
| `config/interface.yaml` | 20 Hz 动作接口、500 Hz 物理、关节限位与速度、夹爪开口 |
| `config/interactive_demo.yaml` | 端口、默认场景、默认布局、观察搜索姿态、`active_view` 主动居中参数、抓空重试次数、渲染频率与时间预算 |
| `config/vlm/qwen.yaml` | 模型路径、显存门槛、图像和生成 token 限制 |

`src/piper_sim/skill_runtime.py` 提供九个技能及调度器，每个技能返回 `SkillResult(status, data, code, message, summary)`，界面记录调用顺序和输出。接口详见 [技能说明](docs/skills.md)。

`demo_skill.py` 中 `TargetEstimate` 为抓取目标接口，`PickPlaceSkill.run(target, destination, tray_floor)` 为抓放接口。可在保留输入和评估的情况下替换为学习策略。`scripts/interactive_demo.py` 编排流程，`qwen_camera_worker.py` 用 JSON-lines 隔离模型，`assets/ui/interactive_demo.html` 为无构建步骤的页面。

目前 demo 拾取区域的 RGB-D 阈值在 `candidates_from_rgbd` 中；方块布局范围在 `Demo.__init__` 中，果蔬布局范围在 `config/interactive_demo.yaml` 的 `demos.fruit_spawn` 中。改变场地时应同步检查。手眼参数变化后重新运行 `scripts/build_piper_model.py`。

## 验证

```bash
.venv/bin/pytest -q
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  .venv/bin/python scripts/check_interactive_skill.py
```

第二条执行 CPU 颜色识别 → 深度定位 → 物理抓放，结果写入 `artifacts/interactive_demo/check/result.json`，不加载模型。界面为每次执行保存独立 runs/*.json（含失败和中断）、腕部 RGB-D、相机位姿与内参、模型原始响应和技能轨迹。目的地未指定时先询问托盘颜色；目标深度无效时优先切换到配置中的退远观察位。固定 seed 42 的红方块→蓝托盘测试约 21.2 秒仿真时间；蓝方块→红托盘通过“首次不确定 → 换视角 → 重新识别 → 抓放 → 验证”，约 23.05 秒仿真时间；真实墙钟耗时受渲染、推理和交互暂停影响，不能当作实机成绩。

## 资产与许可证

项目代码使用 MIT。PiPER 模型来自官方 [piper_ros](https://github.com/agilexrobotics/piper_ros)，固定 commit `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`，安装脚本下载并转换，保留 [MIT 声明](licenses/piper_ros-MIT.txt)。

水果来自 YCB，采用 CC BY 4.0，作者、来源、修改说明见 [NOTICE](assets/competition/ycb/NOTICE.md)。这些不是官方比赛物料。Qwen 权重不随仓库分发，其许可证以官方模型仓库为准。仓库不含训练数据、个人文档、模型权重或本机运行日志。
