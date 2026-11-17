# 来源与打包修改

本包由用户提供的 piper-review.zip、piper-source-review.zip 和 piper-resources-review.zip 整理。

## 保留

- src/piper_sim/ 中的 49 个模块、已有控制与感知实现原样保留。
- 原有 62 个 Python 脚本中，除 setup_robot.py 的资源准备入口外均原样保留。
- 49 个测试文件原样保留；原始 VALIDATION.json 不改写。
- 原 MIT LICENSE 与 PiPER、YCB 的来源说明原样保留。
- 上游文档和旧启动器移至 docs/history/、docs/legacy-launchers/，仅用于追溯。

## 为可迁移上传而修改

- 排除 __pycache__、.pyc、上游 .git（其中含未完成的临时 pack 和锁文件）、模型权重与本机运行程序。
- 将机器人 XML 中的 Windows 绝对路径改为资源模板中的仓库路径；scripts/prepare_assets.py 在本机生成绝对路径场景。几何、关节、执行器、相机和物理参数不变。
- 新增独立虚拟环境启动入口，不依赖原工作区 ../../work/runtime。
- pyproject.toml 改为本地 Python 3.10 路线对应的依赖；原 Python 3.12 元数据保存在 docs/history/pyproject.toml。
- setup_robot.py 不再对嵌套目录 git fetch/checkout，避免移除嵌套 .git 后误操作上层成果仓库。
- 没有改写抓取算法、时间/边界配置或历史成绩。打包整理与文档由 AI 辅助完成。

## 验证范围

执行的打包检查见 PACKAGE_CHECK.json：Python 语法、资源存在性、模型参数结构保持、可迁移路径生成、许可证存在和缓存排除。
未在此环境重新运行 MuJoCo 控制、Windows PowerShell、语言模型或完整测试套件。原电脑需要完成启动检查；不能把打包检查当成实机或任务成功率验证。
