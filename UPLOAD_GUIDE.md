# 上传到自己的 GitHub

## 解压

将本 ZIP 解压到一个新目录 piper-competition-sim。不要覆盖原来的 piper-training-sim。
建议放到 Windows Documents\Codex 下，当前公开版启动器使用 Windows PowerShell。

## 检查

README、src、scripts、config、assets、references、tests、LICENSE 应在同一级。
media 目前只有说明，实机视频尚未加入。原有项目、模型、日志仍留在原目录。

## 新建远程空仓库

打开 https://github.com/new ，Repository name 填 piper-competition-sim。
选择 Public；不要自动添加 README、.gitignore 或 License。
点击 Create repository。当前包没有代你创建或推送远程仓库。

## 在已登录 GitHub CLI 的 Ubuntu 中上传

即使项目以后在 Windows 运行，也可以用 Ubuntu 的 Git 上传。若解压位置与示例一致：

```bash
cd /mnt/c/Users/20299/Documents/Codex/piper-competition-sim
git init -b main
git config user.name "替换为你自己的名字"
git config user.email "替换为你的提交邮箱"
git add .
git status --short
git commit -m "Add PiPER sorting simulation and project documentation"
git remote add origin https://github.com/Zerry-code/piper-competition-sim.git
git push -u origin main
```

不要照抄姓名/邮箱占位符。检查暂存列表不应有 .venv、.git 子仓库、models、tools、artifacts 或个人凭据。
GitHub CLI 认证与 WSL 代理沿用此前 ACT 仓库的方法。若网络超时，保留本地提交并恢复代理后重推，不必重新提交。

若使用不同解压位置，先用文件管理器确认地址再改 cd；不要在原来的比赛工程目录初始化或提交。

## 视频

待选择实机视频后，标注对应任务并添加到 media 或 GitHub Release，再在 README 引用真实链接。无需为了上传源码等待视频，但页面必须保留“视频尚未加入”的准确状态。

## 本地运行

参考 README，在 Windows PowerShell 建立新虚拟环境、准备资源并启动。不需要在上传用的 Ubuntu 中安装第二套仿真环境。
