# Misudoku -> alai.me/play/misudoku 发布说明

## 1) 一次性前置检查（防止覆盖老站）
- 在 `alai.github.io` 仓库 -> `Settings` -> `Pages`，确认 `Custom domain = alai.me`。
- 在 `Idiom-solitaire` 仓库 -> `Settings` -> `Pages` 记录当前配置，作为回归基线。
- 新建 `play` 仓库后，不要给它单独设置自定义域名。

## 2) 创建统一发布仓库 `play`
- 仓库名：`play`
- 开启 GitHub Pages（Source 可保持默认分支发布，或 GitHub Actions，二选一即可）
- 确保仓库根目录可访问 `/play/`。

## 3) Misudoku 仓库 Secret
- 在当前仓库 `Settings` -> `Secrets and variables` -> `Actions` 中新增：
  - `PLAY_PUBLISH_TOKEN`
- Token 建议使用 Fine-grained PAT：
  - Repository access: `play`（Only selected repositories）
  - Permissions: `Contents: Read and write`, `Metadata: Read`

## 4) 自动发布工作流
- 文件：`.github/workflows/publish-to-play.yml`
- 触发：push 到 `master` 或手动 `workflow_dispatch`
- 行为：把当前仓库静态文件同步到 `play` 仓库的 `misudoku/` 子目录。

## 5) 发布后验收
- 新地址：`https://alai.me/play/misudoku/`
- 老地址：`https://alai.me/Idiom-solitaire/` 仍可访问
- iOS Safari 可添加到主屏并正常启动
- 首次联网打开后，断网重启仍可进入应用壳
