# Qwen Image 2.1 Edit 审阅修复记录

针对用户提供的《问题清单.md》，修复基于 Edit 提交 `7e56601`，未合并到 dev。

| 项目 | 处理结果 | 验证方式 |
| --- | --- | --- |
| REV-01 | 隐藏模式字段使用 `external-control` 标记，合法用户状态不再被只读默认值覆盖；枚举转换保留 role/extra；真正 disabled/const/锁定字段仍受保护 | 使用实际 Qwen schema 挂载 TrainingPage，恢复草稿、重新挂载、导入及任务重编辑后断言最终 trainingApi.run 请求 |
| REV-02 | 参考图按父目录建立文件主名索引，每个父目录每次预检只扫描一次；本次已验证的参考图不重复解码检查；保留顺序、重复次数和歧义检测；已有日志记录预检开始/完成 | 128、512 张目标图，2 组参考图、2 个相对目录：均只扫描 4 次，候选量分别 258、1026 |
| REV-03 | Edit batch size 非 1 时，现有字段错误和配置诊断即时提示，并阻止提交；不静默改写参数；保留梯度累积与 T2I batch 能力 | 页面切换后断言字段错误及提交按钮状态；参数校验测试；后端仍独立拒绝 |
| REV-04 | Edit 预览逐样例检查非空参考图路径并指出编号；关闭预览后忽略未完成草稿；服务器文件有效性仍由后端检查 | 多样例、空数组、空白路径、关闭预览、合法路径形式及 T2I 测试 |
| REV-05 | 针对固定上游捕获回调，在适配层按实例包装文本编码器 forward；正常/异常均清理本次新增的上游临时回调；预编码共用相同清理机制 | 使用固定上游真实 forward 方法和微型 Torch 内层模型，连续 32 次成功、3 次异常后检查 hook 基线；保留调用前及调用中添加的其他调试回调 |
| TEST-01 | 新增草稿重载、导出 TOML 再导入、任务重编辑、两种数据集格式、Edit/T2I 往返及最终请求测试；另测后端 /api/run 生成的 Edit 引擎参数 | Vue 页面生命周期 + API mock；后端 TestClient + 任务提交拦截，不启动真实训练 |
| TEST-02 | 真实 GPU / Windows / ComfyUI 验收仍未完成；Edit 明确标为实验性 | 不以 CPU、mock 或 dry-run 替代真实训练验收 |

## 默认值与非法值策略

旧配置缺少 `training_task` 时使用 `text-to-image`。合法的 `image-edit` 在草稿恢复、导入及任务重新编辑后保留。
显式非法模式保持原值并显示错误，必须重新选择模式才能提交，避免静默改变训练类型。
T2I 提交仍去除参考图目录、清空预览参考图，表单草稿继续保留参考图供切回 Edit 使用。

## 回归范围与限制

运行 `frontend/npm run check`（类型检查、lint、全部 Vitest、生产构建）及
`pytest tests/test_diffsynth_engine.py tests/test_diffsynth_review.py tests/test_diffsynth_edit.py -q`。
固定上游从现有完整 checkout 加入测试 Python 路径，不加载真实模型权重。

原来失败的 `test_comfy_component_mode_and_processor_errors` 与 dev 已有的 managed Processor 行为不符；
现改为验证固定托管路径、忽略旧路径覆盖，且预检不下载文件。未改变 Processor 生产实现。

未完成真实 GPU 优化器更新、预览后继续训练、CPU 卸载组合、真实 LoRA 的 ComfyUI Edit 出图及原生 Windows 验收。
Hook 测试仅证明回调生命周期正确；不宣称已证明显存泄漏或量化真实性能提升。

## 本次实际运行结果

- 前端完整 `npm run check`：35 个测试文件、232 项测试通过，类型检查和生产构建通过。
  保留原有 2 条 lint warning 及 bundle 体积提示，无新增检查错误。
- DiffSynth 相关后端：41 passed、1 skipped。跳过项为依赖约定相邻路径及子进程环境的
  `test_http_run_spawns_real_entry_parser_without_training`；实际上游 parser/dataset/model-function
  的进程内契约测试及 /api/run 任务配置测试已执行通过。
- 环境使用 CPU Torch 2.8.0；未加载真实 Qwen 模型，不包含 CUDA 训练验收。

## 二次复审：REV-06（基于 8281a3b）

针对《复审问题清单_8281a3b.md》，在数据提交预检中补齐参考图尺寸保护：
使用与运行参数相同的 `bucket_settings(config).max_pixels`（`resolution` 优先），
按固定上游 `ImageCropAndResize` 的等比缩小、整数截断及向下对齐到 32 的规则预测尺寸。
任意一边归零时拒绝提交，错误包含样本编号、参考路径、原始尺寸、最大像素面积、
对齐后尺寸及每边至少 32 像素的条件。不会临时放大参考图。

- 图片＋TXT、JSON、JSONL、CSV 共用保护；缓存开关均在创建训练任务前校验。
- 原有参考图检查去重、目录索引、多图顺序及目标图分桶保持；预览使用独立的管线几何处理，
  不套用仅属于训练数据加载器的限制。
- 新增 48 组配置适配测试：四种入口 × 缓存开关 × 六组尺寸。
  16×16 及在 65536 像素限制下缩小后短边归零的 4096×32 / 32×4096 被拒绝；
  32×32、256×256、2048×1024 通过，同时断言参考图顺序、目标桶尺寸及重复次数。
- 另用固定上游真实算子核对三个像素限制、九组尺寸的预测；合法尺寸实际执行 resize/crop。
- 本轮后端回归：`pytest tests/test_diffsynth_engine.py tests/test_diffsynth_review.py tests/test_diffsynth_edit.py -q`
  **90 passed、1 skipped**。跳过项仍为原有真实子进程 parser 烟测。
  本轮没有前端代码变更，未重跑前端检查；上文 232 项是上一轮结果。

这些结果覆盖 CPU 数据预检与算子，不代表已完成缓存/非缓存真实 GPU 训练、
原生 Windows 或 ComfyUI 产物验收。TEST-02 仍待完成；分支仍未合并到 dev。
