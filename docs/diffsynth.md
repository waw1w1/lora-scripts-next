# DiffSynth-Studio / Qwen-Image-2.1 LoRA

本适配只开放 Qwen-Image-2.1 文生图 LoRA。运行时保留完整、固定版本的
DiffSynth-Studio，使用 uv 安装独立 Python 3.12 和虚拟环境，不修改上游训练代码。
Python 本体位于 `extensions/diffsynth/.python/`，依赖环境位于
`extensions/diffsynth/.venv/`，不使用 GUI 或其他训练器的 Python 本体。
此前安装过使用 uv 公共 Python 的开发版时，点击修复环境可重建为独立本体。

## 使用

1. 打开「设置 → 训练引擎」，安装 DiffSynth-Studio。下载源沿用引擎管理页设置。
   首次安装会下载数 GB 依赖，需要 Git、uv 和可用的 NVIDIA GPU。
2. 在训练页选择「Qwen-Image-2.1 → DiffSynth-Studio → LoRA」。
3. 通过「训练用模型 → 下载」取得官方模型目录，或选择已下载的目录。
   目录需包含 `transformer/`、`text_encoder/`、`vae/`、`processor/`；
   ComfyUI 单文件量化模型不等同于该目录布局。
4. 选择图片与同名 TXT 标注目录。支持 `重复次数_概念名` 子目录；PNG 透明通道保留。
   「整个数据集重复次数」会与子目录重复次数相乘。
5. 设置训练轮数、学习率和 Rank，点击开始训练。模型、数据和环境在后端再次检查。

训练在独立环境中运行；安装/修复/卸载、排队、停止进程树、日志与任务历史均使用
现有工作台。配置可保存、导出和重新导入。每次提交的 LoRA 权重保存在
`输出目录/任务名称/本次运行编号/` 下。

## 范围与参数

- 单卡 BF16；每批一张图片，通过梯度累积调整有效批量大小。
- 最大像素面积控制动态分辨率；宽高按 32 对齐。
- 默认 AdamW、固定学习率；支持上游 NF4 DiT 量化和模型 CPU 卸载选项。
  这些选项未在本次验证中做 GPU 显存或训练质量测试。
- 目标层留空时，由官方训练脚本自动选择。
- 保存间隔与进度按数据批次数计算；不是梯度累积后的优化器更新次数。
- TensorBoard Loss 曲线自动开启；当前入口不支持训练中周期采样预览。
- 已有 LoRA 权重可作为训练起点，不恢复优化器状态。
- 尚未开放编辑训练、全量微调、其他模型及多卡训练。

## 开发检查

后端引擎包位于 `mikazuki/engines/diffsynth/`；主 `api.py` 无需修改。
前端新增模型/引擎登记、安装门控与 Schema，复用现有训练页。

```bash
python -m pytest tests/test_diffsynth_engine.py tests/test_engine_registry.py tests/test_engine_routes.py -q
```

此命令检查参数转换、安装计划、HTTP 分发与日志读取，不执行模型训练。
前端在 `frontend/` 中使用 Node 22 执行 `npm ci` 和 `npm run check`，后者包含正式构建。

`POST /api/engines/diffsynth/dry-run` 可以生成数据清单和官方命令，不创建训练任务；
它仍要求模型目录与数据标注结构完整。

实际验证边界及环境版本见 `mikazuki/engines/diffsynth/FIELD_NOTES.md`。
云端无 GPU，真实 Windows/CUDA 训练尚未验证，不能将启动链路通过视为训练验收。
