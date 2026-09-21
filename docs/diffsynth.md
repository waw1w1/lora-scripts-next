# DiffSynth-Studio / Qwen-Image-2.1 LoRA

仅开放 Qwen-Image-2.1 **BF16 文生图 LoRA**，不包含 Edit、量化权重、全量微调或多卡。
API 或导入配置中的编辑任务、非空参考图目录也会明确拒绝，不会静默当作文生图启动。
非法训练数值和预览配置返回带字段名的错误；不会因这些配置创建训练任务。
完整 DiffSynth-Studio 固定在 `7686e54d41d25c0e8ed5f1318acc23b6bb832654`，不修改其源码。
训练模块、数据加载器、优化器和训练循环均使用上游实现；适配层负责输入转换、任务管理和 logger 回调。

## 准备与使用

1. 在「设置 → 训练引擎」安装 DiffSynth。沿用已有下载源设置，需要 Git、uv、网络及支持 BF16 的 NVIDIA GPU。
2. 在训练页选择「Qwen-Image-2.1 → DiffSynth-Studio → LoRA」。
3. 模型输入选择 `components`，分别选择 Comfy-Org 的 BF16 文件：
   - `diffusion_models/qwen_image_2.1_bf16.safetensors`
   - `text_encoders/qwen3vl_8b_bf16.safetensors`
   - `vae/qwen_image_2.1_vae_bf16.safetensors`
4. 准备官方 Qwen-Image-2.1 **Processor** 目录。仅有三个权重文件不包含分词器与处理器配置。
   `processor_path` 可手动指定；留空时只在模型相邻目录查找唯一完整目录，不自动下载。
5. 选择数据集格式、保存目录、轮数及 Rank，提交训练。启动时检查本地模型、数据和运行环境。

模型也可用 `directory` 模式选择包含 `transformer/`、`text_encoder/`、`vae/` 的 BF16 目录。
组件路径支持单文件、分片索引，或只有一个模型候选的目录。选择分片时自动查找对应索引，
缺失分片、索引越界、多候选、错误模型结构和非 BF16 权重会报错。官方目录中的 F32 权重不在本版 BF16 输入范围内。

Processor 必需文件：`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、
`chat_template.jinja`、`video_preprocessor_config.json`。文件完整性预检查后，由上游 AutoProcessor 完成实际加载。
训练页复用现有服务端路径浏览，不提供额外模型下载入口。

## ComfyUI 模型及输出

Comfy-Org BF16 文件与 DiffSynth 原生布局并非完全相同：

| 组件 | 转换 |
| --- | --- |
| DiT | `img_mlp.gate_up` 按行拆为 `gate_layer`、`proj` |
| 文本编码器 | 语言模型键名恢复 `model.language_model.*` 前缀 |
| VAE | Wan 风格层名映射；移除卷积权重中长度为 1 的时间轴 |

转换前检查完整键名及尺寸，转换结果必须匹配固定上游的模型签名。
转换文件只写入 `extensions/diffsynth/cache/models/`，按输入路径、大小、修改时间隔离缓存，
不覆盖原模型；首次运行需为转换副本预留磁盘空间。

输出为 safetensors LoRA，使用 ComfyUI 内置加载器支持的 `lora_A.weight` / `lora_B.weight`，
保持上游 alpha=rank 的倍率。MLP 两个子层分别导出，由 ComfyUI 原生映射作用于合并权重的两个区段。
使用**支持 Qwen-Image-2.1 的新版 ComfyUI**，放入 `models/loras`，接内置 Load LoRA / Load LoRA Model Only 节点；
本版只训练 DiT，CLIP 不产生 LoRA。

已通过未修改 ComfyUI 加载器的 CPU 合成张量映射与合并数值测试。
**尚未完成真实 Qwen GPU 训练 → 输出权重 → ComfyUI 出图验收**，不能据此承诺任意硬件开箱即训。

## 数据与步数

- `image_text`：递归读取图片及同名 TXT。`重复次数_名称` 第一层子目录乘全局 `dataset_repeat`（默认 1）；
  普通目录和根目录图片也保留。空 TXT 表示空提示词；缺失 TXT 报错。RGBA 不丢失。
- `metadata`：指定 `dataset_base_path` 与原生 CSV / JSON / JSONL，必需 `image`、`prompt` 字符串。
  直接交给上游 UnifiedDataset，不按文件夹名重复；CSV 空提示词规范化为空字符串。
- 每批一张图；梯度累积、保存间隔、预览间隔、Loss 和总步数统一使用**优化器更新次数**。
  总步数为 `ceil(每轮样本数 / 梯度累积步数) × 轮数`，包含每轮末尾不足一次累积的更新。

## 预览

复用作者 `feat/ai-toolkit-klein` 分支的 PreviewSampleField 交互，首版裁剪为文生图字段。
开启后默认一组，可增删；每组独立设置 prompt、width、height、seed、guidance_scale、sample_steps。
不暴露采样器切换、参考图或网络倍率等本适配未实现的选项。

预览在上游 logger 的优化器更新回调中调用现有 pipeline，不复制训练循环。
图片保存到本次输出目录的 `sample/`，复用任务页现有图片、缩略图与预览 API。
界面显示采样阶段，文件名关联 Step 与 Sample；预览失败会让任务明确失败。
回调恢复随机数状态、训练/评估状态及训练噪声调度器，不修改优化器。

**当前限制：不能同时开启模型 CPU 卸载和训练中预览。** 固定上游的卸载管理器仅存在于训练循环局部变量中，
logger 没有获取它的公开接口，且其重计算标记不适用于重复无梯度采样。
本适配明确拒绝该组合，避免修改私有钩子或复制训练循环。低显存卸载训练请关闭预览；
若要同时支持两者，需要先给 DiffSynth 增加公开的采样/卸载生命周期接口。

## 环境、任务与配置

独立 Python 3.12 位于 `extensions/diffsynth/.python/`，虚拟环境位于 `.venv/`。
PyTorch 2.8.0/cu128、torchvision 0.23.0 与 DiffSynth 依赖均独立安装，不共用 GUI 或其他训练器的 Python 包。
不需要 DeepSpeed、FlashAttention 或 Bash。显卡驱动仍由操作系统提供。

安装由现有 Task 管理一个安装监督进程，停止任务会终止完整子进程树。
安装、修复、卸载与训练通过同一环境锁协调；中断安装变为 broken，源码或依赖变化使 ready 失效。
DiffSynth 训练任务继续使用现有队列；重跑重新检查配置并建立新输出目录。

每次提交保存 UI TOML 与引擎参数 JSON，实际加载参数还保存在输出目录的 `engine_config.json` / `training_args.json`。
输出路径为 `output_dir/output_name/本次运行编号/`。配置沿用原有导入、导出、历史和重新编辑接口。
检查点只含 LoRA，不含优化器状态；已有 LoRA 可作为新训练起点。

## 验证

`POST /api/engines/diffsynth/dry-run` 生成配置，不创建训练任务。
独立入口 `entry.py --check-only --project-root ... --config ...` 只导入上游及解析参数，不读训练模型张量。

前端：Node 22，`npm ci` 后运行 `npm run check`。
后端：`pytest tests/test_diffsynth_engine.py tests/test_diffsynth_review.py -q`。
上游回调和入口烟测需要 DiffSynth 依赖及固定源码；`test_diffsynth_comfy.py` 另需原版 ComfyUI 及其依赖。
具体证据与剩余验收项见 `mikazuki/engines/diffsynth/FIELD_NOTES.md`。
