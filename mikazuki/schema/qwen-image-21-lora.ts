Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("qwen-image-21-lora").disabled().description("训练种类"),
        diffsynth_model_dir: Schema.string().role('filepicker', { type: "folder" }).default("./sd-models/qwen-image-21").required().description("Qwen-Image-2.1 官方模型目录（包含 transformer、text_encoder、vae、processor；可在模型文件检查中下载）"),
    }).description("训练用模型"),
    Schema.object({
        train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/qwen-image-21").required().description("图片与同名 TXT 标注目录；支持 重复次数_概念名 子目录；保留 PNG 透明通道"),
        dataset_repeat: Schema.number().min(1).step(1).default(1).description("整个数据集每轮重复次数（会与子目录重复次数相乘）"),
        max_pixels: Schema.number().min(1024).step(1).default(1048576).description("最大像素面积，1048576 对应 1024×1024；保持宽高比，按 32 对齐"),
    }).description("数据集设置"),
    Schema.object({
        output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").required().description("模型保存目录"),
        output_name: Schema.string().default("qwen-image-21-lora").required().description("任务名称，每次训练保存到独立子目录"),
        save_steps: Schema.number().min(1).step(1).description("每 N 个数据批次保存权重；留空则每轮保存。权重不含优化器状态"),
    }).description("保存设置"),
    Schema.object({
        num_epochs: Schema.number().min(1).step(1).default(5).description("训练轮数"),
        learning_rate: Schema.string().default("1e-4").description("学习率（AdamW，固定学习率）"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).default(1).description("梯度累积步数；每次读取一张图片"),
        lora_rank: Schema.number().min(1).step(1).default(32).description("LoRA Rank"),
        lora_target_modules: Schema.string().default("").description("LoRA 目标层，以逗号分隔；留空由官方脚本自动选择目标层"),
        lora_checkpoint: Schema.string().role('filepicker', { type: "model-file" }).description("可选：从已有 LoRA 权重继续训练（不恢复优化器）"),
    }).description("训练参数"),
    Schema.object({
        use_gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点，节省显存"),
        use_gradient_checkpointing_offload: Schema.boolean().default(false).description("将梯度检查点卸载到内存"),
        initialize_model_on_cpu: Schema.boolean().default(true).description("在 CPU 初始化模型，降低启动显存峰值"),
        enable_model_cpu_offload: Schema.boolean().default(false).description("逐层将模型权重从 CPU 加载到 GPU，节省显存但会变慢"),
        diffsynth_quantization: Schema.union(["none", "bitsandbytes_nf4"]).default("none").description("DiT 训练量化；NF4 更省显存，使用上游可微量化后端"),
    }).description("显存设置（单卡 BF16）"),
])
