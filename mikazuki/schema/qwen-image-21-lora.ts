Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("qwen-image-21-lora").disabled().description("训练种类"),
        model_input_mode: Schema.union(["directory", "components"]).default("directory").description("模型输入方式：ComfyUI 分组件 / 完整模型目录；首版仅支持 Qwen-Image-2.1 BF16 文生图"),
    }).description("训练用模型"),
    Schema.union([
        Schema.object({
            model_input_mode: Schema.const("directory"),
            diffsynth_model_dir: Schema.string().role('filepicker', { type: "folder" }).required().description("模型目录（transformer / text_encoder / vae）"),
        }),
        Schema.object({
            model_input_mode: Schema.const("components"),
            dit_path: Schema.string().role('filepicker', { type: "model-file", filter: "*.safetensors;*.safetensors.index.json" }).required().description("Comfy-Org Qwen-Image-2.1 BF16 DiT；可填写文件、分片索引或目录"),
            text_encoder_path: Schema.string().role('filepicker', { type: "model-file", filter: "*.safetensors;*.safetensors.index.json" }).required().description("Qwen3-VL-8B BF16 文本编码器"),
            vae_path: Schema.string().role('filepicker', { type: "model-file", filter: "*.safetensors;*.safetensors.index.json" }).required().description("Qwen-Image-2.1 BF16 VAE"),
        }),
    ]),
    Schema.object({
        processor_path: Schema.string().role('filepicker', { type: "folder" }).description("Qwen-Image-2.1 Processor 目录；留空时从模型相邻目录查找，不自动下载"),
        dataset_format: Schema.union(["image_text", "metadata"]).default("image_text").description("数据集格式：图片 + TXT / 原生 CSV、JSON、JSONL"),
    }).description("数据集设置"),
    Schema.union([
        Schema.object({
            dataset_format: Schema.const("image_text"),
            train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/qwen-image-21").required().description("图片与同名 TXT；支持 重复次数_概念名 子目录，普通目录和根目录图片均保留"),
            dataset_repeat: Schema.number().min(1).step(1).default(1).description("全局重复次数，与子目录重复次数相乘"),
        }),
        Schema.object({
            dataset_format: Schema.const("metadata"),
            dataset_base_path: Schema.string().role('filepicker', { type: "folder" }).required().description("元数据中图片相对路径的根目录"),
            dataset_metadata_path: Schema.string().role('filepicker', { type: "file", filter: "*.csv;*.json;*.jsonl" }).required().description("包含 image、prompt 字段的原生元数据；不按文件夹名重复"),
        }),
    ]),
    Schema.object({
        max_pixels: Schema.number().min(1024).step(1).default(1048576).description("最大像素面积；保持宽高比，按 32 对齐，保留透明通道"),
    }),
    Schema.object({
        output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").required().description("模型保存目录"),
        output_name: Schema.string().default("qwen-image-21-lora").required().description("任务名称，每次训练保存到独立子目录"),
        save_steps: Schema.number().min(1).step(1).description("每 N 次优化器更新保存权重；留空则每轮保存。权重不含优化器状态"),
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
    }).description("显存设置（单卡 BF16）"),
    Schema.object({
        sample_enabled: Schema.boolean().default(false).description("训练中预览；当前上游版本不可与模型 CPU 卸载同时启用"),
    }).description("预览设置"),
    Schema.union([
        Schema.object({
            sample_enabled: Schema.const(true),
            sample_every_n_steps: Schema.number().min(1).step(1).default(100).description("每 N 次优化器更新预览，与保存步数使用同一口径；失败时任务报错停止"),
            preview_samples: Schema.array(String).role('preview-samples').default(['{"prompt":"","width":1024,"height":1024,"seed":42,"guidance_scale":4,"sample_steps":20}']).description("预览样例，可添加多组独立参数"),
        }),
        Schema.object({ sample_enabled: Schema.const(false) }),
    ]),
])
