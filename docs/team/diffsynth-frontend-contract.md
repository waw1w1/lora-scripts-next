# DiffSynth 编辑数据集与训练预览：前后端接入约定

本次给 #371 提供可复用组件和字段约定，不带入另一套 DiffSynth 后端。
组件尚未接入训练页面，不代表当前后端已经支持编辑训练或训练中预览。

## 可直接复用的代码

- `frontend/src/components/ReferencePathsField.vue`：有序参考路径列表，支持增删和服务器路径选择；目录用于训练集，文件用于 Sample。
- `frontend/src/components/PreviewSampleField.vue`：默认一个 Sample，可增删；每个 Sample 独立配置提示词、宽高、Seed、CFG、采样步数；编辑任务额外显示参考图。
- `frontend/src/training/sampleContract.ts`：类型、默认值、Sample 编解码和可选字段校验。
- 继续使用现有 `useServerPathPick` / `PathPickerDialog`，无需新增文件浏览接口。路径是训练服务所在机器的路径，不是浏览器本机路径。

复用的是 Klein 的操作习惯及 JSON 字符串数组约定，没有移植其三张参考图上限、专属采样器或训练代码。
本次不提供上传或缩略图接口；不要调用当前分支不存在的 `/api/path_browser/image`。

## 字段约定

现有 `train_data_dir` 为目标图片目录。新增字段：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `training_task` | `"text-to-image"` / `"image-edit"` | 默认文生图，编辑能力就绪后再显示切换 |
| `control_data_dirs` | `string[]` | 编辑训练的有序参考图目录 |
| `enable_preview` | `boolean` | 默认关闭，开启后要求后端真正生成预览 |
| `sample_every_n_steps` | 正整数 | 训练预览间隔 |
| `preview_samples` | `string[]` | 每一项是一个 Sample 的 JSON 字符串 |

单个 Sample 解码后的形状：

```json
{
  "prompt": "将画面改为水彩风格",
  "controlImages": ["D:/dataset/reference/example.png"],
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "guidance_scale": 4,
  "sample_steps": 20
}
```

文生图的 `controlImages` 为空数组。默认 Sample 使用空提示词、上述数值和空参考列表。
没有全局覆盖 Sample 参数的另一套设置。LoRA Scale、负面提示词和采样器暂不提供，等确认后端支持再加。
解码兼容缺省数值并保留已有额外键；后端必须显式读取支持的字段，不能直接将整个对象展开给上游。

## 页面接入方式

```vue
<ReferencePathsField v-model="form.control_data_dirs" mode="folder" />
<PreviewSampleField
  v-model:samples="form.preview_samples"
  :editing="form.training_task === 'image-edit'"
/>
```

表单初始化 `preview_samples: encodeSamples([createSample()])`。
组件在传入空数组时显示一个默认 Sample，但不会在挂载时偷偷改写已保存配置。
提交前调用 `buildOptionalInputs(form, { editing, preview })`，能力默认均为 false；
后端明确接通并验收后才开启。它会忽略未启用功能的历史值，拒绝未支持能力的请求。
请先从待提交对象去掉这五个可选字段，再合入返回值，不能把原始表单中的隐藏字段一起提交。
纯文生图且不预览时返回空对象，兼容当前后端。

路径列表允许空行供用户编辑，但提交前需要删除未使用的空行。
切换任务时，前端可以保留编辑配置供切回使用；提交文生图预览时不能携带编辑参考图。
配置导入、历史任务重填和启动训练都要经过同样的校验，服务端仍需独立校验。

## 后端需要完成

1. 在训练器的 DiffSynth 适配层解析上述字段，不修改上游源码。
2. 编辑数据按目标图片与各参考根目录的相对路径、同名文件配对；缺图、重复匹配应明确报错。
   图片与同名 txt、Kohya 子目录重复次数沿用统一数据集转换逻辑，不把参考目录当独立目标训练集重复训练。
   具体参考图数量和模型支持范围由后端校验，不由通用组件写死。
3. 首版文生图预览按每 N 次优化器更新触发（不是梯度累积的 microbatch），需后端实现并确认；
   恢复训练后的步数、失败处理及模型训练/推理状态切换也需覆盖。
4. 预览产物接入既有任务预览/产物服务，附带训练步数和 Sample 序号；无需另建任务系统。
5. BF16 为本次适配范围。训练完成后的 ComfyUI LoRA 格式转换是另一个后端事项。

## 验收边界

- 本 PR 自动化检查覆盖 Sample 序列化、多 Sample 独立编辑、参考路径增删、能力开关和非法参数。
- 接入后再验证配置保存/导入、任务切换、编辑数据配对及每 N 步逐条生成 Sample。
- 没有跑实机训练，不能据此宣布编辑训练或预览已经可用；这次也不改安装、训练循环和模型加载。
