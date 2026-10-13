import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { createDefaultModel, executeSchemaSources, serializeModel, validateModel } from "../schema/adapter"

function loadSchema(name: string) {
  const read = (file: string) => readFileSync(resolve(process.cwd(), `../mikazuki/schema/${file}.ts`), "utf8")
  return executeSchemaSources(
    [
      { name: "shared", hash: "test", schema: read("shared") },
      { name, hash: "test", schema: read(name) },
    ],
    name,
  )
}

const sd15 = loadSchema("ai-toolkit-sd15-lora")
const ideogram4 = loadSchema("ai-toolkit-ideogram4-lora")

const fieldsOf = (schema: ReturnType<typeof loadSchema>) => schema.sections.flatMap((section) => section.fields)

describe("AI Toolkit SD 1.5 schema", () => {
  it("locks the train type and defaults to a local model directory", () => {
    const model = createDefaultModel(sd15)
    expect(model).toMatchObject({
      model_train_type: "ai-toolkit-sd15-lora",
      model_input_mode: "model_directory",
      output_name: "next-sd15-lora",
      learning_rate: "1e-4",
      network_dim: 16,
    })
    // SDXL parity: no quantization controls for SD 1.5.
    expect(model).not.toHaveProperty("quantize")
  })

  it("single_file keeps the Comfy/LDM checkpoint and defaults the config dir", () => {
    const model = {
      ...createDefaultModel(sd15),
      model_input_mode: "single_file",
      dit_path: "./sd-models/sd15/v1-5-pruned-emaonly.safetensors",
    }
    const config = serializeModel(sd15, model)
    expect(config.model_input_mode).toBe("single_file")
    expect(config.dit_path).toBe("./sd-models/sd15/v1-5-pruned-emaonly.safetensors")
    expect(config.model_config_dir).toBe("./sd-models/sd15/config")
    expect(validateModel(sd15, model)).toEqual({})
  })

  it("keeps negative prompts for ddpm previews", () => {
    expect(fieldsOf(sd15).map((field) => field.key)).toContain("negative_prompts")
  })
})

describe("AI Toolkit Ideogram 4 schema", () => {
  it("defaults to the Comfy single-file components", () => {
    const model = createDefaultModel(ideogram4)
    expect(model).toMatchObject({
      model_train_type: "ai-toolkit-ideogram4-lora",
      model_input_mode: "comfyui_files",
      output_name: "next-ideogram4-lora",
      quantize: true,
    })
    // Upstream Ideogram 4 has no image-edit task in this engine yet.
    expect(model).not.toHaveProperty("training_task")
  })

  it("comfyui_files asks for DiT/VAE/TE single files plus the tokenizer directory", () => {
    const model = {
      ...createDefaultModel(ideogram4),
      dit_path: "./sd-models/ideogram4/ideogram4_fp8_scaled.safetensors",
      vae_path: "./sd-models/ideogram4/flux2-vae.safetensors",
      text_encoder_path: "./sd-models/ideogram4/qwen3vl_8b_fp8_scaled.safetensors",
      text_encoder_tokenizer_path: "./sd-models/ideogram4/qwen3-vl-8b-instruct",
    }
    const config = serializeModel(ideogram4, model)
    expect(config.dit_path).toBe("./sd-models/ideogram4/ideogram4_fp8_scaled.safetensors")
    expect(config.vae_path).toBe("./sd-models/ideogram4/flux2-vae.safetensors")
    expect(config.text_encoder_path).toBe("./sd-models/ideogram4/qwen3vl_8b_fp8_scaled.safetensors")
    expect(config.text_encoder_tokenizer_path).toBe("./sd-models/ideogram4/qwen3-vl-8b-instruct")
    expect(config).not.toHaveProperty("model_config_dir")
    // The unconditional-pass LoRA is optional: absent by default, carried when set.
    expect(createDefaultModel(ideogram4)).not.toHaveProperty("unconditional_lora_path")
    expect(validateModel(ideogram4, model)).toEqual({})
    const withLora = serializeModel(ideogram4, {
      ...model,
      unconditional_lora_path: "./sd-models/ideogram4/ideogram_4_unconditional_lora_r16.safetensors",
    })
    expect(withLora.unconditional_lora_path).toBe("./sd-models/ideogram4/ideogram_4_unconditional_lora_r16.safetensors")
  })

  it("model_directory keeps the diffusers layout option without a VAE path", () => {
    const model = {
      ...createDefaultModel(ideogram4),
      model_input_mode: "model_directory",
      model_path: "./sd-models/ideogram4/diffusers",
      text_encoder_path: "./sd-models/ideogram4/qwen3-vl-8b-instruct",
    }
    const config = serializeModel(ideogram4, model)
    expect(config.model_path).toBe("./sd-models/ideogram4/diffusers")
    expect(config).not.toHaveProperty("vae_path")
    expect(config).not.toHaveProperty("dit_path")
  })

  it("snaps preview dimensions to the 32px VAE patch grid", () => {
    const preview = fieldsOf(ideogram4).find((field) => field.role === "preview-samples")
    expect(preview?.extra).toMatchObject({ dimensionStep: 32 })
  })
})
