# -*- coding: utf-8 -*-
"""RoBERTa-large Fill-Mask WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：RoBERTa-large 已就绪（演示模式，未加载真实权重）"


def fake_fill_mask(text: str, top_k: int) -> str:
    """模拟对含 [MASK] 的句子进行掩码预测并返回可视化描述。"""
    if not (text or "").strip():
        return "请输入包含 [MASK] 的英文句子。"
    if "[MASK]" not in text and "<mask>" not in text.lower():
        return "请在句子中保留至少一个 [MASK] 占位符，以便进行掩码语言模型预测。\n示例：The capital of France is [MASK]."
    k = max(1, min(10, int(top_k) if isinstance(top_k, (int, float)) else 5))
    lines = [
        "[演示] 已对掩码位置进行预测（未加载真实模型）。",
        f"Top-{k} 候选词示例（占位）：",
    ]
    for i in range(k):
        lines.append(f"  {i+1}. candidate_{i+1} (score: 0.{9-i}xx)")
    lines.append("\n加载真实 RoBERTa-large 后，将在此显示实际预测词及其置信度。")
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="RoBERTa-large Fill-Mask WebUI") as demo:
        gr.Markdown("## RoBERTa-large 掩码语言模型 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 RoBERTa-large 掩码语言模型（MLM）的典型使用流程，"
            "包括模型加载状态与含 [MASK] 句子的预测结果展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("Fill-Mask 预测"):
                gr.Markdown(
                    "在下方输入包含 `[MASK]` 的英文句子，模型将预测该位置的候选词。"
                )
                inp = gr.Textbox(
                    label="输入句子（需包含 [MASK]）",
                    placeholder="例如：The capital of France is [MASK].",
                    lines=3,
                )
                top_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, label="Top-K 候选数"
                )
                out = gr.Textbox(label="预测结果说明", lines=10, interactive=False)
                run_btn = gr.Button("执行预测（演示）")
                run_btn.click(
                    fn=fake_fill_mask, inputs=[inp, top_k], outputs=out
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 RoBERTa-large 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8760, share=False)


if __name__ == "__main__":
    main()
