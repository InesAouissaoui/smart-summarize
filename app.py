"""
SmartSummarize - AI-Powered Document Summarizer
Built by Ines Aouissaoui

Uses HuggingFace transformers to generate concise summaries
from long-form text. Supports adjustable summary length
and multiple input formats.
"""

import gradio as gr
from transformers import pipeline

# Load the summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU, change to 0 for GPU
)


def summarize_text(text: str, max_length: int = 150, min_length: int = 40) -> dict:
    """
    Summarize input text using BART model.
    
    Args:
        text: The input text to summarize
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
    
    Returns:
        dict with summary, original length, summary length, and compression ratio
    """
    if not text or len(text.strip()) < 50:
        return {
            "summary": "Please provide a longer text (at least 50 characters).",
            "original_words": 0,
            "summary_words": 0,
            "compression_ratio": "N/A"
        }

    # Split long texts into chunks (BART has a 1024 token limit)
    max_chunk_length = 1024
    words = text.split()
    
    if len(words) > max_chunk_length:
        chunks = []
        for i in range(0, len(words), max_chunk_length):
            chunk = " ".join(words[i:i + max_chunk_length])
            chunks.append(chunk)
    else:
        chunks = [text]

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    final_summary = " ".join(summaries)
    
    # If we had multiple chunks, do a final pass
    if len(chunks) > 1:
        result = summarizer(
            final_summary,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        final_summary = result[0]["summary_text"]

    original_words = len(words)
    summary_words = len(final_summary.split())
    ratio = f"{((1 - summary_words / original_words) * 100):.1f}%"

    return {
        "summary": final_summary,
        "original_words": original_words,
        "summary_words": summary_words,
        "compression_ratio": ratio
    }


def process(text, max_len, min_len):
    """Gradio interface handler."""
    result = summarize_text(text, int(max_len), int(min_len))
    stats = f"Original: {result['original_words']} words | Summary: {result['summary_words']} words | Compressed by: {result['compression_ratio']}"
    return result["summary"], stats


# Sample texts for demo
SAMPLE_ARTICLE = """Artificial intelligence has transformed numerous industries over the past decade, 
from healthcare to finance, transportation to entertainment. Machine learning algorithms now power 
recommendation systems that suggest what we watch, read, and buy. Natural language processing enables 
virtual assistants to understand and respond to human speech with increasing accuracy. Computer vision 
systems can identify objects, faces, and even emotions in images and video streams. In healthcare, AI 
is being used to detect diseases from medical images, predict patient outcomes, and accelerate drug 
discovery. Financial institutions leverage AI for fraud detection, algorithmic trading, and risk 
assessment. The automotive industry is racing toward fully autonomous vehicles, with AI systems 
processing data from cameras, lidar, and radar sensors in real time. Despite these advances, 
challenges remain in areas such as bias in AI systems, data privacy concerns, and the need for 
explainable AI that can justify its decisions to human users. Researchers are also grappling with 
the environmental impact of training large models, which requires significant computational resources 
and energy consumption. As AI continues to evolve, the focus is shifting toward developing more 
efficient, fair, and transparent systems that can benefit society while minimizing potential harms."""

# Build Gradio UI
with gr.Blocks(
    title="SmartSummarize",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="zinc",
    ),
    css="""
    .gradio-container { max-width: 800px !important; }
    footer { display: none !important; }
    """
) as app:
    gr.Markdown("# SmartSummarize")
    gr.Markdown("*AI-powered document summarization using BART transformer*")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Paste your article, document, or any long text here...",
                lines=10,
                value=SAMPLE_ARTICLE
            )
            with gr.Row():
                max_length = gr.Slider(
                    50, 300, value=150, step=10,
                    label="Max Summary Length (tokens)"
                )
                min_length = gr.Slider(
                    20, 100, value=40, step=10,
                    label="Min Summary Length (tokens)"
                )
            btn = gr.Button("Summarize", variant="primary", size="lg")

        with gr.Column():
            output_summary = gr.Textbox(label="Summary", lines=8, interactive=False)
            output_stats = gr.Textbox(label="Statistics", interactive=False)

    btn.click(fn=process, inputs=[input_text, max_length, min_length], outputs=[output_summary, output_stats])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
