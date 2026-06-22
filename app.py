import gradio as gr
import os
import logging
from src.rag_engine import CrediTrustRAG

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Initialize RAG Engine safely
try:
    rag = CrediTrustRAG()
    status_indicator = "🟢 ONLINE (ENCRYPTED)"
except Exception as e:
    rag = None
    status_indicator = f"🔴 OFFLINE: {e}"

# --- 2. ELITE CBE-INSPIRED CSS ---
CREDITRUST_CSS = """
footer {display: none !important;}
.gradio-container { background-color: #fdfdfd !important; }
.cbe-header { 
    background-color: #4d148c !important; 
    color: white !important; 
    padding: 25px; 
    text-align: center; 
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.sidebar-panel { 
    background-color: #ffffff !important; 
    border-left: 6px solid #4d148c !important; 
    padding: 15px !important; 
    border-radius: 8px !important;
    border: 1px solid #e0e0e0;
    margin-bottom: 12px;
}
.guidance-box { 
    background-color: #f3f0ff !important; 
    padding: 15px !important; 
    border-radius: 8px !important; 
    font-size: 0.9em;
}
.cbe-button { 
    background-color: #4d148c !important; 
    color: white !important; 
    font-weight: bold !important;
    border: none !important;
}
.cbe-button:hover { background-color: #3b106d !important; }
"""

# --- 3. UI LOGIC (Gradio 6 Dictionary Messages Format) ---

def predict(message, history):
    # Validation Check
    if not message.strip():
        yield history
        return

    if rag is None:
        history.append({"role": "assistant", "content": "❌ **System Error**: Logic engine failed to initialize. Please check configurations."})
        yield history
        return

    # Add interaction sequence
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "🔄 *Synthesizing institutional memory...*" })
    yield history 

    try:
        # Generate Streaming Response
        response_gen = rag.generate_answer_stream(message)
        
        for part_response, sources in response_gen:
            # Format high-transparency sources footer for Compliance
            source_box = "\n\n---\n### 🔍 Verified Sources (Evidence Archive):\n"
            for i, s in enumerate(sources[:3]):
                source_box += f"- **{s['product']}**: \"{s['text'][:140]}...\"\n"
            
            history[-1]["content"] = part_response + source_box
            yield history
            
    except Exception as e:
        history[-1]["content"] = f"⚠️ **Critical Alert**: Analysis interrupted. Error: {str(e)}"
        yield history

def clear_session():
    """Handles the UI 'Reset' capability requested in feedback."""
    return [{"role": "assistant", "content": "Asha, the analysis session has been reset. How can I help you improve our product portfolio now?"}]

# --- 4. THE COMMAND CENTER LAYOUT ---
with gr.Blocks(title="CrediTrust Intelligent Analyst", css=CREDITRUST_CSS) as demo:
    
    # Professional Header
    gr.HTML("""
        <div class='cbe-header'>
            <h1 style='color: white; margin: 0;'>🛡️ CrediTrust Strategic Intelligence Analyst</h1>
            <p style='color: white; margin: 5px 0 0 0;'>Corporate Feedback Analytics Platform — Institutional Integrity v1.2</p>
        </div>
    """)

    with gr.Row():
        # --- LEFT SIDEBAR: STAKEHOLDER PANELS ---
        with gr.Column(scale=3):
            # Panel A: Guidance (Specific Feedback Fix)
            with gr.Group(elem_classes="guidance-box"):
                gr.Markdown("### 📖 Operational Guidance")
                gr.Markdown(
                    "1. Type a specific question about **banking pain points**.\n"
                    "2. Use **Analyze** to start the neural retrieval process.\n"
                    "3. Review the **Verification Panel** for evidence logs."
                )

            # Panel B: Vitals
            with gr.Group(elem_classes="sidebar-panel"):
                gr.Markdown(f"### **System Integrity**\n**Status:** {status_indicator}")
            
            # Panel C: Stats
            with gr.Group(elem_classes="sidebar-panel"):
                gr.Markdown("### **AI Engine Spec**\n- Index: 464k Complaints\n- Grounding: strictly enforced\n- Latency: Optimized")

            # Reset Button (Specific Feedback Fix)
            clear_btn = gr.Button("🗑️ CLEAR CURRENT ANALYSIS", variant="secondary")

        # --- RIGHT SIDE: ANALYST PORTAL ---
        with gr.Column(scale=9):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Welcome Asha. Our data-processing engine is ready. Please input your trend analysis request."}],
                height=520,
                show_label=False
            )
            
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Search complaints (e.g., billing disputes, loan denials, transfer speeds)...",
                    container=False,
                    scale=10,
                )
                submit_btn = gr.Button("ANALYZE", elem_classes="cbe-button", scale=2)

            # Industry-Relevant Examples for PMs (KPI Enrichment)
            gr.Examples(
                examples=[
                    "Analyze main causes for Credit Card application denials.",
                    "Identify friction points regarding Money Transfer speed.",
                    "Summarize recurring complaints about Savings Account monthly fees.",
                    "Provide evidence for unauthorized activity on Personal Loans."
                ],
                inputs=txt,
                label="Strategic Analyst Presets (Asha's Portal)"
            )

    # UI Interaction Connections
    submit_btn.click(predict, [txt, chatbot], [chatbot])
    txt.submit(predict, [txt, chatbot], [chatbot])
    clear_btn.click(clear_session, None, [chatbot])

# --- 5. EXECUTION POINT ---
if __name__ == "__main__":
    demo.launch(server_port=7860)