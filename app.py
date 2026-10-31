"""
Gradio Web Application for Commitment and Deadline Extraction
"""

import os
import io
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import nlp_engine

# Preset single sentence examples
SINGLE_EXAMPLES = [
    "I will submit the NLP assignment by Friday.",
    "Ahmed promised to send the project report tomorrow.",
    "Sara needs to finish the presentation before Monday.",
    "I must complete my homework within two days.",
    "The weather is pleasant today.",
    "We have to prepare the project.",
    "Bilal should review the document before next Tuesday.",
    "I will attend the meeting today."
]

# Preset multi-sentence document
SAMPLE_DOCUMENT = """Hi Team,
Ahmed promised to send the project report tomorrow.
I will submit the NLP assignment by Friday.
Sara needs to finish the presentation before Monday.
The weather is pleasant today.
We have to prepare the project review within two weeks.
I enjoy learning Python and NLP.
Bilal should review the document before Thursday.
We will see you at the university."""


def analyze_single_sentence(sentence, natural_order, extended_deadlines, resolve_pronouns):
    if not sentence or not sentence.strip():
        return (
            "<div class='status-box status-neutral'>Please enter a sentence above to analyze.</div>",
            "-", "-", "-", "-",
            pd.DataFrame(columns=["Token", "POS", "Lemma", "Dependency"]),
            pd.DataFrame(columns=["Entity", "Label"])
        )

    info = nlp_engine.extract_commitment_info(
        sentence,
        natural_order=natural_order,
        use_extended_deadlines=extended_deadlines,
        resolve_pronouns=resolve_pronouns
    )

    is_comm = info["Commitment Detected"]
    if is_comm:
        status_html = f"""
        <div class='status-box status-commitment'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #166534;'>✅ Commitment Detected</div>
            <div style='font-size: 0.9rem; color: #15803d; margin-top: 4px;'>Trigger matched: <code>{info['Trigger']}</code></div>
        </div>
        """
    else:
        status_html = """
        <div class='status-box status-non-commitment'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #991b1b;'>❌ No Commitment Detected</div>
            <div style='font-size: 0.9rem; color: #b91c1c; margin-top: 4px;'>No commitment trigger pattern matched in this sentence.</div>
        </div>
        """

    person = info["Responsible Person"] or "Unknown"
    action = info["Action"] or "Not identified"
    deadline = info["Deadline"] or "None detected"
    trigger = info["Trigger"] or "None"

    # Linguistic breakdown (POS & Dependency tags as in notebook cell 52)
    doc = nlp_engine.nlp(sentence)
    tokens_data = []
    for token in doc:
        if token.text.strip():
            tokens_data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Lemma": token.lemma_,
                "Dependency": token.dep_
            })
    tokens_df = pd.DataFrame(tokens_data)

    # Named Entities
    entities_data = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents]
    entities_df = pd.DataFrame(entities_data) if entities_data else pd.DataFrame(columns=["Entity", "Label"])

    return status_html, person, action, deadline, trigger, tokens_df, entities_df


def analyze_document(text, filter_option, natural_order, extended_deadlines, resolve_pronouns):
    df, stats = nlp_engine.analyze_text(
        text,
        natural_order=natural_order,
        use_extended_deadlines=extended_deadlines,
        resolve_pronouns=resolve_pronouns
    )

    if df.empty:
        stats_html = "<div class='stats-container'>No text provided to analyze.</div>"
        return df, stats_html, None

    # Filter dataframe
    if filter_option == "Commitments Only":
        display_df = df[df["Commitment Detected"] == True].copy()
    elif filter_option == "Non-Commitments Only":
        display_df = df[df["Commitment Detected"] == False].copy()
    else:
        display_df = df.copy()

    # Create CSV file for download
    csv_filename = "extracted_commitments.csv"
    display_df.to_csv(csv_filename, index=False)

    stats_html = f"""
    <div class='stats-container'>
        <div class='stat-card'>
            <div class='stat-number'>{stats['total']}</div>
            <div class='stat-label'>Total Sentences</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #16a34a;'>
            <div class='stat-number' style='color: #16a34a;'>{stats['commitments']}</div>
            <div class='stat-label'>Commitments</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #dc2626;'>
            <div class='stat-number' style='color: #dc2626;'>{stats['non_commitments']}</div>
            <div class='stat-label'>Non-Commitments</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #2563eb;'>
            <div class='stat-number' style='color: #2563eb;'>{stats['deadlines']}</div>
            <div class='stat-label'>Deadlines Found</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #9333ea;'>
            <div class='stat-number' style='color: #9333ea;'>{stats['persons']}</div>
            <div class='stat-label'>Assignees</div>
        </div>
    </div>
    """

    return display_df, stats_html, csv_filename


def handle_file_upload(file_obj, natural_order, extended_deadlines, resolve_pronouns):
    if file_obj is None:
        return "", pd.DataFrame(), "Please upload a file."
    
    try:
        content = ""
        # Handle file paths or file-like objects
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        
        if path.endswith(".csv"):
            upload_df = pd.read_csv(path)
            # Find the best column containing text
            candidate_cols = [c for c in upload_df.columns if "sentence" in c.lower() or "text" in c.lower() or "message" in c.lower()]
            if candidate_cols:
                content = "\n".join(upload_df[candidate_cols[0]].dropna().astype(str).tolist())
            else:
                content = "\n".join(upload_df.iloc[:, 0].dropna().astype(str).tolist())
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        df, stats_html, csv_file = analyze_document(content, "All Sentences", natural_order, extended_deadlines, resolve_pronouns)
        return content, df, stats_html, csv_file
    except Exception as e:
        empty_df = pd.DataFrame()
        return "", empty_df, f"<div style='color:red;'>Error reading file: {str(e)}</div>", None


def get_evaluation_results():
    metrics, results_df = nlp_engine.evaluate_model()

    # Generate Matplotlib chart
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1-Score"]]
    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706"]

    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Model Evaluation on Benchmark Dataset (20 Sentences)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10.5, fontweight="bold")

    plt.tight_layout()

    # Metrics HTML summary cards
    kpi_html = f"""
    <div class='stats-container'>
        <div class='stat-card' style='border-top: 4px solid #2563eb;'>
            <div class='stat-number' style='color: #2563eb;'>{metrics['Accuracy']}%</div>
            <div class='stat-label'>Accuracy</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #059669;'>
            <div class='stat-number' style='color: #059669;'>{metrics['Precision']}%</div>
            <div class='stat-label'>Precision</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #7c3aed;'>
            <div class='stat-number' style='color: #7c3aed;'>{metrics['Recall']}%</div>
            <div class='stat-label'>Recall</div>
        </div>
        <div class='stat-card' style='border-top: 4px solid #d97706;'>
            <div class='stat-number' style='color: #d97706;'>{metrics['F1-Score']}%</div>
            <div class='stat-label'>F1-Score</div>
        </div>
    </div>
    <div style='margin-top: 15px; display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;'>
        <span class='badge' style='background: #dcfce7; color: #166534;'>True Positives (TP): {metrics['TP']}</span>
        <span class='badge' style='background: #fee2e2; color: #991b1b;'>False Positives (FP): {metrics['FP']}</span>
        <span class='badge' style='background: #e0f2fe; color: #075985;'>True Negatives (TN): {metrics['TN']}</span>
        <span class='badge' style='background: #fef3c7; color: #92400e;'>False Negatives (FN): {metrics['FN']}</span>
    </div>
    """

    return kpi_html, fig, results_df


CUSTOM_CSS = """
/* Clean Modern App Theme */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 24px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.app-header h1 {
    color: white !important;
    margin: 0 0 8px 0;
    font-size: 2rem;
    font-weight: 800;
}
.app-header p {
    color: #e0e7ff !important;
    margin: 0;
    font-size: 1.05rem;
}
.tech-badges {
    margin-top: 12px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.tech-badge {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-box {
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.status-commitment {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
}
.status-non-commitment {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
}
.status-neutral {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    color: #64748b;
    text-align: center;
}
.stats-container {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    margin: 15px 0;
}
.stat-card {
    flex: 1;
    min-width: 130px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-number {
    font-size: 1.85rem;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 5px;
}
.badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.section-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 15px;
}
"""

with gr.Blocks(title="Commitment & Deadline Extraction") as demo:
    # Header Banner
    gr.HTML("""
    <div class='app-header'>
        <h1> Commitment & Deadline Extraction System</h1>
        <p>Extract commitments, responsible assignees, actionable tasks, and target deadlines from everyday communication.</p>
        <div class='tech-badges'>
            <span class='tech-badge'>spaCy 3.8</span>
            <span class='tech-badge'>NLTK 3.10</span>
            <span class='tech-badge'>Regular Expressions</span>
            <span class='tech-badge'>Pandas</span>
            <span class='tech-badge'>Gradio 6.31</span>
        </div>
    </div>
    """)

    # Main Tabs
    with gr.Tabs():
        # TAB 1: Single Sentence Quick Inspector
        with gr.Tab(" Single Sentence Inspector", id="tab_single"):
            with gr.Row():
                with gr.Column(scale=5):
                    single_input = gr.Textbox(
                        label="Input Sentence",
                        placeholder="Type or paste a sentence here (e.g., 'Ahmed promised to send the project report tomorrow.')...",
                        lines=3,
                        value="Ahmed promised to send the project report tomorrow."
                    )
                    
                    with gr.Row():
                        btn_analyze_single = gr.Button(" Extract Information", variant="primary")
                        btn_clear_single = gr.Button(" Clear", variant="secondary")

                    gr.Markdown("####  Quick Examples:")
                    gr.Examples(
                        examples=SINGLE_EXAMPLES,
                        inputs=single_input,
                        label="Click any example to load it:"
                    )

                    with gr.Accordion(" Extraction Options", open=False):
                        opt_natural = gr.Checkbox(label="Natural Action Word Order (e.g. 'submit the NLP assignment')", value=True)
                        opt_extended_deadlines = gr.Checkbox(label="Extended Deadline Patterns (e.g. 'on 12 November', 'by 5pm')", value=True)
                        opt_resolve_pronouns = gr.Checkbox(label="Detect Speaker Pronouns (e.g. 'I' as Speaker)", value=True)

                with gr.Column(scale=5):
                    status_output = gr.HTML()
                    
                    with gr.Row():
                        out_person = gr.Textbox(label=" Responsible Person", interactive=False)
                        out_action = gr.Textbox(label=" Action / Task", interactive=False)
                    with gr.Row():
                        out_deadline = gr.Textbox(label=" Deadline", interactive=False)
                        out_trigger = gr.Textbox(label=" Trigger Matched", interactive=False)

                    with gr.Accordion(" Linguistic & Grammatical Breakdown", open=True):
                        gr.Markdown("**Token POS & Dependency Parsing (spaCy):**")
                        out_tokens_df = gr.DataFrame(headers=["Token", "POS", "Lemma", "Dependency"], interactive=False)
                        gr.Markdown("**Named Entities Detected:**")
                        out_entities_df = gr.DataFrame(headers=["Entity", "Label"], interactive=False)

            # Tab 1 Events
            btn_analyze_single.click(
                fn=analyze_single_sentence,
                inputs=[single_input, opt_natural, opt_extended_deadlines, opt_resolve_pronouns],
                outputs=[status_output, out_person, out_action, out_deadline, out_trigger, out_tokens_df, out_entities_df]
            )
            btn_clear_single.click(
                fn=lambda: ("", "", "", "", "", pd.DataFrame(), pd.DataFrame(), "<div class='status-box status-neutral'>Please enter a sentence above to analyze.</div>"),
                outputs=[single_input, out_person, out_action, out_deadline, out_trigger, out_tokens_df, out_entities_df, status_output]
            )

        # TAB 2: Document / Multi-Sentence Analyzer
        with gr.Tab(" Document & Batch Analyzer", id="tab_doc"):
            with gr.Row():
                with gr.Column(scale=5):
                    doc_input = gr.Textbox(
                        label="Email, Meeting Minutes, or Multi-sentence Message",
                        placeholder="Paste multi-line text here...",
                        lines=10,
                        value=SAMPLE_DOCUMENT
                    )
                    
                    with gr.Row():
                        btn_analyze_doc = gr.Button(" Analyze Document", variant="primary")
                        btn_clear_doc = gr.Button(" Clear", variant="secondary")

                    with gr.Accordion(" Upload File (.txt or .csv)", open=False):
                        file_upload = gr.File(label="Upload Document", file_types=[".txt", ".csv"])

                with gr.Column(scale=5):
                    doc_stats_html = gr.HTML()
                    
                    with gr.Row():
                        filter_dropdown = gr.Dropdown(
                            choices=["All Sentences", "Commitments Only", "Non-Commitments Only"],
                            value="All Sentences",
                            label="Filter Table View"
                        )
                        download_file = gr.File(label="Download Structured CSV", interactive=False)

            gr.Markdown("###  Extracted Commitments Table")
            doc_df_output = gr.DataFrame(
                headers=["Sentence", "Commitment Detected", "Trigger", "Responsible Person", "Action", "Deadline"],
                interactive=False,
                wrap=True
            )

            # Tab 2 Events
            btn_analyze_doc.click(
                fn=analyze_document,
                inputs=[doc_input, filter_dropdown, opt_natural, opt_extended_deadlines, opt_resolve_pronouns],
                outputs=[doc_df_output, doc_stats_html, download_file]
            )
            filter_dropdown.change(
                fn=analyze_document,
                inputs=[doc_input, filter_dropdown, opt_natural, opt_extended_deadlines, opt_resolve_pronouns],
                outputs=[doc_df_output, doc_stats_html, download_file]
            )
            file_upload.change(
                fn=handle_file_upload,
                inputs=[file_upload, opt_natural, opt_extended_deadlines, opt_resolve_pronouns],
                outputs=[doc_input, doc_df_output, doc_stats_html, download_file]
            )
            btn_clear_doc.click(
                fn=lambda: ("", pd.DataFrame(), "", None),
                outputs=[doc_input, doc_df_output, doc_stats_html, download_file]
            )

        # TAB 3: Model Evaluation & Benchmark
        with gr.Tab(" Evaluation & Benchmark", id="tab_eval"):
            gr.Markdown("""
            ###  Performance Evaluation on Notebook Benchmark
            This evaluation runs the detection pipeline across the 20 benchmark sentences used in `task.ipynb`.
            It calculates Accuracy, Precision, Recall, F1-Score, and displays the Confusion Matrix.
            """)
            
            btn_run_eval = gr.Button(" Run Evaluation Benchmark", variant="primary")
            eval_stats_html = gr.HTML()
            
            with gr.Row():
                with gr.Column(scale=5):
                    eval_plot = gr.Plot(label="Performance Metrics Chart")
                with gr.Column(scale=5):
                    gr.Markdown("####  Detailed Benchmark Results (20 Sentences):")
                    eval_df = gr.DataFrame(interactive=False, wrap=True)

            btn_run_eval.click(
                fn=get_evaluation_results,
                outputs=[eval_stats_html, eval_plot, eval_df]
            )

        # TAB 4: Grammar & Patterns Reference
        with gr.Tab(" Pattern Library & Rules", id="tab_rules"):
            gr.Markdown("###  Configured Rules and Grammar Patterns")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  Commitment Trigger Patterns:")
                    patterns_df = pd.DataFrame({
                        "Pattern": nlp_engine.BASE_COMMITMENT_PATTERNS,
                        "Example Usage": [
                            "I will submit the assignment",
                            "I'll take care of it",
                            "They will deliver the goods",
                            "Ahmed promised to send it",
                            "We promise to assist you",
                            "Sara needs to finish this",
                            "I need to upload files",
                            "You must attend the seminar",
                            "We have to prepare slides",
                            "She has to review code",
                            "Bilal should verify data"
                        ]
                    })
                    gr.DataFrame(patterns_df, interactive=False)

                with gr.Column():
                    gr.Markdown("####  Deadline Regular Expressions:")
                    deadlines_df = pd.DataFrame({
                        "Pattern": nlp_engine.BASE_DEADLINE_PATTERNS,
                        "Matched Examples": [
                            "by Friday, by next Monday",
                            "within 3 days, within two weeks",
                            "before Monday, before next Friday",
                            "on Monday, on next Sunday",
                            "today, tomorrow, tonight",
                            "next week, next month, next year",
                            "in 3 days, in 2 weeks",
                            "by 12/05/2026, by 5-10",
                            "by 15 November, by 20 October"
                        ]
                    })
                    gr.DataFrame(deadlines_df, interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  Monitored Action Verbs:")
                    verbs_str = ", ".join([f"`{v}`" for v in nlp_engine.ACTION_VERBS])
                    gr.Markdown(f"Verbs detected via lemma matching: {verbs_str}")

    # Initialize Tab 1 and Tab 3 on initial page load
    demo.load(
        fn=analyze_single_sentence,
        inputs=[single_input, opt_natural, opt_extended_deadlines, opt_resolve_pronouns],
        outputs=[status_output, out_person, out_action, out_deadline, out_trigger, out_tokens_df, out_entities_df]
    )
    demo.load(
        fn=get_evaluation_results,
        outputs=[eval_stats_html, eval_plot, eval_df]
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True, css=CUSTOM_CSS)
    except OSError:
        # Fallback to any available port if the specified port is already in use
        print(f"Port {port} is occupied. Finding an available port...")
        demo.launch(server_name="127.0.0.1", inbrowser=True, css=CUSTOM_CSS)

