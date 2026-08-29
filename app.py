import logging
import queue
import threading
import time
import gradio as gr
import pandas as pd
from job_agent_framework import JobAgentFramework
from log_utils import reformat
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)

TABLE_HEADERS = ["Job Description", "Advertised", "Estimated Fair", "Gap", "URL"]


COLUMN_WIDTHS = ["44%", "12%", "14%", "12%", "18%"]


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-24:])
    return f"""
    <div id="scrollContent" style="height: 420px; overflow-y: auto; border: 1px solid #333;
         background-color: #0d0d12; color: #e8e8ec; padding: 12px;
         font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
         font-size: 13px; line-height: 1.5; border-radius: 8px;">
    {output}
    </div>
    """


def setup_logging(log_queue):
 
    logger = logging.getLogger()
    for existing in [h for h in logger.handlers if isinstance(h, QueueHandler)]:
        logger.removeHandler(existing)

    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

body, .gradio-container, .gradio-container *:not(#scrollContent):not(#scrollContent *) {
    color: #e6e6ea !important;
}

.dataframe, .dataframe *, table, table * {
    color: #e6e6ea !important;
    background-color: #1a1a1e !important;
}

h1, h2, h3, .markdown-text strong {
    font-weight: 700 !important;
}
"""


class App:
    def __init__(self):
        self.agent_framework = None

        self._framework_lock = threading.Lock()

    def get_agent_framework(self):
        if not self.agent_framework:
            with self._framework_lock:
                if not self.agent_framework:
                    self.agent_framework = JobAgentFramework()
        return self.agent_framework

    def run(self):

        with gr.Blocks(
            title="Is This Salary Fair?",
            fill_width=True,
        ) as ui:
            log_data = gr.State([])

            def table_for(opps):

                rows = [
                    [
                        opp.posting.job_description[:80] + "...",
                        f"${opp.posting.salary:,.0f}",
                        f"${opp.estimate:,.0f}",
                        f"${opp.gap:,.0f}",
                        opp.posting.url,
                    ]
                    for opp in opps
                ]
                return pd.DataFrame(rows, columns=TABLE_HEADERS)

            def update_output(log_data, log_queue, result_queue):
                try:
                    initial_result = table_for(self.get_agent_framework().memory)
                except Exception:
                    logging.exception("Failed to load initial agent framework / memory")
                    initial_result = table_for([])


                final_result = None
                have_final_result = False
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        current = final_result if have_final_result else initial_result
                        yield log_data, html_for(log_data), current
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            have_final_result = True
                            yield log_data, html_for(log_data), final_result
                        except queue.Empty:
                            if have_final_result:
                                break
                            time.sleep(0.1)

            def placeholder_plot(message="Loading vector DB..."):
                fig = go.Figure()
                fig.update_layout(
                    title=message,
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                )
                return fig

            def get_initial_plot():
                return placeholder_plot()

            def get_plot():
                try:
                    documents, vectors, colors = JobAgentFramework.get_plot_data(max_datapoints=800)
                    fig = go.Figure(
                        data=[
                            go.Scatter3d(
                                x=vectors[:, 0],
                                y=vectors[:, 1],
                                z=vectors[:, 2],
                                mode="markers",
                                marker=dict(size=2.5, color=colors, opacity=0.75),
                            )
                        ]
                    )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="z",
                            aspectmode="manual",
                            aspectratio=dict(x=2.2, y=2.2, z=1),
                            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                        ),
                        height=400,
                        margin=dict(r=5, b=1, l=5, t=2),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    return fig
                except Exception as e:

                    logging.exception("get_plot failed while building the embedding plot")
                    return placeholder_plot(f"Plot failed: {type(e).__name__}")

            def do_run():
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                    except Exception:
                        logging.exception("Agent run failed")
                        result = table_for(self.get_agent_framework().memory)
                    result_queue.put(result)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:26px"><strong>💰 Is This Salary Fair?</strong></div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px;color:#999">'
                    'A fine-tuned LLM specialist + RAG frontier model + custom neural network — combined into an autonomous '
                    'agent that scans real job postings and flags underpaid roles.</div>'
                )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Agent Legend")
                    gr.Markdown(
                        "<span style='color:#ff7800'>■</span> Agent Framework &nbsp;"
                        "<span style='color:#dd0000'>■</span> Specialist &nbsp;"
                        "<span style='color:#0000ee'>■</span> Frontier &nbsp;"
                        "<span style='color:#aa00dd'>■</span> Neural Network &nbsp;"
                        "<span style='color:#00dddd'>■</span> Scanner &nbsp;"
                        "<span style='color:#00dd00'>■</span> Planning &nbsp;"
                        "<span style='color:#87CEEB'>■</span> Messaging"
                    )
            with gr.Row():

                opportunities_dataframe = gr.Dataframe(
                    value=pd.DataFrame(columns=TABLE_HEADERS),
                    headers=TABLE_HEADERS,
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=True,
                    column_widths=COLUMN_WIDTHS,
                    max_height=400,
                    interactive=False,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📡 Live Agent Activity")
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("### 🌐 Job Posting Embedding Space")
                    plot = gr.Plot(value=get_initial_plot(), show_label=False)

            with gr.Row():
                run_button = gr.Button("🚀 Run Agent Scan Now", variant="primary", size="lg")

            def load_plot_after_start():
                return get_plot()

            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )
            ui.load(load_plot_after_start, outputs=[plot])

            run_button.click(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            ).then(load_plot_after_start, outputs=[plot])

            timer = gr.Timer(value=1800, active=True)  # auto re-scan every 30 min
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            ).then(load_plot_after_start, outputs=[plot])

            opportunities_dataframe.select(do_select)


        ui.queue(default_concurrency_limit=4)
        ui.launch(
            share=False,
            inbrowser=True,
            theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
            css=CUSTOM_CSS,
        )


if __name__ == "__main__":
    App().run()