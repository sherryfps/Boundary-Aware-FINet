"""FINet Vision AI: domain-isolated real segmentation and research comparison."""

import argparse
from pathlib import Path
import os

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import torch
from src.app_service import AppService, ModelRegistry, BASELINE, BOUNDARY, CAMOUFLAGE, INDUSTRIAL, DOMAINS

CSS = """
.gradio-container {max-width:1280px!important; font-family:'Segoe UI',system-ui,sans-serif!important;
  --body-background-fill:oklch(0.97 0.006 70); --block-background-fill:oklch(0.99 0.004 70);
  --body-text-color:oklch(0.26 0.008 65); --border-color-primary:oklch(0.85 0.009 70);}
#project-title h1 {font-size:2rem;line-height:1.2;font-weight:650;letter-spacing:-0.025em;margin-bottom:8px;}
#project-title p {max-width:72ch;color:oklch(0.47 0.008 65);}
button.primary {background:oklch(0.48 0.11 40)!important;border-color:oklch(0.48 0.11 40)!important;color:oklch(0.99 0.004 70)!important;}
button:focus-visible {outline:3px solid oklch(0.48 0.11 40);outline-offset:3px;}
.result-summary {font-variant-numeric:tabular-nums;}
footer {display:none!important;}
"""


def app_theme():
    # Keep contrast consistent when the browser requests a dark color scheme.
    colors = {'body_background_fill': 'oklch(0.97 0.006 70)',
              'body_text_color': 'oklch(0.26 0.008 65)',
              'body_text_color_subdued': 'oklch(0.47 0.008 65)',
              'block_background_fill': 'oklch(0.99 0.004 70)',
              'block_label_text_color': 'oklch(0.26 0.008 65)',
              'block_title_text_color': 'oklch(0.26 0.008 65)',
              'input_background_fill': 'oklch(0.99 0.004 70)',
              'button_secondary_background_fill': 'oklch(0.93 0.008 70)',
              'button_secondary_text_color': 'oklch(0.26 0.008 65)',
              'background_fill_primary': 'oklch(0.99 0.004 70)',
              'background_fill_secondary': 'oklch(0.94 0.006 70)',
              'table_even_background_fill': 'oklch(0.99 0.004 70)',
              'table_odd_background_fill': 'oklch(0.94 0.006 70)',
              'border_color_primary': 'oklch(0.85 0.009 70)'}
    tokens = {key: value for name, value in colors.items() for key in (name, name + '_dark')}
    return gr.themes.Base(font=['Segoe UI', 'system-ui', 'sans-serif']).set(**tokens)


def build_app(service):
    def segment(path, name, threshold, target, domain):
        try:
            state = service.predict(path, name, domain)
            return (*service.render(state, threshold, target), state)
        except (ValueError, OSError, RuntimeError) as error:
            return None, None, None, None, None, f'**Unable to segment:** {error}', None, None

    def redraw(state, threshold, target):
        if state is None:
            return (None, None, None, None, None, 'Upload an image and run segmentation.', None)
        try:
            return service.render(state, threshold, target)
        except (ValueError, OSError, RuntimeError) as error:
            return None, None, None, None, None, f'**Unable to update results:** {error}', None

    def compare(path, threshold, target, domain):
        try:
            left, right = service.compare(path, threshold, target, domain)
            return left[3], right[3], left[4], right[4], left[5] + '\n\n' + right[5], left[6], right[6]
        except (ValueError, OSError, RuntimeError) as error:
            return None, None, None, None, f'**Unable to compare:** {error}', None, None

    with gr.Blocks(title='FINet Vision AI') as demo:
        gr.Markdown('# FINet Vision AI\nInspect subtle objects and surface defects with domain-specific segmentation models.', elem_id='project-title')
        domain = gr.Radio(list(DOMAINS), value=CAMOUFLAGE, label='Application domain')
        with gr.Accordion('Model availability', open=True):
            model_status = gr.Markdown(service.registry.describe())
            refresh = gr.Button('Refresh checkpoints', size='sm')
        refresh.click(service.registry.describe, inputs=domain, outputs=model_status, api_name=False)
        with gr.Tabs():
            with gr.Tab('Segment an image'):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        upload = gr.Image(label='Upload a photograph', type='filepath', sources=['upload'], height=280, format='png')
                        available = service.registry.available(CAMOUFLAGE)
                        model = gr.Radio(available or [BASELINE], value=BASELINE if BASELINE in available or not available else available[0], label='Model weights')
                        threshold = gr.Slider(0, 1, value=0.5, step=0.01, label='Foreground threshold')
                        run = gr.Button('Analyze Image', variant='primary')
                        with gr.Accordion('Optional ground-truth evaluation', open=False):
                            ground_truth = gr.File(label='Matching object mask (PNG)', file_types=['.png'], type='filepath')
                            gr.Markdown('Use a mask for this exact photograph and pixel dimensions. Without it, no accuracy score is calculated.')
                    with gr.Column(scale=2, min_width=400):
                        with gr.Row():
                            original = gr.Image(label='Original image', interactive=False, height=280)
                            overlay = gr.Image(label='Camouflaged Object', interactive=False, height=280)
                        with gr.Row():
                            gray = gr.Image(label='Grayscale probability', interactive=False, height=210)
                            binary = gr.Image(label='Binary mask', interactive=False, height=210)
                        with gr.Accordion('Boundary visualization', open=False):
                            boundary = gr.Image(label='Predicted boundary', interactive=False, height=210)
                        summary = gr.Markdown('Upload an image, select an available model, then run segmentation.', elem_classes='result-summary')
                        download = gr.File(label='Download masks, overlay and inference report', interactive=False)
                state = gr.State(None)
                result_outputs = [original, gray, binary, overlay, boundary, summary, download]
                run.click(segment, [upload, model, threshold, ground_truth, domain], result_outputs + [state], api_name='segment',
                          concurrency_limit=1, concurrency_id='finet')
                threshold.release(redraw, [state, threshold, ground_truth], result_outputs, api_name=False,
                                  concurrency_limit=1, concurrency_id='finet')
                def clear_result():
                    return None, None, None, None, None, 'Run segmentation for the selected image and model.', None, None
                for control in (upload, model, ground_truth):
                    control.change(clear_result, outputs=result_outputs + [state], api_name=False)
            with gr.Tab('Research Comparison'):
                gr.Markdown('Both models process the same photograph at the same threshold. Both trained checkpoints are required.')
                with gr.Row():
                    compare_upload = gr.Image(label='Comparison photograph', type='filepath', sources=['upload'], height=240, format='png')
                    with gr.Column():
                        compare_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label='Shared foreground threshold')
                        compare_target = gr.File(label='Optional matching ground truth (PNG)', file_types=['.png'], type='filepath')
                        compare_button = gr.Button('Compare segmentation', variant='primary')
                with gr.Row():
                    left = gr.Image(label='Original FINet overlay', interactive=False, height=300)
                    right = gr.Image(label='Boundary-Aware FINet overlay', interactive=False, height=300)
                with gr.Row():
                    left_edge = gr.Image(label='Original FINet predicted boundary', interactive=False, height=200)
                    right_edge = gr.Image(label='Boundary-Aware FINet predicted boundary', interactive=False, height=200)
                comparison = gr.Markdown(elem_classes='result-summary')
                with gr.Row():
                    left_download = gr.File(label='Original FINet results', interactive=False)
                    right_download = gr.File(label='Boundary-Aware FINet results', interactive=False)
                comparison_outputs = [left, right, left_edge, right_edge, comparison, left_download, right_download]
                compare_button.click(compare, [compare_upload, compare_threshold, compare_target, domain], comparison_outputs,
                                     api_name='compare', concurrency_limit=1, concurrency_id='finet')
                for control in (compare_upload, compare_threshold, compare_target, domain):
                    control.change(lambda: (None, None, None, None, 'Run comparison with the current inputs.', None, None),
                                   outputs=comparison_outputs, api_name=False)
        def switch_domain(selected):
            choices = service.registry.available(selected)
            return (gr.update(choices=choices or [BASELINE], value=BASELINE if BASELINE in choices or not choices else choices[0]),
                    service.registry.describe(selected),
                    gr.update(label='Predicted Defect Region' if selected == INDUSTRIAL else 'Camouflaged Object', value=None),
                    gr.update(label='Matching defect mask (PNG)' if selected == INDUSTRIAL else 'Matching object mask (PNG)', value=None))
        domain.change(switch_domain, inputs=domain, outputs=[model, model_status, overlay, ground_truth], api_name=False).then(
            clear_result, outputs=result_outputs + [state], api_name=False)
        gr.Markdown('Based on FINet by Liang et al. (IEEE SPL, 2024). Boundary supervision changes training only; both models retain the same architecture.')
        gr.Markdown('Academic prototype. Factory mode uses a custom supervised MVTec AD split under CC BY-NC-SA 4.0. It is not certified for production inspection.')
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='checkpoints/baseline/last.pth')
    parser.add_argument('--boundary', default='checkpoints/boundary/last.pth')
    parser.add_argument('--industrial-baseline', default='checkpoints/industrial/baseline/last.pth')
    parser.add_argument('--industrial-boundary', default='checkpoints/industrial/boundary/last.pth')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(2)
    service = AppService(ModelRegistry(args.baseline, args.boundary, args.device,
                                      args.industrial_baseline, args.industrial_boundary))
    service.output_root.mkdir(parents=True, exist_ok=True)
    demo = build_app(service)
    demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=args.port, share=False,
                                allowed_paths=[str(service.output_root)], show_error=True,
                                css=CSS, theme=app_theme())


if __name__ == '__main__':
    main()
