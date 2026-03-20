"""GUI module for the gradio web app."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, List

import gradio as gr

if TYPE_CHECKING:
    from swarm_gpt.core import AppBackend


def padding_column():
    """Create a column with a hidden textbox to add padding to the UI."""
    with gr.Column():
        gr.Textbox(visible=False)


def centered_markdown(text: str) -> gr.Markdown:
    """Create a centered markdown element."""
    md = f'<div align="center"> <font size = "10"> <span style="color:grey">{text}</span>'
    return gr.Markdown(md, visible=False)


def update_visibility(visible_flags: List[bool]) -> Callable:
    """Return a function that yields gradio visibility updates."""
    def gradio_ui_update() -> List[dict]:
        return [gr.update(visible=x) for x in visible_flags]
    return gradio_ui_update


def run_with_bar(backend: AppBackend, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> str:
    """Run the simulation with a progress bar."""
    for key, data, total in backend.simulate():
        if key == "progress":
            if data != total:
                percent = int(data / total)
                progress(percent, desc="Simulation Loading...", total=100)
            else:
                progress(100, desc="Simulation Playing", total=100)
        else:
            return "Simulation Playing!"


def create_ui(backend: AppBackend) -> gr.Blocks:
    """Create the gradio web app."""
    warnings.filterwarnings("ignore", category=UserWarning, message="api_name")

    with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
        gr.Markdown(""" <div align="center"> <font size = "50"> SwarmGPT-Command""")

        # ── Input row ──────────────────────────────────────────────────────────
        with gr.Row():
            padding_column()
            with gr.Column():
                user_prompt_input = gr.Textbox(
                    label="What should the drones do?",
                    placeholder="E.g.: Form a triangle at 1.5m high for 10 seconds...",
                    lines=2,
                )
            with gr.Column():
                prompt_choices = list(backend.choreographer.prompts.keys())
                prompt_mode = gr.Dropdown(
                    choices=prompt_choices,
                    label="Prompt Mode:",
                    value=prompt_choices[0] if prompt_choices else None,
                    visible=False,
                    interactive=True,
                )
            padding_column()

        # ── Status messages ────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                replay_msg  = centered_markdown("Replaying simulation")
                sim_msg     = centered_markdown("Simulating safe choreography")
                choreo_msg  = centered_markdown("LLM is generating choreography")

        # ── Chatbot / message ──────────────────────────────────────────────────
        chatbot = gr.Chatbot(visible=False)
        message = gr.Textbox(label="Enter prompt:", visible=False)

        # ── Progress bar ───────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                progress_bar = gr.Textbox("Progress", visible=False)

        # ── Action buttons ─────────────────────────────────────────────────────
        with gr.Row():
            padding_column()
            with gr.Column():
                replay_sim_button = gr.Button("Replay simulation", visible=False)
                sim_button        = gr.Button("Simulate",          visible=False)
            with gr.Column():
                alter_button = gr.Button("Refine/Modify the choreography", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                new_command_button = gr.Button("New Command", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                start_button      = gr.Button("Send selections to LLM", visible=False)
                deploy_button     = gr.Button("Let the Crazyflies move", visible=False)
                save_preset_button = gr.Button("Save preset",            visible=False)
                show_output = gr.Checkbox(
                    label="Display conversation with LLM",
                    visible=False,
                    value=False,
                    container=True,
                    interactive=True,
                )
            padding_column()

        # ── All UI elements that the reset function must touch ─────────────────
        ALL_RESETABLE = [
            user_prompt_input,   # 0
            start_button,        # 1
            sim_button,          # 2
            alter_button,        # 3
            new_command_button,  # 4
            deploy_button,       # 5
            save_preset_button,  # 6
            chatbot,             # 7
            choreo_msg,          # 8
            show_output,         # 9
            replay_sim_button,   # 10
            sim_msg,             # 11
            replay_msg,          # 12
            progress_bar,        # 13
            message,             # 14
        ]

        def full_reset_ui():
            """Return gradio updates that bring every widget back to initial state."""
            return [
                gr.update(value="", visible=True),   # user_prompt_input  – visible again
                gr.update(visible=True),             # start_button – sempre visibile dopo reset
                gr.update(visible=False),             # sim_button
                gr.update(visible=False),             # alter_button
                gr.update(visible=False),             # new_command_button
                gr.update(visible=False),             # deploy_button
                gr.update(visible=False),             # save_preset_button
                gr.update(visible=False, value=[]),   # chatbot
                gr.update(visible=False),             # choreo_msg
                gr.update(visible=False, value=False),# show_output
                gr.update(visible=False),             # replay_sim_button
                gr.update(visible=False),             # sim_msg
                gr.update(visible=False),             # replay_msg
                gr.update(visible=False, value=""),   # progress_bar
                gr.update(visible=False, value=None), # message
            ]

        # ══════════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        # Show start button only when the user has typed something
        user_prompt_input.change(
            lambda x: gr.update(visible=len(x) > 0),
            inputs=[user_prompt_input],
            outputs=[start_button],
        )

        # ── start_button ───────────────────────────────────────────────────────
        start_button_flow = start_button.click(
            update_visibility([False, False, True, True]),
            [],
            [user_prompt_input, start_button, choreo_msg, show_output],
        )
        start_button_flow = start_button_flow.success(
            backend.initial_prompt,
            inputs=[user_prompt_input],
            outputs=chatbot,
        )
        start_button_flow = start_button_flow.success(
            update_visibility([False, True, True, True, True, True]),
            [],
            [choreo_msg, sim_button, alter_button, new_command_button, deploy_button, save_preset_button],
        )

        # ── alter_button ───────────────────────────────────────────────────────
        alter_button_flow = alter_button.click(
            lambda: gr.update(visible=True, value=None), [], [message]
        )
        alter_button_flow = alter_button_flow.success(
            update_visibility([False, False, False, False, True]),
            [],
            [alter_button, deploy_button, replay_sim_button, sim_button, chatbot],
        )

        # ── show_output checkbox ───────────────────────────────────────────────
        def on_select(evt: gr.SelectData) -> dict:
            return gr.update(visible=evt.value)

        show_output.select(on_select, [], [chatbot])

        # ── message (reprompt) ─────────────────────────────────────────────────
        message_flow = message.submit(
            update_visibility([False, False, True]), [], [sim_msg, replay_msg, choreo_msg]
        )
        message_flow = message_flow.success(backend.reprompt, [message], [chatbot])
        message_flow = message_flow.success(
            update_visibility([False, False, True, True, False]),
            [],
            [alter_button, choreo_msg, sim_button, deploy_button, replay_sim_button],
        )
        message_flow = message_flow.success(
            lambda: gr.update(visible=True, value=None), [], message
        )

        # ── sim_button ─────────────────────────────────────────────────────────
        sim_button_flow = sim_button.click(
            update_visibility([False, False, True, True]),
            [],
            [replay_msg, choreo_msg, sim_msg, progress_bar],
        )
        sim_button_flow = sim_button_flow.success(
            lambda: run_with_bar(backend), outputs=progress_bar
        )
        sim_button_flow = sim_button_flow.success(
            update_visibility([False, False, True, True, True, True, False]),
            [],
            [sim_msg, sim_button, replay_sim_button, alter_button, deploy_button, new_command_button, progress_bar],
        )

        # ── deploy_button ──────────────────────────────────────────────────────
        deploy_button.click(backend.deploy, [], chatbot)

        # ── save_preset_button ─────────────────────────────────────────────────
        save_preset_button.click(backend.save_preset, [], [])

        # ── replay_sim_button ──────────────────────────────────────────────────
        # Replay riusa le splines già calcolate — NON riesegue axswarm.
        # Il visualizzatore parte in background (thread), quindi il click
        # ritorna subito senza bloccare la UI.
        replay_sim_flow = replay_sim_button.click(
            update_visibility([False, False, True, False, False, False]),
            [],
            [sim_msg, choreo_msg, replay_msg,
             replay_sim_button, alter_button, new_command_button],
        )
        replay_sim_flow = replay_sim_flow.success(
            backend.replay,
            inputs=[],
            outputs=[],
        )
        replay_sim_flow = replay_sim_flow.success(
            update_visibility([False, True, True, True]),
            [],
            [replay_msg, replay_sim_button, alter_button, new_command_button],
        )

        # ── new_command_button ─────────────────────────────────────────────────
        # NOTE: single .click() chain – no js reload, no double-binding
        new_command_button.click(
            fn=backend.reset_data,
            inputs=[],
            outputs=[],
        ).success(
            fn=full_reset_ui,
            inputs=[],
            outputs=ALL_RESETABLE,
        )

    return ui