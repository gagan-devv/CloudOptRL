"""
Interactive Gradio UI for Cloud Resource Allocation RL Environment.

This module provides a web-based interface for manually interacting with the
cloud resource allocation environment. Users can observe system metrics and
take actions to learn about system dynamics before training an RL agent.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from env.environment import CloudResourceEnv
from env.grader import EpisodeGrader


# Initialize environment and grader
env = CloudResourceEnv()
grader = EpisodeGrader()

# Global state for UI
current_state = None
episode_active = False
cpu_history = []
resource_history = []
step_history = []


def create_cpu_plot():
    """Create CPU utilization plot over time."""
    if not cpu_history:
        # Return empty plot if no data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel('Step')
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title('CPU Utilization Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_history, cpu_history, marker='o', linewidth=2, markersize=4, color='#2E86AB')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Target Max (70%)')
    ax.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Target Min (40%)')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical (95%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title('CPU Utilization Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig


def create_resource_plot():
    """Create resource allocation plot over time."""
    if not resource_history:
        # Return empty plot if no data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel('Step')
        ax.set_ylabel('Allocated Resources')
        ax.set_title('Resource Allocation Over Time')
        ax.grid(True, alpha=0.3)
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_history, resource_history, marker='s', linewidth=2, markersize=4, color='#A23B72')
    ax.set_xlabel('Step')
    ax.set_ylabel('Allocated Resources')
    ax.set_title('Resource Allocation Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def reset_environment() -> tuple:
    """
    Reset the environment to start a new episode.
    
    Returns:
        Tuple of display values for all UI components
    """
    global current_state, episode_active, cpu_history, resource_history, step_history
    
    # Reset environment
    current_state = env.reset()
    episode_active = True
    
    # Clear history
    cpu_history = []
    resource_history = []
    step_history = []
    
    # Extract state components
    cpu_util = current_state[0]
    memory_util = current_state[1]
    request_rate = int(current_state[2])
    allocated_resources = int(current_state[3])
    
    # Initialize history with first state
    cpu_history.append(cpu_util)
    resource_history.append(allocated_resources)
    step_history.append(0)
    
    # Calculate initial latency
    latency = request_rate / max(allocated_resources, 1)
    
    # Determine system status
    status = get_system_status(cpu_util, memory_util)
    
    return (
        f"{cpu_util:.2f}%",
        f"{memory_util:.2f}%",
        str(request_rate),
        str(allocated_resources),
        f"{latency:.2f} ms",  # latency display
        "0.00",  # cumulative reward
        "0",  # step count
        status,
        "",  # episode completion message
        gr.update(interactive=True),  # increase button
        gr.update(interactive=True),  # decrease button
        gr.update(interactive=True),  # maintain button
        create_cpu_plot(),  # CPU plot
        create_resource_plot()  # Resource plot
    )


def get_system_status(cpu_util: float, memory_util: float) -> str:
    """
    Determine system status based on utilization metrics.
    
    Args:
        cpu_util: CPU utilization percentage
        memory_util: Memory utilization percentage
    
    Returns:
        Status string indicating system health
    """
    if cpu_util > 95.0 or memory_util > 95.0:
        return "🔴 UNSTABLE - Critical utilization!"
    elif cpu_util > 70.0 or memory_util > 70.0:
        return "🟡 WARNING - High utilization"
    elif cpu_util < 35.0 and memory_util < 35.0:
        return "🟢 OVER-PROVISIONED - Excess resources"
    else:
        return "🟢 NORMAL - Optimal operation"


def execute_action(action: int) -> tuple:
    """
    Execute an action in the environment and update display.
    
    Args:
        action: Action to execute (0: decrease, 1: maintain, 2: increase)
    
    Returns:
        Tuple of display values for all UI components
    """
    global current_state, episode_active, cpu_history, resource_history, step_history
    
    if not episode_active:
        return get_inactive_episode_display()
    
    try:
        # Execute step
        observation, reward, done, info = env.step(action)
        current_state = observation
        
        # Extract state components
        cpu_util = observation[0]
        memory_util = observation[1]
        request_rate = int(observation[2])
        allocated_resources = int(observation[3])
        
        # Update history
        cpu_history.append(cpu_util)
        resource_history.append(allocated_resources)
        step_history.append(info['step'])
        
        # Get cumulative reward and step count
        cumulative_reward = info['cumulative_reward']
        step_count = info['step']
        latency = info['latency']
        
        # Determine system status
        status = get_system_status(cpu_util, memory_util)
        
        # Check if episode terminated
        if done:
            episode_active = False
            completion_msg = handle_episode_completion()
            
            return (
                f"{cpu_util:.2f}%",
                f"{memory_util:.2f}%",
                str(request_rate),
                str(allocated_resources),
                f"{latency:.2f} ms",  # latency display
                f"{cumulative_reward:.2f}",
                str(step_count),
                status,
                completion_msg,
                gr.update(interactive=False),  # disable increase button
                gr.update(interactive=False),  # disable decrease button
                gr.update(interactive=False),  # disable maintain button
                create_cpu_plot(),  # CPU plot
                create_resource_plot()  # Resource plot
            )
        else:
            return (
                f"{cpu_util:.2f}%",
                f"{memory_util:.2f}%",
                str(request_rate),
                str(allocated_resources),
                f"{latency:.2f} ms",  # latency display
                f"{cumulative_reward:.2f}",
                str(step_count),
                status,
                "",  # no completion message yet
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                create_cpu_plot(),  # CPU plot
                create_resource_plot()  # Resource plot
            )
    
    except Exception as e:
        # Handle errors gracefully
        return get_error_display(str(e))


def handle_episode_completion() -> str:
    """
    Handle episode completion and generate grading report.
    
    Returns:
        Formatted completion message with grading results
    """
    # Grade the episode
    grading_result = grader.grade_episode(
        env.episode_states,
        env.episode_actions,
        env.episode_rewards
    )
    
    # Format completion message
    status_emoji = "✅" if grading_result['passed'] else "❌"
    
    message = f"""
    🏁 **Episode Complete!**
    
    {status_emoji} **Status**: {'PASSED' if grading_result['passed'] else 'FAILED'}
    
    **Performance Metrics:**
    - Final Score: {grading_result['final_score']:.3f}
    - Average Reward: {grading_result['avg_reward']:.3f}
    - Stability Score: {grading_result['stability_score']:.3f}
    - Efficiency Score: {grading_result['efficiency_score']:.3f}
    
    **System Metrics:**
    - Average CPU: {grading_result['avg_cpu']:.2f}%
    - Average Memory: {grading_result['avg_memory']:.2f}%
    - Average Latency: {grading_result['avg_latency']:.2f} ms
    
    **Episode Stats:**
    - Total Steps: {len(env.episode_rewards)}
    - Cumulative Reward: {env.cumulative_reward:.2f}
    
    Click **Reset** to start a new episode!
    """
    
    return message


def get_inactive_episode_display() -> tuple:
    """
    Get display values when no episode is active.
    
    Returns:
        Tuple of display values indicating inactive state
    """
    return (
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A",  # latency
        "0.00",
        "0",
        "⚪ No active episode",
        "Click Reset to start a new episode",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        create_cpu_plot(),
        create_resource_plot()
    )


def get_error_display(error_msg: str) -> tuple:
    """
    Get display values when an error occurs.
    
    Args:
        error_msg: Error message to display
    
    Returns:
        Tuple of display values with error information
    """
    return (
        "ERROR",
        "ERROR",
        "ERROR",
        "ERROR",
        "ERROR",  # latency
        "0.00",
        "0",
        "🔴 ERROR",
        f"⚠️ Error: {error_msg}\n\nClick Reset to recover.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        create_cpu_plot(),
        create_resource_plot()
    )


# Action handler functions
def handle_increase() -> tuple:
    """Handle increase action button click."""
    return execute_action(CloudResourceEnv.ACTION_INCREASE)


def handle_decrease() -> tuple:
    """Handle decrease action button click."""
    return execute_action(CloudResourceEnv.ACTION_DECREASE)


def handle_maintain() -> tuple:
    """Handle maintain action button click."""
    return execute_action(CloudResourceEnv.ACTION_MAINTAIN)


def handle_run_multiple_steps(num_steps: int, action_choice: str) -> tuple:
    """
    Execute multiple steps with the same action.
    
    Args:
        num_steps: Number of steps to execute
        action_choice: Action to repeat ("Increase", "Decrease", or "Maintain")
    
    Returns:
        Tuple of display values for all UI components
    """
    global episode_active
    
    if not episode_active:
        return get_inactive_episode_display()
    
    # Map action choice to action integer
    action_map = {
        "Increase": CloudResourceEnv.ACTION_INCREASE,
        "Decrease": CloudResourceEnv.ACTION_DECREASE,
        "Maintain": CloudResourceEnv.ACTION_MAINTAIN
    }
    
    action = action_map.get(action_choice, CloudResourceEnv.ACTION_MAINTAIN)
    
    # Execute multiple steps
    for _ in range(int(num_steps)):
        if not episode_active:
            break
        result = execute_action(action)
    
    return result


# Create Gradio interface
with gr.Blocks(title="Cloud Resource Allocation RL") as demo:
    gr.Markdown("""
    # 🌐 Cloud Resource Allocation RL Environment
    
    Manually control cloud resource allocation and observe system dynamics.
    Learn how actions affect CPU, memory, and system stability before training an RL agent.
    
    **Actions:**
    - **Increase**: Add 1 server instance
    - **Decrease**: Remove 1 server instance (minimum 1)
    - **Maintain**: Keep current allocation
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 System Metrics")
            cpu_display = gr.Textbox(label="CPU Utilization", value="N/A", interactive=False)
            memory_display = gr.Textbox(label="Memory Utilization", value="N/A", interactive=False)
            request_display = gr.Textbox(label="Request Rate", value="N/A", interactive=False)
            resource_display = gr.Textbox(label="Allocated Resources", value="N/A", interactive=False)
            latency_display = gr.Textbox(label="Latency", value="N/A", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Episode Progress")
            reward_display = gr.Textbox(label="Cumulative Reward", value="0.00", interactive=False)
            step_display = gr.Textbox(label="Step Count", value="0", interactive=False)
            status_display = gr.Textbox(label="System Status", value="⚪ No active episode", interactive=False)
    
    # Visualization section
    gr.Markdown("### 📉 Real-Time Metrics")
    with gr.Row():
        cpu_plot = gr.Plot(label="CPU Utilization Over Time")
        resource_plot = gr.Plot(label="Resource Allocation Over Time")
    
    gr.Markdown("### 🎮 Actions")
    with gr.Row():
        increase_btn = gr.Button("⬆️ Increase Resources", variant="primary", interactive=False)
        maintain_btn = gr.Button("➡️ Maintain Resources", variant="secondary", interactive=False)
        decrease_btn = gr.Button("⬇️ Decrease Resources", variant="stop", interactive=False)
    
    # Multi-step execution
    gr.Markdown("### ⚡ Run Multiple Steps")
    with gr.Row():
        num_steps_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Steps")
        action_dropdown = gr.Dropdown(
            choices=["Increase", "Maintain", "Decrease"],
            value="Maintain",
            label="Action to Repeat"
        )
        run_multi_btn = gr.Button("▶️ Run Multiple Steps", variant="secondary")
    
    gr.Markdown("### 🔄 Episode Control")
    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
    
    completion_display = gr.Markdown("")
    
    # Wire up event handlers
    outputs = [
        cpu_display,
        memory_display,
        request_display,
        resource_display,
        latency_display,
        reward_display,
        step_display,
        status_display,
        completion_display,
        increase_btn,
        decrease_btn,
        maintain_btn,
        cpu_plot,
        resource_plot
    ]
    
    reset_btn.click(fn=reset_environment, outputs=outputs)
    increase_btn.click(fn=handle_increase, outputs=outputs)
    decrease_btn.click(fn=handle_decrease, outputs=outputs)
    maintain_btn.click(fn=handle_maintain, outputs=outputs)
    run_multi_btn.click(
        fn=handle_run_multiple_steps,
        inputs=[num_steps_input, action_dropdown],
        outputs=outputs
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Optimal Range**: Keep CPU and memory between 40-70% for best rewards
    - **Over-Provisioning**: Too many resources waste money (negative rewards)
    - **Under-Provisioning**: Too few resources risk system instability (negative rewards)
    - **Stochastic Dynamics**: Request rate fluctuates randomly each step
    - **Multi-Step Mode**: Use "Run Multiple Steps" to quickly test action sequences
    """)


# Create FastAPI app at module level (required for OpenEnv)
app = FastAPI()

@app.post("/reset")
async def reset_endpoint():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse(url="/ui")

app = gr.mount_gradio_app(app, demo, path="/ui")


def main():
    """Main entry point for the server application."""
    import uvicorn
    import logging
    
    # Suppress uvicorn access logs to keep stdout clean for inference
    # Only errors will be logged to stderr
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.disabled = True
    
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="error")


if __name__ == "__main__":
    main()
