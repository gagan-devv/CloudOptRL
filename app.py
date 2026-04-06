"""
Interactive Gradio UI for Cloud Resource Allocation RL Environment.

This module provides a web-based interface for manually interacting with the
cloud resource allocation environment. Users can observe system metrics and
take actions to learn about system dynamics before training an RL agent.
"""

import gradio as gr
import numpy as np
from env.environment import CloudResourceEnv
from env.grader import EpisodeGrader


# Initialize environment and grader
env = CloudResourceEnv()
grader = EpisodeGrader()

# Global state for UI
current_state = None
episode_active = False


def reset_environment() -> tuple:
    """
    Reset the environment to start a new episode.
    
    Returns:
        Tuple of display values for all UI components
    """
    global current_state, episode_active
    
    # Reset environment
    current_state = env.reset()
    episode_active = True
    
    # Extract state components
    cpu_util = current_state[0]
    memory_util = current_state[1]
    request_rate = int(current_state[2])
    allocated_resources = int(current_state[3])
    
    # Determine system status
    status = get_system_status(cpu_util, memory_util)
    
    return (
        f"{cpu_util:.2f}%",
        f"{memory_util:.2f}%",
        str(request_rate),
        str(allocated_resources),
        "0.00",  # cumulative reward
        "0",  # step count
        status,
        "",  # episode completion message
        gr.update(interactive=True),  # increase button
        gr.update(interactive=True),  # decrease button
        gr.update(interactive=True)   # maintain button
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
    global current_state, episode_active
    
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
        
        # Get cumulative reward and step count
        cumulative_reward = info['cumulative_reward']
        step_count = info['step']
        
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
                f"{cumulative_reward:.2f}",
                str(step_count),
                status,
                completion_msg,
                gr.update(interactive=False),  # disable increase button
                gr.update(interactive=False),  # disable decrease button
                gr.update(interactive=False)   # disable maintain button
            )
        else:
            return (
                f"{cpu_util:.2f}%",
                f"{memory_util:.2f}%",
                str(request_rate),
                str(allocated_resources),
                f"{cumulative_reward:.2f}",
                str(step_count),
                status,
                "",  # no completion message yet
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True)
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
    - Overall Score: {grading_result['score']:.3f}
    - Average Reward: {grading_result['avg_reward']:.3f}
    - Stability Score: {grading_result['stability_score']:.3f}
    - Efficiency Score: {grading_result['efficiency_score']:.3f}
    
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
        "0.00",
        "0",
        "⚪ No active episode",
        "Click Reset to start a new episode",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
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
        "0.00",
        "0",
        "🔴 ERROR",
        f"⚠️ Error: {error_msg}\n\nClick Reset to recover.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
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
        with gr.Column():
            gr.Markdown("### 📊 System Metrics")
            cpu_display = gr.Textbox(label="CPU Utilization", value="N/A", interactive=False)
            memory_display = gr.Textbox(label="Memory Utilization", value="N/A", interactive=False)
            request_display = gr.Textbox(label="Request Rate", value="N/A", interactive=False)
            resource_display = gr.Textbox(label="Allocated Resources", value="N/A", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 📈 Episode Progress")
            reward_display = gr.Textbox(label="Cumulative Reward", value="0.00", interactive=False)
            step_display = gr.Textbox(label="Step Count", value="0", interactive=False)
            status_display = gr.Textbox(label="System Status", value="⚪ No active episode", interactive=False)
    
    gr.Markdown("### 🎮 Actions")
    with gr.Row():
        increase_btn = gr.Button("⬆️ Increase Resources", variant="primary", interactive=False)
        maintain_btn = gr.Button("➡️ Maintain Resources", variant="secondary", interactive=False)
        decrease_btn = gr.Button("⬇️ Decrease Resources", variant="stop", interactive=False)
    
    gr.Markdown("### 🔄 Episode Control")
    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
    
    completion_display = gr.Markdown("")
    
    # Wire up event handlers
    outputs = [
        cpu_display,
        memory_display,
        request_display,
        resource_display,
        reward_display,
        step_display,
        status_display,
        completion_display,
        increase_btn,
        decrease_btn,
        maintain_btn
    ]
    
    reset_btn.click(fn=reset_environment, outputs=outputs)
    increase_btn.click(fn=handle_increase, outputs=outputs)
    decrease_btn.click(fn=handle_decrease, outputs=outputs)
    maintain_btn.click(fn=handle_maintain, outputs=outputs)
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Optimal Range**: Keep CPU and memory between 35-70% for best rewards
    - **Over-Provisioning**: Too many resources waste money (negative rewards)
    - **Under-Provisioning**: Too few resources risk system instability (negative rewards)
    - **Stochastic Dynamics**: Request rate fluctuates randomly each step
    """)


if __name__ == "__main__":
    demo.launch()
