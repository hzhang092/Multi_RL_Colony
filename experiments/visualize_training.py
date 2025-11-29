import re
import matplotlib.pyplot as plt
import os



"""Training log visualization utility.

Supports plotting either by raw environment steps or by update number.
Since the training script logs only every 10 updates, the update axis will be sparse
but useful for comparing runs with different steps_per_update settings.

Set X_AXIS_MODE to 'steps' or 'updates'.
"""

# Configuration toggle (no argparse per user request)
X_AXIS_MODE = 'steps'  # change to 'updates' to use update number as x-axis
FILENAME = "ppo_train_20251129-095445.txt"  # hardcoded log file path

def parse_log_file(filepath):
    data = {
        'steps': [],
        'updates': [],
        'avg_reward': [],
        'p_loss': [],
        'v_loss': [],
        'entropy': [],
        'avg_return': [],
        'avg_num_cells': [],
        'invalid_divides': []
    }

    # Regex patterns
    patterns = {
        'update': r'Update\s+(\d+)/(?:\d+)',
        'steps': r'Steps:\s+(\d+)',
        'avg_reward': r'AVG Reward:\s+([-\d.]+)',
        'p_loss': r'P_Loss:\s+([-\d.]+)',
        'v_loss': r'V_Loss:\s+([-\d.]+)',
        'entropy': r'Entropy:\s+([-\d.]+)',
        'avg_return': r'Avg_Return:\s+([-\d.]+)',
        'avg_num_cells': r'Avg Num Cells:\s+(\d+)',
        'invalid_divides': r'Invalid_Divides%:\s+([-\d.]+)'
    }

    with open(filepath, 'r') as f:
        for line in f:
            if "Update" in line and "|" in line:
                try:
                    # Extract data using regex
                    extracted = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            val = float(match.group(1))
                            if key in ['steps', 'avg_num_cells', 'update']:
                                val = int(val)
                            extracted[key] = val
                    # Require steps for legacy axis; still store update if present
                    if 'steps' in extracted:
                        data['steps'].append(extracted['steps'])
                        if 'update' in extracted:
                            data['updates'].append(extracted['update'])
                        
                        # Append other found metrics, or None/NaN if missing (though usually they are all present)
                        for key in ['avg_reward', 'p_loss', 'v_loss', 'entropy', 'avg_return', 'avg_num_cells', 'invalid_divides']:
                            if key in extracted:
                                data[key].append(extracted[key])
                            elif len(data[key]) < len(data['steps']):
                                # If metric is missing for this step but we added a step, append None or last value
                                # For simplicity, let's just not append and handle length mismatch if it occurs, 
                                # but in this structured log, they usually appear together.
                                # Better approach: ensure lists are same length
                                pass

                except Exception as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(e)
                    continue
    
    # Ensure all lists are same length as steps
    length = len(data['steps'])
    for key in data:
        if len(data[key]) < length:
            # Truncate steps to match shortest data or pad? 
            # Let's just truncate everything to the minimum length found to be safe, 
            # or filter out incomplete records.
            # Given the log format, it's likely all or nothing.
            pass
            
    return data

def plot_training_data(data, filename):
    steps = data['steps']
    updates = data['updates']
    if not steps:
        print("No data found to plot.")
        return

    if X_AXIS_MODE not in ('steps', 'updates'):
        print(f"Invalid X_AXIS_MODE '{X_AXIS_MODE}', defaulting to 'steps'.")
        x_vals = steps
        x_label = 'Steps'
    elif X_AXIS_MODE == 'updates' and updates:
        x_vals = updates
        x_label = 'Update'
    else:
        x_vals = steps
        x_label = 'Steps'

    # Create a figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle(f'Training Metrics ({x_label} axis): {filename}', fontsize=16)
    
    # Helper to plot
    def plot_metric(ax, x, y, label, color, ylabel):
        if len(y) == len(x):
            ax.plot(x, y, label=label, color=color)
            ax.set_title(label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Data mismatch', ha='center', transform=ax.transAxes)

    plot_metric(axes[1, 0], x_vals, data['p_loss'], 'P_Loss', 'red', 'Loss')
    plot_metric(axes[0, 0], x_vals, data['avg_reward'], 'AVG Reward', 'blue', 'Reward')
    plot_metric(axes[1, 1], x_vals, data['v_loss'], 'V_Loss', 'orange', 'Loss')
    plot_metric(axes[0, 1], x_vals, data['avg_return'], 'Avg Return', 'green', 'Return')
    plot_metric(axes[2, 0], x_vals, data['entropy'], 'Entropy', 'purple', 'Entropy')
    plot_metric(axes[2, 1], x_vals, data['avg_num_cells'], 'Avg Num Cells', 'cyan', 'Count')
    plot_metric(axes[3, 0], x_vals, data['invalid_divides'], 'Invalid Divides %', 'brown', 'Percentage')
    
    # Hide the last empty subplot if unused
    axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
    plt.show()

if __name__ == "__main__":
    # Hardcoded log file path
    logfile = os.path.join("logs", FILENAME)

    if not os.path.exists(logfile):
        print(f"File not found: {logfile}")
    else:
        data = parse_log_file(logfile)
        plot_training_data(data, os.path.basename(logfile))
