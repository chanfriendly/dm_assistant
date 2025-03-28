# scripts/visualize_log.py
import re
import matplotlib.pyplot as plt
import argparse

def parse_log(log_content):
    """Parses training log content to extract train and val losses."""
    train_losses = []
    val_losses = []

    # Regex for train loss: Iter <iter>: Train loss <loss>, ...
    train_pattern = re.compile(r"Iter (\d+): Train loss ([\d.]+),")
    # Regex for val loss: Iter <iter>: Val loss <loss>, ...
    val_pattern = re.compile(r"Iter (\d+): Val loss ([\d.]+),")

    for line in log_content.splitlines():
        train_match = train_pattern.search(line)
        if train_match:
            iteration = int(train_match.group(1))
            loss = float(train_match.group(2))
            # Only add if iter > 0 (ignore potential initial val loss before first train step)
            if iteration > 0:
                 train_losses.append({"iter": iteration, "loss": loss})

        val_match = val_pattern.search(line)
        if val_match:
            iteration = int(val_match.group(1))
            loss = float(val_match.group(2))
            # Store val loss associated with the iteration it was computed AT
            val_losses.append({"iter": iteration, "loss": loss})

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, output_file="loss_curve.png"):
    """Plots train and validation losses."""
    if not train_losses and not val_losses:
        print("No loss data found to plot.")
        return

    plt.figure(figsize=(12, 6))

    if train_losses:
        train_iters = [d["iter"] for d in train_losses]
        train_loss_vals = [d["loss"] for d in train_losses]
        plt.plot(train_iters, train_loss_vals, label="Train Loss", alpha=0.8)

    if val_losses:
        val_iters = [d["iter"] for d in val_losses]
        val_loss_vals = [d["loss"] for d in val_losses]
        plt.plot(val_iters, val_loss_vals, label="Validation Loss", marker='o', linestyle='-') # Markers help see points

        # Find and mark the minimum validation loss
        if val_loss_vals:
            min_val_loss = min(val_loss_vals)
            min_val_iter = val_iters[val_loss_vals.index(min_val_loss)]
            plt.scatter([min_val_iter], [min_val_loss], color='red', zorder=5, label=f'Min Val Loss: {min_val_loss:.3f} at Iter {min_val_iter}')


    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Loss curve saved to {output_file}")
    # plt.show() # Uncomment to display the plot directly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot training logs.")
    parser.add_argument("log_file", type=str, help="Path to the training log file.")
    parser.add_argument("-o", "--output", type=str, default="loss_curve.png", help="Path to save the plot image.")
    args = parser.parse_args()

    try:
        with open(args.log_file, 'r') as f:
            log_content = f.read()
        train_data, val_data = parse_log(log_content)
        plot_losses(train_data, val_data, args.output)
    except FileNotFoundError:
        print(f"Error: Log file not found at {args.log_file}")
    except Exception as e:
        print(f"An error occurred: {e}")