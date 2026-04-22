import time
import functools
import matplotlib.pyplot as plt


ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def timeit(label=None, *, include_class=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            key = label if label is not None else func.__name__
            if include_class:
                key = f"{self.__class__.__name__}.{key}"
            self.timings[key] = elapsed
            return result
        return wrapper
    return decorator


def format_elapsed_time(seconds: float) -> str:
    if seconds >= 1:
        value, unit = seconds, "s"
    elif seconds >= 1e-3:
        value, unit = seconds * 1e3, "ms"
    else:
        value, unit = seconds * 1e6, "us"

    if value >= 100:
        precision = 0
    elif value >= 10:
        precision = 1
    else:
        precision = 2
    return f"{value:.{precision}f} {unit}"


def format_timings(timings, total_elapsed_time=None, header="Simulation timings"):
    if not timings:
        if total_elapsed_time is None:
            return f"{header}\n  No recorded timings"
        return f"{header}\n  Total  {format_elapsed_time(total_elapsed_time)}  100.0%"

    stage_rows = [(label, label, elapsed) for label, elapsed in timings.items()]
    rows = []
    if total_elapsed_time is not None:
        rows.append(("__total__", "Total", total_elapsed_time))
    rows.extend(stage_rows)

    highlighted_keys = {
        key for key, _label, _elapsed in sorted(stage_rows, key=lambda row: row[2], reverse=True)[:3]
    }
    percentage_total = total_elapsed_time if total_elapsed_time is not None else sum(
        elapsed for _key, _label, elapsed in rows
    )
    label_width = max(len("Total"), *(len(label) for label in timings))
    values = [format_elapsed_time(total_elapsed_time)] if total_elapsed_time is not None else []
    values.extend(format_elapsed_time(elapsed) for elapsed in timings.values())
    value_width = max(len(value) for value in values)
    percentage_values = []
    for key, _label, elapsed in rows:
        if key == "__total__" and total_elapsed_time is not None:
            percentage_values.append("100.0%")
        elif percentage_total > 0:
            percentage_values.append(f"{elapsed / percentage_total * 100:5.1f}%")
        else:
            percentage_values.append("  0.0%")
    percentage_width = max(len(value) for value in percentage_values)

    lines = [header]
    for (key, label, elapsed), percentage in zip(rows, percentage_values):
        value = f"{format_elapsed_time(elapsed):>{value_width}}"
        percentage_text = f"{percentage:>{percentage_width}}"
        if key in highlighted_keys:
            value = f"{ANSI_RED}{value}{ANSI_RESET}"
            percentage_text = f"{ANSI_RED}{percentage_text}{ANSI_RESET}"
        lines.append(f"  {label:<{label_width}}  {value}  {percentage_text}")
        if key == "__total__":
            lines.append(
                f"  {'-' * label_width}  {'-' * value_width}  {'-' * percentage_width}"
            )
    return "\n".join(lines)

def plot_timings(timings):
    labels = list(timings.keys())
    values = [timings[label] for label in labels]
    x_positions = list(range(len(labels)))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.8
    ax.bar(x_positions, values, color='skyblue', align='edge', width=bar_width)
    ax.set_xlabel("Function")
    ax.set_ylabel("Time (s)")
    ax.set_title("Execution Time per Function")
    # Adjust tick positions to be at the center of each bar:
    ax.set_xticks([x + bar_width/2 for x in x_positions])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()