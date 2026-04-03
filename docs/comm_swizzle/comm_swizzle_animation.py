"""
CommSwizzle Communication Scheduling Algorithm - Parallel Core Animation

Visualizes CommSwizzle with parallel cores:
- Each "step" dispatches numCores consecutive loopIdx values simultaneously
- Each core is shown with a distinct color
- Supports IS_DETERMINISTIC mode (fixed group size with RoundUp)

Usage:
    python comm_swizzle_animation.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ============ Configurable Parameters ============
GRID_ROW = 4             # gridShape.row
GRID_COL = 4             # gridShape.column
GRID_RANK = 4            # gridShape.rank
CORE_SPLIT_ROW = 12      # coreSplit.row  (data-dim parallelism)
CORE_SPLIT_COL = 1       # coreSplit.column (rank-dim parallelism)
SWIZZLE_DIRECTION = 0    # 0: row-major swizzle
IS_DETERMINISTIC = True  # deterministic
IS_RANK_SHIFT = True     # only for simulating
# ==================================================

N_DATA = GRID_ROW * GRID_COL
NUM_CORES = CORE_SPLIT_ROW * CORE_SPLIT_COL

# Box dimensions (width x height in data units)
BOX_W = 0.9
BOX_H = 0.5


def roundup(a, b):
    return ((a + b - 1) // b) * b


def comm_swizzle_get_coord(loop_idx):
    """Python implementation of CommSwizzle::GetCoord"""
    ROW_DIM = SWIZZLE_DIRECTION
    COL_DIM = 1 - SWIZZLE_DIRECTION

    core_split_row = CORE_SPLIT_ROW
    core_split_col = CORE_SPLIT_COL

    # IS_DETERMINISTIC: collapse coreSplit to (row*col, 1)
    if IS_DETERMINISTIC:
        core_split_row = CORE_SPLIT_ROW * CORE_SPLIT_COL
        core_split_col = 1

    swizzle_offset = [core_split_row, core_split_col][ROW_DIM]
    flatten_shape = [GRID_ROW * GRID_COL, GRID_RANK]

    group_size = swizzle_offset * flatten_shape[COL_DIM]
    group_idx = loop_idx // group_size
    group_offset = loop_idx - group_idx * group_size

    if IS_DETERMINISTIC:
        in_group_rows = swizzle_offset  # fixed, not clamped
    else:
        in_group_rows = min(swizzle_offset, flatten_shape[ROW_DIM] - group_idx * swizzle_offset)

    coord = [0, 0]
    coord[COL_DIM] = group_offset // in_group_rows
    in_group_row_idx = group_offset - coord[COL_DIM] * in_group_rows
    coord[ROW_DIM] = group_idx * swizzle_offset + in_group_row_idx

    if IS_RANK_SHIFT:
        n_stride = GRID_RANK // core_split_col
        offset = coord[1] * n_stride
        coord[1] = (offset + offset // GRID_RANK + coord[0]) % GRID_RANK

    return coord[0], coord[1]


def get_total_iters():
    """Match GetCoreLoop logic"""
    if IS_DETERMINISTIC:
        det_cores = CORE_SPLIT_ROW * CORE_SPLIT_COL
        return roundup(N_DATA, det_cores) * GRID_RANK
    else:
        return N_DATA * GRID_RANK


def generate_core_colors(n):
    cmap = plt.cm.hsv
    return [matplotlib.colors.to_hex(cmap(i / max(n, 1))) for i in range(n)]


def main():
    total_iters = get_total_iters()
    num_steps = (total_iters + NUM_CORES - 1) // NUM_CORES

    # Pre-compute steps
    steps = []
    for s in range(num_steps):
        core_tasks = []
        for c in range(NUM_CORES):
            loop_idx = s * NUM_CORES + c
            if loop_idx < total_iters:
                data_idx, rank = comm_swizzle_get_coord(loop_idx)
                # Skip out-of-range data (deterministic padding)
                if data_idx < N_DATA:
                    core_tasks.append((c, data_idx, rank))
        steps.append(core_tasks)

    core_colors = generate_core_colors(NUM_CORES)

    # Figure sizing based on box dimensions
    fig_w = max(6, GRID_RANK * BOX_W + 2.5)
    fig_h = max(4, N_DATA * BOX_H + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#1a1a2e')

    det_tag = '  DETERMINISTIC' if IS_DETERMINISTIC else ''
    fig.suptitle(f'CommSwizzle  grid=({GRID_ROW},{GRID_COL},{GRID_RANK})  '
                 f'coreSplit=({CORE_SPLIT_ROW},{CORE_SPLIT_COL}){det_tag}',
                 fontsize=12, fontweight='bold', color='white', y=0.97)

    cell_core = np.full((N_DATA, GRID_RANK), -1, dtype=int)
    cell_step = np.full((N_DATA, GRID_RANK), -1, dtype=int)
    prev_texts = []

    # Use box center spacing = box size (no gaps)
    x_step = BOX_W  # horizontal spacing
    y_step = BOX_H  # vertical spacing

    def update(frame):
        nonlocal prev_texts
        step = frame - 1

        ax.clear()
        ax.set_facecolor('#16213e')
        ax.set_xlabel('rank', color='white', fontsize=10)
        ax.set_ylabel('data idx', color='white', fontsize=10)

        current_tasks = []
        if 0 <= step < num_steps:
            current_tasks = steps[step]
            for core_id, data_idx, rank in current_tasks:
                cell_core[data_idx, rank] = core_id
                cell_step[data_idx, rank] = step

        current_set = set((d, r) for _, d, r in current_tasks)

        for d in range(N_DATA):
            for r in range(GRID_RANK):
                cx = r * x_step
                cy = d * y_step
                cid = cell_core[d, r]
                if cid >= 0:
                    is_cur = (d, r) in current_set
                    alpha = 1.0 if is_cur else 0.45
                    color = core_colors[cid]
                    rect = patches.FancyBboxPatch(
                        (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                        boxstyle="round,pad=0.02",
                        facecolor=(*matplotlib.colors.to_rgb(color), alpha),
                        edgecolor='white' if is_cur else 'gray',
                        linewidth=2.0 if is_cur else 0.5
                    )
                    ax.add_patch(rect)
                    sid = cell_step[d, r]
                    ax.text(cx, cy, f'C{cid} S{sid}', ha='center', va='center',
                            color='white', fontsize=12, fontweight='bold')
                else:
                    rect = patches.FancyBboxPatch(
                        (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                        boxstyle="round,pad=0.02",
                        facecolor=(0.2, 0.2, 0.3, 0.3),
                        edgecolor='gray', linewidth=0.5, linestyle='--'
                    )
                    ax.add_patch(rect)

        x_max = (GRID_RANK - 1) * x_step
        y_max = (N_DATA - 1) * y_step
        ax.set_xlim(-BOX_W/2 - 0.1, x_max + BOX_W/2 + 0.1)
        ax.set_ylim(y_max + BOX_H/2 + 0.1, -BOX_H/2 - 0.1)
        ax.set_xticks([r * x_step for r in range(GRID_RANK)])
        ax.set_xticklabels([f'r{i}' for i in range(GRID_RANK)], color='white', fontsize=9)
        ax.set_yticks([d * y_step for d in range(N_DATA)])
        ax.set_yticklabels([f'd{i}' for i in range(N_DATA)], color='white', fontsize=9)
        ax.tick_params(colors='white')
        ax.set_aspect('equal')

        for t in prev_texts:
            t.remove()
        prev_texts.clear()

        if step < 0:
            info = 'Ready...'
        elif step < num_steps:
            tasks_str = ' '.join([f'C{c}:(d{d},r{r})' for c, d, r in current_tasks])
            if len(tasks_str) > 100:
                tasks_str = tasks_str[:97] + '...'
            info = f'Step {step}/{num_steps-1}: {tasks_str}'
        else:
            info = f'Done! {num_steps} steps x {NUM_CORES} cores = {total_iters} iters'

        txt = fig.text(0.5, 0.02, info, ha='center', va='center',
                       fontsize=7, color='#0f3460', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8e8e8',
                                 edgecolor='#0f3460', alpha=0.95))
        prev_texts.append(txt)

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps + 3,
        interval=1200, repeat=True, repeat_delay=3000
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    output_path = 'comm_swizzle_animation.gif'
    print(f'Generating animation {output_path} ...')
    print(f'  cores={NUM_CORES}, steps={num_steps}, iters={total_iters}, '
          f'deterministic={IS_DETERMINISTIC}')
    for s, tasks in enumerate(steps):
        ranks_hit = [r for _, _, r in tasks]
        dup = len(ranks_hit) != len(set(ranks_hit))
        mark = ' *** RANK COLLISION ***' if dup else ''
        print(f'  Step {s}: {["(C"+str(c)+":d"+str(d)+",r"+str(r)+")" for c,d,r in tasks]}{mark}')
    ani.save(output_path, writer='pillow', fps=1, dpi=120)
    print(f'Animation saved to: {output_path}')


if __name__ == '__main__':
    main()
