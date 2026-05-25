import sys
import pandas as pd
import json
from pathlib import Path
import re

CYCLE_TO_US = 50.0

TRACE_COLOR_PALETTE = [
    "good",
    "thread_state_uninterruptible",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "startup",
    "yellow",
    "olive",
    "bad",
    "terrible",
]

TRACE_COLOR_MAP = {
    "AIC": "good",
    "AIV": "thread_state_uninterruptible",
}

def parse_event_name(column):
    """从列名解析事件名称"""
    for suffix in ['_start(us)', '_end(us)', '_duration(us)', '_start_cycle', '_end_cycle', '_duration_us']:
        if column.endswith(suffix):
            return column[:-len(suffix)]
    return None

def is_cycle_column(col):
    """判断是否为 cycle 列"""
    return '_start_cycle' in col or '_end_cycle' in col

def to_number(value, default=0.0):
    value = pd.to_numeric(value, errors='coerce')
    return default if pd.isna(value) else float(value)

def timeline_label(event_name):
    """AIV_RS_0 -> AIV_RS, AIC_12 -> AIC"""
    return re.sub(r'_\d+$', '', event_name)

def color_for_event(event_name):
    label = timeline_label(event_name)
    if label in TRACE_COLOR_MAP:
        return TRACE_COLOR_MAP[label]

    extra_palette = TRACE_COLOR_PALETTE[2:]
    color_idx = sum(ord(ch) for ch in label) % len(extra_palette)
    TRACE_COLOR_MAP[label] = extra_palette[color_idx]
    return TRACE_COLOR_MAP[label]

def lane_name(core_type, sub_id):
    core_type = str(core_type).strip()
    if core_type == "AIC":
        return "AIC"
    if core_type == "AIV":
        return f"AIV_{int(sub_id)}"
    return f"{core_type}_{int(sub_id)}"

def lane_tid(group_id, core_type, sub_id):
    return f"group_{int(group_id)}_{lane_name(core_type, sub_id)}"

def lane_sort_key(group_id, core_type, sub_id):
    core_type = str(core_type).strip()
    if core_type == "AIC":
        lane_idx = 0
    elif core_type == "AIV":
        lane_idx = 1 + int(sub_id)
    else:
        lane_idx = 100 + int(sub_id)
    return int(group_id) * 100 + lane_idx

def find_base_start_cycle(df):
    """以 raw timeline CSV 中所有正数 start_cycle 的最小值作为 0 点"""
    base_cycle = None
    for col in df.columns:
        if not col.endswith('_start_cycle'):
            continue

        values = pd.to_numeric(df[col], errors='coerce')
        positive_values = values[values > 0]
        if positive_values.empty:
            continue

        col_min = int(positive_values.min())
        base_cycle = col_min if base_cycle is None else min(base_cycle, col_min)

    return base_cycle

def build_lane_metadata(df, pid):
    """固定 Chrome Tracing lane 顺序：group0 AIC, AIV0, AIV1, group1 ..."""
    metadata = [{
        "ph": "M",
        "pid": pid,
        "name": "process_name",
        "args": {"name": str(pid)}
    }]

    group_ids = sorted(int(g) for g in pd.to_numeric(df['group_id'], errors='coerce').dropna().unique())
    lanes = []
    for group_id in group_ids:
        lanes.extend([
            (group_id, "AIC", 0),
            (group_id, "AIV", 0),
            (group_id, "AIV", 1),
        ])

    seen = set(lanes)
    for _, row in df.iterrows():
        group_id = int(to_number(row['group_id']))
        core_type = str(row['core_type']).strip()
        sub_id = int(to_number(row['sub_group_id']))
        lane = (group_id, core_type, sub_id)
        if lane not in seen:
            seen.add(lane)
            lanes.append(lane)

    for group_id, core_type, sub_id in sorted(lanes, key=lambda item: lane_sort_key(*item)):
        tid = lane_tid(group_id, core_type, sub_id)
        sort_idx = lane_sort_key(group_id, core_type, sub_id)
        metadata.append({
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "name": "thread_name",
            "args": {"name": f"group {group_id} {lane_name(core_type, sub_id)}"}
        })
        metadata.append({
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "name": "thread_sort_index",
            "args": {"sort_index": sort_idx}
        })

    return metadata

def collect_events_from_row(row, group_id, base_cycle):
    """从一行数据中收集所有事件及其时间信息（严格屏蔽干扰事件）"""
    events = []
    core_type = row['core_type'].strip() if isinstance(row['core_type'], str) else row['core_type']
    sub_id = int(to_number(row['sub_group_id']))

    for col in row.index:
        if '_start(us)' in col or '_start_cycle' in col:
            event_name = parse_event_name(col)
            if event_name is None:
                continue
            
            if not (event_name.startswith('AIC') or event_name.startswith('AIV')):
                continue

            # 构建 end_col 和 duration_col
            if '_start_cycle' in col:
                end_col = col.replace('_start_cycle', '_end_cycle')
                duration_col = col.replace('_start_cycle', '_duration_us')
            else:
                end_col = col.replace('_start(us)', '_end(us)')
                duration_col = col.replace('_start(us)', '_duration(us)')

            start_value = to_number(row[col])
            end_value = to_number(row[end_col]) if end_col in row.index else 0.0
            duration = to_number(row[duration_col]) if duration_col in row.index else 0.0

            if is_cycle_column(col):
                start_cycle = int(start_value)
                end_cycle = int(end_value)
                if start_cycle <= 0:
                    continue

                start_time = (start_cycle - base_cycle) / CYCLE_TO_US
                if duration == 0.0 and end_cycle > start_cycle:
                    duration = (end_cycle - start_cycle) / CYCLE_TO_US
            else:
                start_cycle = None
                end_cycle = None
                start_time = start_value

            if start_time >= 0 and duration > 0:
                events.append({
                    'name': event_name,
                    'start': start_time,
                    'start_cycle': start_cycle,
                    'end_cycle': end_cycle,
                    'duration': duration,
                    'core_type': core_type,
                    'sub_id': sub_id,
                    'group_id': group_id
                })

    return events

def generate_timestamp_timeline_json(csv_path, output_json_path, op_name="unknown_op"):
    """完全基于原始时间戳生成时间线，不按流水线 step 重排"""
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    if 'core_type' in df.columns:
        df['core_type'] = df['core_type'].str.strip()

    base_cycle = find_base_start_cycle(df)
    if base_cycle is None:
        print("  [WARNING] 未找到有效 start_cycle！")
        final_json = []
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"  输出: {output_json_path}\n")
        return

    pid = f"rank_{op_name}"
    final_json = build_lane_metadata(df, pid)
    all_events = []
    grouped = df.groupby('group_id')

    for group_id, group_df in grouped:
        for _, row in group_df.iterrows():
            all_events.extend(collect_events_from_row(row, group_id, base_cycle))

    if len(all_events) == 0:
        print("  [WARNING] 未收集到任何流水线事件！")
    else:
        for e in sorted(all_events, key=lambda item: (lane_sort_key(item['group_id'], item['core_type'], item['sub_id']), item['start'], item['name'])):
            tid = lane_tid(e['group_id'], e['core_type'], e['sub_id'])
            final_json.append({
                "ph": "X",
                "cat": f"{e['core_type'].lower()}_event",
                "pid": pid,
                "tid": tid,
                "name": e['name'],
                "ts": round(e['start'], 6),
                "dur": round(e['duration'], 6),
                "cname": color_for_event(e['name']),
                "args": {
                    "group_id": int(e['group_id']),
                    "core_type": str(e['core_type']).strip(),
                    "sub_group_id": int(e['sub_id']),
                    "start_cycle": e['start_cycle'],
                    "end_cycle": e['end_cycle'],
                    "label": timeline_label(e['name']),
                }
            })

        print(f"  [Timeline] 基准 start_cycle: {base_cycle}")
        print(f"  生成 {len(all_events)} 个真实时间戳流水线事件")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  输出: {output_json_path}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("./output_timer")

    if not base_dir.exists():
        print(f"[INFO] 目录 {base_dir} 不存在。")
        sys.exit(0)

    CURRENT_OP_NAME = "UnknownOp"

    if len(sys.argv) > 2:
        CURRENT_OP_NAME = sys.argv[2]

    data_dir = base_dir / "data"
    processed_count = 0
    if data_dir.exists():
        print(f"=== 开始处理 {CURRENT_OP_NAME} (真实时间戳模式) ===")
        for csv_path in data_dir.glob("timer_*_timeline.csv"):
            match = re.search(r'(rank_\d+)', csv_path.stem)
            # 生成干净漂亮的输出文件名
            json_name = f"{match.group(1) if match else csv_path.stem}_{CURRENT_OP_NAME}_timeline.json"
            json_path = base_dir / json_name
            
            print(f"-> 正在处理: {csv_path.name}")
            # 调用真实时间戳时间线生成器
            generate_timestamp_timeline_json(
                csv_path=str(csv_path), 
                output_json_path=str(json_path), 
                op_name=CURRENT_OP_NAME
            )
            processed_count += 1
            
    if processed_count == 0:
        print("[WARNING] 没有在 data/ 目录下找到 timer_*_timeline.csv 文件。请检查上游输出。")
    else:
        print("[Done] 所有 Timeline JSON 生成完毕！请导入 chrome://tracing 查看流水线排布。")
