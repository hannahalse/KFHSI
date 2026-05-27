import os

from Generate_cube import BASE_DIR


DATA_ROOT = os.path.join(BASE_DIR, "edge", "data")
DEFAULT_EXPERIMENT = "Experiment2"

# Add or edit scan groups here. Both batch cube generation and batch
# visualization use this same mapping.
POSITION_SCANS_BY_EXPERIMENT = {
    "Experiment1": {
        "1": [
            "scan_27April_15:26:44",
            "scan_28April_12:11:41",
            "scan_29April_11:18:58",
            "scan_30April_09:46:32",
        ],
        "2": [
            "scan_27April_16:52:45",
            "scan_28April_13:13:09",
            "scan_29April_12:09:10",
            "scan_30April_10:35:05",
        ],
        "3": [
            "scan_27April_17:55:28",
            "scan_28April_14:13:25",
            "scan_29April_13:00:22",
            "scan_30April_11:26:57",
        ],
    },
    "Experiment2": {
        "1": [
            "scan_04May_11:01:08",
            "scan_05May_11:58:21",
            "scan_06May_11:10:08",
            "scan_07May_12:03:15",
            "scan_08May_12:09:29",
            "scan_09May_12:17:32",
            "scan_11May_09:20:25",
            "scan_12May_10:58:19",
            "scan_13May_10:07:42",
        ],
        "2": [
            "scan_04May_11:53:02",
            "scan_05May_12:50:07",
            "scan_06May_11:59:40",
            "scan_07May_12:51:47",
            "scan_08May_12:58:07",
            "scan_09May_13:09:07",
            "scan_11May_10:09:11",
            "scan_12May_12:45:45",
            "scan_13May_10:56:34",
        ],
        "3": [
            "scan_04May_13:00:39",
            "scan_05May_13:39:47",
            "scan_06May_12:48:43",
            "scan_07May_13:44:14",
            "scan_08May_13:49:28",
            "scan_09May_14:03:51",
            "scan_11May_10:57:40",
            "scan_12May_13:36:09",
            "scan_13May_11:46:46",
        ],
    },
}


def experiment_dir(experiment):
    return os.path.join(DATA_ROOT, experiment)


def scan_folder_for_name(data_dir, scan_name):
    return os.path.join(data_dir, scan_name)


def discover_scan_names(data_dir):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    return sorted(
        name
        for name in os.listdir(data_dir)
        if name.startswith("scan_") and os.path.isdir(scan_folder_for_name(data_dir, name))
    )


def sort_position_keys(position_keys):
    def key(position):
        return (not position.isdigit(), int(position) if position.isdigit() else position)

    return sorted(position_keys, key=key)


def selected_position_keys(experiment, requested_position):
    position_scans = POSITION_SCANS_BY_EXPERIMENT.get(experiment, {})
    if not position_scans:
        raise ValueError(
            f"No POSITION_SCANS_BY_EXPERIMENT entry defined for {experiment!r}. "
            "Add scan groups in experiment_scan_groups.py or run with --discover."
        )

    if requested_position == "all":
        return sort_position_keys(position_scans.keys())

    if requested_position not in position_scans:
        available = ", ".join(sort_position_keys(position_scans.keys()))
        raise ValueError(
            f"Position {requested_position!r} is not defined for {experiment!r}. "
            f"Available positions: {available}"
        )

    return [requested_position]
