import os
import re

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


# Use this when scan order is not consistent enough for automatic grouping.
# For Experiment3, edit the relative scan paths below so each plant contains
# its actual scans across days.
PLANT_SCANS_BY_EXPERIMENT = {
    "Experiment3": {
        "1": [
            #"day1/scan_22May_10:46:38",
            #"day2/scan_23May_08:48:39",
            #"day3/scan_24May_09:06:05",
            #"day4/scan_25May_08:47:32",
            #"day5/scan_26May_12:10:12",
            #"day6/scan_27May_11:24:58",
            #"day7/scan_28May_08:28:09",
            #"day8/scan_29May_08:25:38",
            "day9/scan_30May_09:18:59",
            "day10/scan_31May_07:02:27",
        ],
        "2": [
            # "day1/scan_22May_11:56:08",
            # "day2/scan_23May_09:36:43",
            # "day3/scan_24May_09:55:29",
            # "day4/scan_25May_09:37:45",
            # "day5/scan_26May_11:15:48",
            # "day6/scan_27May_08:42:56",
            # "day7/scan_28May_09:16:46",
            # "day8/scan_29May_09:24:40",
            "day9/scan_30May_10:27:51",
            "day10/scan_31May_07:55:26",
        ],
        "3": [
            # "day1/scan_22May_12:51:20",
            # "day2/scan_23May_10:24:59",
            # "day3/scan_24May_10:48:19",
            # "day4/scan_25May_10:39:32",
            # "day5/scan_26May_10:14:44",
            # "day6/scan_27May_07:52:31",
            # "day7/scan_28May_10:16:38",
            # "day8/scan_29May_10:21:45",
            "day9/scan_30May_11:22:12",
            "day10/scan_31May_08:58:55",
        ],
        "4": [
            # "day1/scan_22May_13:45:54",
            # "day2/scan_23May_17:03:57",
            # "day3/scan_24May_17:03:02",
            # "day4/scan_25May_11:52:32",
            # "day5/scan_26May_09:22:06",
            # "day6/scan_27May_10:32:01",
            # "day7/scan_28May_11:13:06",
            # "day8/scan_29May_11:37:13",
            "day9/scan_30May_12:19:01",
            "day10/scan_31May_10:00:49",
        ],
        "5": [
            # "day1/scan_22May_14:42:53",
            # "day2/scan_23May_17:56:23",
            # "day3/scan_24May_17:53:53",
            # "day4/scan_25May_12:54:31",
            # "day5/scan_26May_08:33:02",
            # "day6/scan_27May_09:39:26",
            # "day7/scan_28May_12:07:00",
            # "day8/scan_29May_12:29:24",
            "day9/scan_30May_13:14:59",
            "day10/scan_31May_10:55:21",
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


def natural_day_key(name):
    match = re.fullmatch(r"day(\d+)", name, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def discover_ordinal_scan_groups(data_dir, parent_prefix="day", scan_prefix="scan_"):
    """
    Group nested scan folders by their order inside each parent folder.

    For Experiment3-style data, this treats the first scan in every day folder
    as group 1, the second scan as group 2, etc. Returned scan names are
    relative paths such as day1/scan_22May_10:46:38.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    parent_names = sorted(
        (
            name
            for name in os.listdir(data_dir)
            if name.startswith(parent_prefix)
            and os.path.isdir(os.path.join(data_dir, name))
        ),
        key=natural_day_key,
    )

    grouped_scans = {}
    for parent_name in parent_names:
        parent_dir = os.path.join(data_dir, parent_name)
        scan_names = discover_scan_names(parent_dir)
        scan_names = [name for name in scan_names if name.startswith(scan_prefix)]
        for scan_index, scan_name in enumerate(scan_names, start=1):
            group_key = str(scan_index)
            grouped_scans.setdefault(group_key, []).append(
                os.path.join(parent_name, scan_name)
            )

    return grouped_scans


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
