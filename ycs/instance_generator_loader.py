import random
from os import listdir
import json

import numpy as np

try:
    from ycs.gp_simulation import simulation_run, dyn_lookahead
    from ycs.gp_manual_heuristics import all_manual_rules
except ImportError:
    try:
        from .gp_simulation import simulation_run, dyn_lookahead
        from .gp_manual_heuristics import all_manual_rules
    except ImportError:
        from gp_simulation import simulation_run, dyn_lookahead
        from gp_manual_heuristics import all_manual_rules

min_internal_processing_time = 2
max_internal_processing_time = 4
external_processing_time = 2
min_bay_number = 1
max_bay_number = 40
min_internal_task_tardiness_weight: int = 3
max_internal_task_tardiness_weight: int = 4
external_task_tardiness_weight = 1
travel_time_per_bay = 0.06
acceleration_deceleration_time = 0.09

average_task_per_hour_list = [12, 15, 18]
external_task_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]


def generate_task_list(number_of_internal_task, number_of_external_task, planning_window_in_hours):
    planning_window_in_minutes = planning_window_in_hours * 60

    tasks_dict = {}

    dummy_task = {}
    dummy_task['id'] = 0
    dummy_task['is_internal'] = True
    dummy_task['processing_time'] = 0
    dummy_task['tardiness_weight'] = 0
    dummy_task['due_date'] = 0
    dummy_task['release_time'] = 0
    dummy_task['bay_number'] = 1
    tasks_dict[0] = dummy_task

    for i in range(1, number_of_internal_task + 1):
        internal_task = {}
        internal_task['id'] = i
        internal_task['is_internal'] = True
        internal_task['processing_time'] = round(random.uniform(min_internal_processing_time,
                                                                max_internal_processing_time), 2)
        internal_task['tardiness_weight'] = random.randint(min_internal_task_tardiness_weight,
                                                           max_internal_task_tardiness_weight)
        internal_task['release_time'] = round(random.uniform(0, planning_window_in_minutes), 2)
        internal_task['due_date'] = round(random.uniform(internal_task['release_time'] +
                                                         internal_task['processing_time'],
                                                         internal_task['release_time'] + 6), 2)
        internal_task['bay_number'] = random.randint(min_bay_number, max_bay_number)

        tasks_dict[internal_task['id']] = internal_task

    for i in range(1, number_of_external_task + 1):
        external_task = {}
        external_task['id'] = i + number_of_internal_task
        external_task['is_internal'] = False
        external_task['processing_time'] = external_processing_time
        external_task['tardiness_weight'] = external_task_tardiness_weight
        external_task['release_time'] = round(random.uniform(0, planning_window_in_minutes), 2)
        external_task['due_date'] = external_task['release_time'] + external_task['processing_time']
        external_task['bay_number'] = random.randint(min_bay_number, max_bay_number)

        tasks_dict[external_task['id']] = external_task

    return tasks_dict


def calculate_time_adjacency(task1, task2):
    if task1['bay_number'] == task2['bay_number']:
        return 0
    else:
        return round(travel_time_per_bay *
                     abs(task1['bay_number'] - task2['bay_number']) + acceleration_deceleration_time, 2)


def generate_time_adjacency_matrix(task_dict):
    # Number of tasks, including dummy task
    number_of_tasks = len(task_dict)
    time_adjacency_matrix = [[0 for i in range(number_of_tasks)] for j in range(number_of_tasks)]

    for i in range(number_of_tasks):
        for j in range(number_of_tasks):
            if i == j:
                time_adjacency_matrix[i][j] = 0
            else:
                time_adjacency_matrix[i][j] = calculate_time_adjacency(task_dict[i], task_dict[j])

    return time_adjacency_matrix


def generate_single_instance(planning_window_in_hours, average_task_per_hour, external_task_ratio):
    number_of_tasks = int(planning_window_in_hours * average_task_per_hour)
    number_of_external_tasks = int(number_of_tasks * external_task_ratio)
    number_of_internal_tasks = number_of_tasks - number_of_external_tasks

    instance = {}
    instance['planning_window_in_hours'] = planning_window_in_hours
    instance['average_task_per_hour'] = average_task_per_hour
    instance['external_task_ratio'] = external_task_ratio
    instance['task_dict'] = generate_task_list(number_of_internal_tasks, number_of_external_tasks,
                                               planning_window_in_hours)
    instance['time_adjacency_matrix'] = generate_time_adjacency_matrix(instance['task_dict'])

    return instance


def get_instance_offline_optimal(instance):
    min_obj = np.inf
    for rule in all_manual_rules:
        obj, _, _ = simulation_run(instance['task_dict'], instance['time_adjacency_matrix'], dyn_lookahead, rule)
        min_obj = min(obj, min_obj)
    return min_obj


def generate_instances(planning_window_in_hours, number_of_instances_per_setting):
    instances = []
    for average_task_per_hour in average_task_per_hour_list:
        for external_task_ratio in external_task_ratio_list:
            for i in range(number_of_instances_per_setting):
                instance = generate_single_instance(planning_window_in_hours,
                                                    average_task_per_hour, external_task_ratio)
                instance['offline_optimal'] = get_instance_offline_optimal(instance)
                instances.append(instance)

    return instances


def generate_write_instances_to_folder(planning_window_in_hours, number_of_instances_per_setting, folder_path):
    instances = generate_instances(planning_window_in_hours, number_of_instances_per_setting)
    for i, instance in enumerate(instances):
        file_path = folder_path + '/instance_' + str(i) + '.json'
        with open(file_path, 'w') as f:
            json.dump(instance, f, indent=4)
        print('dump', file_path)


def read_instances_from_folder(folder_path):
    instances = []
    for filename in listdir(folder_path):
        if filename.endswith('.json'):
            with open(folder_path + '/' + filename, 'r') as f:
                data = json.load(f)
                data['task_dict'] = {int(k): v for k, v in data['task_dict'].items()}
                instances.append(data)
    return instances


def read_instances_from_folder_old(folder_path):
    instances = []
    for filename in listdir(folder_path):
        if filename.endswith('.json'):
            with open(folder_path + '/' + filename, 'r') as f:
                data = json.load(f)

                # Convert task list to task_dict
                task_dict = {task['id']: task for task in data['tasks']}

                # Convert all task_dict keys to int (already the case here, but for consistency)
                new_task_dict = {int(k): v for k, v in task_dict.items()}

                # Assemble into unified instance format
                instance = {
                    'planning_window_in_hours': data['instance_setting']['planning_window_length_in_hours'],
                    'average_task_per_hour': data['instance_setting'][
                        'average_number_of_tasks_executed_by_yc_per_hour'],
                    'external_task_ratio': data['instance_setting']['external_task_ratio'],
                    'task_dict': new_task_dict,
                    'time_adjacency_matrix': data['time_adjacency_matrix'],
                    'offline_solution_task_sequence': data.get('offline_solution_task_sequence'),
                    'offline_optimal': data.get('offline_optimal'),
                    'minimum_twt': data.get('offline_optimal'),
                    'offline_solution_computation_time_in_seconds': data.get(
                        'offline_solution_computation_time_in_seconds'),
                }

                # For time adjacency matrix, convert to float and round to 2 decimal places
                for i in range(len(instance['time_adjacency_matrix'])):
                    for j in range(len(instance['time_adjacency_matrix'][i])):
                        instance['time_adjacency_matrix'][i][j] = round(float(instance['time_adjacency_matrix'][i][j]), 2)

                instances.append(instance)

    return instances


if __name__ == "__main__":
    generate_write_instances_to_folder(4, 5, 'train_set')
    generate_write_instances_to_folder(4, 5, 'validation_set')
    generate_write_instances_to_folder(4, 50, 'test_set')

    train_set = read_instances_from_folder('train_set')
    validation_set = read_instances_from_folder('validation_set')
    test_set = read_instances_from_folder('test_set')
