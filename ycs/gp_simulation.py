import sys
from statistics import mean

epsilon = 0.0000001


def min_lookahead(
        available_task_ids, task_map, time_adjacency_matrix, current_time, previous_task_id
):
    candidate_task_ids = []

    for available_task_id in available_task_ids:
        unscheduled_task = task_map[available_task_id]
        if unscheduled_task['is_internal']:
            if unscheduled_task['release_time'] < current_time + time_adjacency_matrix[previous_task_id][
                available_task_id] + epsilon:
                candidate_task_ids.append(available_task_id)
        else:
            if unscheduled_task['release_time'] < current_time + epsilon:
                candidate_task_ids.append(available_task_id)

    return candidate_task_ids


def dyn_lookahead(available_task_ids, task_map, time_adjacency_matrix, current_time, previous_task_id
                  ):
    candidate_task_ids = min_lookahead(available_task_ids, task_map, time_adjacency_matrix, current_time,
                                       previous_task_id)

    if not candidate_task_ids:
        return candidate_task_ids

    look_ahead_task = []

    for available_task_id in available_task_ids:
        if available_task_id not in candidate_task_ids:
            available_task_release_time = task_map[available_task_id]['release_time']
            is_candidate = True

            for candidate_task_id in candidate_task_ids:
                completion_time = max(current_time, task_map[candidate_task_id]['release_time']) + \
                                  time_adjacency_matrix[previous_task_id][candidate_task_id] + task_map[
                                      candidate_task_id]['processing_time']
                if available_task_release_time > completion_time + time_adjacency_matrix[candidate_task_id][
                    available_task_id] + epsilon:
                    is_candidate = False
                    break

            if is_candidate:
                look_ahead_task.append(available_task_id)

    candidate_task_ids.extend(look_ahead_task)

    return candidate_task_ids


def get_terminal_argument_dict(next_task_id, candidate_task_ids, task_dict, time_adjacency_matrix, current_time,
                               previous_task_id, argument_dict):
    for terminal in [
        'r_i', 'p_i', 's_ji', 'd_i', 'w_i', 't', 'n', 'bar_r', 'bar_p', 'bar_s', 'bar_d', 'bar_w'
    ]:
        if terminal == 'r_i':
            argument_dict['r_i'] = task_dict[next_task_id]['release_time']
        elif terminal == 'p_i':
            argument_dict['p_i'] = task_dict[next_task_id]['processing_time']
        elif terminal == 's_ji':
            argument_dict['s_ji'] = time_adjacency_matrix[previous_task_id][next_task_id]
        elif terminal == 'd_i':
            argument_dict['d_i'] = task_dict[next_task_id]['due_date']
        elif terminal == 'w_i':
            argument_dict['w_i'] = task_dict[next_task_id]['tardiness_weight']
        elif terminal == 't' and 't' not in argument_dict:
            argument_dict['t'] = current_time
        elif terminal == 'n' and 'n' not in argument_dict:
            argument_dict['n'] = len(candidate_task_ids)
        elif terminal == 'bar_r' and 'bar_r' not in argument_dict:
            argument_dict['bar_r'] = mean([task_dict[task_id]['release_time'] for task_id in candidate_task_ids])
        elif terminal == 'bar_p' and 'bar_p' not in argument_dict:
            argument_dict['bar_p'] = mean([task_dict[task_id]['processing_time'] for task_id in candidate_task_ids])
        elif terminal == 'bar_s' and 'bar_s' not in argument_dict:
            argument_dict['bar_s'] = mean([time_adjacency_matrix[previous_task_id][task_id]
                                           for task_id in candidate_task_ids])
        elif terminal == 'bar_d' and 'bar_d' not in argument_dict:
            argument_dict['bar_d'] = mean([task_dict[task_id]['due_date'] for task_id in candidate_task_ids])
        elif terminal == 'bar_w' and 'bar_w' not in argument_dict:
            argument_dict['bar_w'] = mean([task_dict[task_id]['tardiness_weight'] for task_id in candidate_task_ids])
    return argument_dict


def select_highest_priority_single(priority_rule, candidate_task_ids, task_dict, time_adjacency_matrix, current_time,
                                   previous_task_id):
    terminal_arguments = get_terminal_argument_dict(candidate_task_ids[0], candidate_task_ids,
                                                    task_dict, time_adjacency_matrix, current_time,
                                                    previous_task_id, {})

    id_2_argument = {}
    id_2_priority = {}

    for task_id in candidate_task_ids:
        terminal_arguments = get_terminal_argument_dict(task_id, candidate_task_ids, task_dict, time_adjacency_matrix,
                                                        current_time, previous_task_id, terminal_arguments)
        priority = priority_rule(**terminal_arguments)

        id_2_argument[task_id] = terminal_arguments
        id_2_priority[task_id] = priority

    # Rank tasks by priority
    id_2_rank = {task_id: rank for rank, task_id in enumerate(sorted(id_2_priority, key=id_2_priority.get, reverse=True))}
    top_rank_id = min(id_2_rank, key=id_2_rank.get)

    return top_rank_id, id_2_rank, id_2_argument


def select_highest_priority(priority_rules, candidate_task_ids, task_dict, time_adjacency_matrix, current_time,
                            previous_task_id):
    if len(priority_rules) == 1:
        return select_highest_priority_single(priority_rules[0], candidate_task_ids, task_dict,
                                              time_adjacency_matrix, current_time, previous_task_id)
    else:
        id_2_argument = {}
        id_2_borda_count = {}
        vote = []

        for priority_rule in priority_rules:
            top_rank_id, id_2_rank_temp, id_2_argument_temp = select_highest_priority_single(
                priority_rule, candidate_task_ids, task_dict, time_adjacency_matrix, current_time, previous_task_id
            )

            vote.append(top_rank_id)

            for task_id in id_2_rank_temp:
                if task_id not in id_2_borda_count:
                    id_2_borda_count[task_id] = 0
                id_2_borda_count[task_id] += id_2_rank_temp[task_id]

            # Only set arguments once
            if id_2_argument is None:
                id_2_argument = id_2_argument_temp

        # Majority vote for top rank id
        id_2_vote = {task_id: vote.count(task_id) for task_id in set(vote)}
        top_rank_id = max(id_2_vote, key=id_2_vote.get)

        # Assign 0 to top-ranked task
        id_2_rank = {top_rank_id: 0}

        # Remove top_rank_id from Borda count and rank the rest
        remaining_tasks = [task_id for task_id in candidate_task_ids if task_id != top_rank_id]

        # Sort by descending Borda count (higher score → higher priority → lower rank index)
        sorted_remaining = sorted(remaining_tasks, key=lambda x: -id_2_borda_count.get(x, 0))

        for i, task_id in enumerate(sorted_remaining, start=1):
            id_2_rank[task_id] = i

        return top_rank_id, id_2_rank, id_2_argument


def calculate_weighted_tardiness(current_time, previous_task_id, task_dict, time_adjacency_matrix, next_task_id):
    s_i_j = time_adjacency_matrix[previous_task_id][next_task_id]
    r_j = task_dict[next_task_id]['release_time']
    p_j = task_dict[next_task_id]['processing_time']
    d_j = task_dict[next_task_id]['due_date']

    c_j = max(current_time + s_i_j, r_j) + p_j if task_dict[next_task_id]['is_internal'] \
        else max(current_time, r_j) + s_i_j + p_j

    tardiness = max(0, c_j - d_j)
    return c_j, tardiness * task_dict[next_task_id]['tardiness_weight']


def calculate_twt_with_sequence(task_dict, time_adjacency_matrix, task_sequence):
    current_time, previous_task_id, total_weighted_tardiness = 0, 0, 0
    for j in task_sequence:
        current_time, weighted_tardiness = calculate_weighted_tardiness(current_time, previous_task_id, task_dict,
                                                                        time_adjacency_matrix, j)
        total_weighted_tardiness += weighted_tardiness
        previous_task_id = j

    return total_weighted_tardiness


def get_candidate_task_ids(lookahead_method, ordered_unscheduled_task_ids, task_dict, time_adjacency_matrix,
                           current_time, previous_task_id, predict_window_in_minutes):
    task_earliest_start_time = []

    for task_id, _ in ordered_unscheduled_task_ids:
        task = task_dict[task_id]
        if task['is_internal']:
            est = max(current_time, task['release_time'] - time_adjacency_matrix[previous_task_id][task_id])
        else:
            est = max(current_time, task['release_time'])
        task_earliest_start_time.append((task_id, est))

    # Sort tasks by earliest start time
    task_earliest_start_time.sort(key=lambda x: x[1])

    for task_id, temp_current_time in task_earliest_start_time:
        available_task_ids = []
        for t_id, _ in ordered_unscheduled_task_ids:
            task = task_dict[t_id]
            if task['is_internal']:
                if task['release_time'] < temp_current_time + predict_window_in_minutes + epsilon:
                    available_task_ids.append(t_id)
            else:
                if task['release_time'] < temp_current_time + epsilon:
                    available_task_ids.append(t_id)

        candidate_task_ids = lookahead_method(
            available_task_ids, task_dict, time_adjacency_matrix, temp_current_time, previous_task_id
        )

        if candidate_task_ids:
            return temp_current_time, available_task_ids, candidate_task_ids

    print("No available tasks")
    sys.exit(0)


def simulation_run(task_dict, time_adjacency_matrix, look_ahead_method, priority_rules):
    ordered_unscheduled_tasks = []
    for task_id, task in task_dict.items():
        if task_id != 0:
            ordered_unscheduled_tasks.append((task_id, task['release_time']))

    ordered_unscheduled_tasks.sort(key=lambda x: x[1])

    current_time, previous_task_id, total_weighted_tardiness = 0, 0, 0
    task_sequence = [0]

    decision_situations = []

    while ordered_unscheduled_tasks:
        current_time, available_task_ids, candidate_task_ids = get_candidate_task_ids(look_ahead_method,
                                                                                      ordered_unscheduled_tasks,
                                                                                      task_dict, time_adjacency_matrix,
                                                                                      current_time, previous_task_id,
                                                                                      20)

        assert len(candidate_task_ids) != 0, available_task_ids

        next_task_id, id_2_rank, id_2_argument = select_highest_priority(priority_rules, candidate_task_ids,
                                                                             task_dict, time_adjacency_matrix,
                                                                             current_time, previous_task_id)

        decision_situations.append({
            'candidate_task_ids': candidate_task_ids,
            'next_task_id': next_task_id,
            'id_2_rank': id_2_rank,
            'id_2_argument': id_2_argument
        })

        current_time, weighted_tardiness = calculate_weighted_tardiness(current_time, previous_task_id, task_dict,
                                                                        time_adjacency_matrix, next_task_id)
        total_weighted_tardiness += weighted_tardiness
        previous_task_id = next_task_id

        task_sequence.append(next_task_id)
        ordered_unscheduled_tasks = [(task_id, release_time) for task_id, release_time in ordered_unscheduled_tasks
                                     if task_id != next_task_id]

    try:
        assert len(task_sequence) == len(task_dict)
        assert round(calculate_twt_with_sequence(task_dict, time_adjacency_matrix, task_sequence), 2) == round(
            total_weighted_tardiness, 2)
    except AssertionError:
        print('Error in simulation_run')
        exit(1)

    return total_weighted_tardiness, task_sequence, decision_situations
