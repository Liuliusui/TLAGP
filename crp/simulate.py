# crp_simulation.py
import math
import os
import random
from copy import deepcopy
from typing import List, Tuple

from crp.bay import Bay

PENALTY = 10000



def score(bay, retrieve_seq, stack_idx):
    # 这里调用 GP 演化出的优先函数，例如：
    # return pf_tree(bay, retrieve_seq, stack_idx)
    raise NotImplementedError
def simulate_with_heuristic(
    initial_layout: List[List[int]],
    retrieve_seq: List[int],
    heuristic_fn
) ->  Tuple[int, float]:
    """
    使用启发式函数动态指导搬运，计算总重排次数。

    参数：
        initial_layout: List[List[int]]
            每列（stack）从底层到顶层的容器编号列表，例如 [[2,5,1],[3,4],[]]。
        retrieve_seq: List[int]
            按顺序要取出的容器编号列表，例如 [1,2,3,4,5]。
        heuristic_fn: Callable
            用户定义的启发式函数，满足接口：
                (bay: Bay, retrieve_seq: List[int]) -> Tuple[str,int,int]
            例如返回 ("move", src_stack, dst_stack) 或 ("retrieve",-1,-1)。

    返回：
        int: 执行完整个 retrieve_seq 过程中累计的重排次数。
    """
    # 1. 深拷贝初始布局，避免原数据被修改
    bay_conf = deepcopy(initial_layout)
    seq = deepcopy(retrieve_seq)

    # 2. 初始化 Bay 对象
    total_items = sum(len(col) for col in initial_layout)
    n_stacks = len(bay_conf)
    initial_height = max(len(col) for col in bay_conf)
    n_tiers = total_items
    bay = Bay(n_stacks=n_stacks, n_tiers=n_tiers, conf=bay_conf)

    rehandles = 0

    # 3. 主循环：直到所有容器都被取走
    while seq:
        target = seq[0]  # 当前要取出的最小编号容器

        # 3.1 在 bay 中定位 target 的位置 (stack_i, tier_j)
        found = False
        for s in range(bay.n_stacks):
            for t in range(bay.h[s]):
                if bay.pri[s][t] == target:
                    stack_i, tier_j = s, t
                    found = True
                    break
            if found:
                break
        if not found:
            # 说明 target 可能在早期步骤就被取走或序列有误
            raise ValueError(f"目标容器 {target} 不在 Bay 当前状态中")

        # 3.2 决策：调用启发式函数
        action_type, src, dst = heuristic_fn(bay, seq)

        if action_type == "retrieve":
            # 确保 target 在顶层
            if tier_j != bay.h[stack_i] - 1:
                raise ValueError(
                    f"启发式函数要求取容器 {target}，但它并不在列 {stack_i} 顶层"
                )
            bay.retrieve_from(stack_i)
            # 取走 target
            bay.pri[stack_i][tier_j] = None
            bay.qlt[stack_i][tier_j] = None
            bay.h[stack_i] -= 1
            # 从 seq 中移除 target
            seq.pop(0)

        elif action_type == "move":
            # 验证 src 列顶层确实有容器可搬
            if bay.h[src] == 0:
                raise ValueError(f"启发式函数要求从空列 {src} 搬动容器")
            moving_tier = bay.h[src] - 1
            moving_container = bay.pri[src][moving_tier]
            if moving_container is None:
                raise ValueError(f"列 {src} 顶层没有容器可移动")
            bay.move_crane_and_pick(src, dst)

            # 验证 dst 列有空间
            if bay.h[dst] >= bay.n_tiers:
                raise ValueError(f"启发式函数要求放到已满列 {dst}")

            # 执行搬动：从 src 移除
            bay.pri[src][moving_tier] = None
            bay.qlt[src][moving_tier] = None
            bay.h[src] -= 1

            # 放到 dst 顶层
            new_tier = bay.h[dst]
            bay.pri[dst][new_tier] = moving_container
            bay.h[dst] += 1
            if bay.h[dst] == 1:
                bay.qlt[dst][new_tier] = moving_container
            else:
                below_min = bay.qlt[dst][new_tier - 1]
                bay.qlt[dst][new_tier] = min(below_min, moving_container)

            rehandles += 1  # 重排次数加 1

        else:
            raise ValueError(f"启发式函数返回未知动作类型 {action_type}")

    return rehandles,float(bay.crane_time)
# 在 simulate.py 的顶部补充必要的 imports



# ------------------------------------------------------------------
# 1) 两种搬迁子例程
# ------------------------------------------------------------------


# … 之前的 import 和 Bay 定义省略 …

# ------------------------------------------------------------------
# 1) 两种搬迁子例程，统一接收剩余序列 seq
# ------------------------------------------------------------------
def restricted_relocation(
    bay: Bay,
    seq: List[int],
    pf: callable,
    skip_next: bool = False
) -> int:
    """
    Algorithm 1 RE / REN:
      - seq[0] 是当前要取的容器 DC
      - skip_next=True 时跳过 seq[1] 所在列（REN 方案）
    返回：本次取 DC 前的搬迁次数
    """
    rehandles = 0
    DC = seq[0]

    # 找到 DC 当前所在列 s
    s = None
    for i in range(bay.n_stacks):
        for t in range(bay.h[i]):
            if bay.pri[i][t] == DC:
                s = i
                break
        if s is not None:
            break

    if s is None:
        # DC 不在 bay 中了，说明要么已经被取走，要么序列错位
        # 这里直接返回 0 表示“无需搬迁”
        return 0

    # 如果是 REN，就把下一个目标所在列加到 forbidden
    forbidden = set()
    if skip_next and len(seq) > 1:
        next_DC = seq[1]
        forbidden.add(
            next(
                j for j in range(bay.n_stacks)
                for t in range(bay.h[j])
                if bay.pri[j][t] == next_DC
            )
        )

    # 只要 DC 还没到栈顶，就把它下面的阻塞容器搬走
    while bay.pri[s][bay.h[s] - 1] != DC:
        moving = bay.pri[s][bay.h[s] - 1]
        # 候选列：非 origin、未满、且不在 forbidden 中
        cands = [
            j for j in range(bay.n_stacks)
            if j != s and bay.h[j] < bay.n_tiers and j not in forbidden
        ]
        if not cands:
            raise ValueError("No relocation candidates — bay is stuck")

        # 按 PF 打分，选最低分列
        best_j, best_score = None, float('inf')
        for j in cands:
            bay.last_dst = j
            try:
                score = float(pf(bay, seq))
                if math.isnan(score):
                    score = float('inf')
            except Exception:
                score = float('inf')
            if score < best_score:
                best_score, best_j = score, j

        # 防御：如果 best_j 还是 None，就随机选一个
        if best_j is None:
            best_j = random.choice(cands)
        bay.move_crane_and_pick(s, best_j)
        # 执行搬迁
        bay.pri[best_j][bay.h[best_j]] = moving
        bay.h[best_j] += 1
        bay.pri[s][bay.h[s] - 1] = None
        bay.h[s] -= 1
        rehandles += 1

    return rehandles

def unrestricted_relocation(bay: Bay, seq: List[int], pf: callable) -> int:
    rehandles = 0
    DC = seq[0]

    # 找到 DC 所在列 s
    s = None
    for i in range(bay.n_stacks):
        for t in range(bay.h[i]):
            if bay.pri[i][t] == DC:
                s = i
                break
        if s is not None:
            break
    if s is None:
        return 0

    # 直到 DC 到顶：反复搬走阻挡箱 moving
    while bay.pri[s][bay.h[s] - 1] != DC:
        moving = bay.pri[s][bay.h[s] - 1]

        # 1) 先按 PF 选一个候选目的列
        cands = [j for j in range(bay.n_stacks) if j != s and bay.h[j] < bay.n_tiers]
        if not cands:
            raise ValueError("No relocation candidates — bay is stuck")

        best_j, best_score = None, float('inf')
        for j in cands:
            try:
                sc = float(pf(bay, seq, j))
                if math.isnan(sc):
                    sc = float('inf')
            except Exception:
                sc = float('inf')
            if sc < best_score:
                best_score, best_j = sc, j
        if best_j is None:
            best_j = random.choice(cands)

        # 2) 预清理 best_j：把会压住 moving 的更小ID箱先搬走
        def top_id(idx):
            return bay.pri[idx][bay.h[idx]-1] if bay.h[idx] > 0 else None
        def min_id_in_stack(idx):
            vals = [c for c in bay.pri[idx] if c is not None]
            return min(vals) if vals else float('inf')

        while bay.h[best_j] > 0 and min_id_in_stack(best_j) < moving:
            # 把 best_j 顶部 top_j 搬到其他列 k（k 由 PF 决定，且 k != s, k != best_j）
            top_j = top_id(best_j)
            k_cands = [k for k in range(bay.n_stacks)
                       if k != best_j and k != s and bay.h[k] < bay.n_tiers]
            if not k_cands:
                break  # 没法清了，就先容忍（退化为 RE 的一次放置）

            k_best, k_score = None, float('inf')
            for k in k_cands:
                try:
                    sc = float(pf(bay, seq, k))
                    if math.isnan(sc):
                        sc = float('inf')
                except Exception:
                    sc = float('inf')
                if sc < k_score:
                    k_score, k_best = sc, k
            if k_best is None:
                k_best = random.choice(k_cands)

            # 执行：从 best_j 弹出 top_j 放到 k_best
            bay.move_crane_and_pick(best_j, k_best)
            # pop from best_j
            bj_top_tier = bay.h[best_j] - 1
            bay.pri[best_j][bj_top_tier] = None
            bay.h[best_j] -= 1
            # push to k_best
            new_tier = bay.h[k_best]
            bay.pri[k_best][new_tier] = top_j
            bay.h[k_best] += 1
            # 计作一次重排
            rehandles += 1

        # 3) 把 moving 从 s 放到 best_j
        bay.move_crane_and_pick(s, best_j)
        src_top = bay.h[s] - 1
        # pop moving
        bay.pri[s][src_top] = None
        bay.h[s] -= 1
        # push moving
        dst_tier = bay.h[best_j]
        bay.pri[best_j][dst_tier] = moving
        bay.h[best_j] += 1
        rehandles += 1

        # 如果把 moving 放到了 best_j，DC 仍在列 s；继续循环直到 DC 到顶

    return rehandles



# ------------------------------------------------------------------
# 2) 修改 apply_relocation_scheme，按剩余序列调用
# ------------------------------------------------------------------
def apply_relocation_scheme(
    instances: List[ Tuple[List[List[int]], List[int]] ],
    scheme: str,
    pf: callable,
    max_steps: int = 500000
) -> Tuple[float,float]:
    totals = []
    times = []
    for conf, seq in instances:
        total_items = sum(len(col) for col in conf)
        bay = Bay(len(conf), total_items, deepcopy(conf))
        rehandles = 0
        steps     = 0

        # 按 idx 遍历 seq，传入剩余序列 remaining
        for idx in range(len(seq)):
            remaining = seq[idx:]
            bay.current_target = remaining[0]

            try:
                if scheme == 'RE':
                    cnt = restricted_relocation(bay, remaining, pf, skip_next=False)
                elif scheme == 'REN':
                    cnt = restricted_relocation(bay, remaining, pf, skip_next=True)
                else:  # 默认当作 UN
                    cnt = unrestricted_relocation(bay, remaining, pf)
            except ValueError:
                rehandles = PENALTY
                break

            rehandles += cnt
            steps     += cnt

            # 然后正式“取走”目标箱
            found = False
            for s_idx in range(bay.n_stacks):
                if bay.h[s_idx] > 0 and bay.pri[s_idx][bay.h[s_idx] - 1] == remaining[0]:
                    bay.retrieve_from(s_idx)
                    # 找到了，就执行取箱
                    bay.pri[s_idx][bay.h[s_idx] - 1] = None
                    bay.h[s_idx] -= 1
                    found = True
                    break
            if not found:
                # DC 不在任何堆列顶部，跳过（可能已被提前取走）
                # 也可以打印日志：print(f"Warning: target {remaining[0]} not found")
                pass

            if steps > max_steps:
                rehandles = PENALTY
                break
        times.append(bay.crane_time)
        totals.append(rehandles)

    return float(sum(totals)), float(sum(times))
