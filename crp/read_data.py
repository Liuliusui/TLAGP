from copy import deepcopy
from typing import List, Tuple
import os

# 1) Function to load a CRP instance from a .dat file
def load_instance_from_dat(file_path: str) -> Tuple[List[List[int]], List[int]]:
    """
    Parse a .dat file where:
      - Line 1: W N
      - Next W lines: H c1 c2 … cH  (H containers bottom to top)
    Assumes retrieval sequence is [1..N].
    Returns:
      initial_layout: List of W stacks (each a list of container IDs from bottom to top)
      retrieve_seq:   List of container IDs in ascending order [1,2,...,N]
    """
    with open(file_path, 'r') as f:
        lines = f.read().strip().splitlines()

    # First line: number of stacks (W) and total containers (N)
    W, N = map(int, lines[0].split())

    # Next W lines: each describes one stack
    initial_layout = []
    for i in range(1, 1 + W):
        parts = list(map(int, lines[i].split()))
        height = parts[0]
        containers = parts[1 : 1 + height]
        initial_layout.append(containers)

    # Retrieval sequence is [1,2,...,N]
    retrieve_seq = list(range(1, N + 1))
    return initial_layout, retrieve_seq
