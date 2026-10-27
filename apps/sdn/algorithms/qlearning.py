"""Reinforcement Learning sederhana — Q-Learning untuk routing adaptif."""
import time
import random


def run(G, source, target, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2,
        max_steps=40, seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    Q = {n: {v: 0.0 for v in G.neighbors(n)} for n in G.nodes()}
    first_success_episode = None

    for ep in range(1, episodes + 1):
        state = source
        steps = 0
        while state != target and steps < max_steps:
            actions = list(Q[state].keys())
            if not actions:
                break
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            reward = -G[state][action]["delay"]
            next_state = action
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            Q[state][action] += alpha * (reward + gamma * max_next_q - Q[state][action])

            state = next_state
            steps += 1

        if state == target and first_success_episode is None:
            first_success_episode = ep

    # ekstraksi jalur terbaik dari Q-table hasil pembelajaran
    path, current, visited = [source], source, {source}
    success = False
    for _ in range(max_steps):
        if current == target:
            success = True
            break
        if not Q[current]:
            break
        best_action = max(Q[current], key=Q[current].get)
        if best_action in visited:
            break
        path.append(best_action)
        visited.add(best_action)
        current = best_action
    success = success or (path[-1] == target)

    runtime_ms = (time.perf_counter() - t0) * 1000
    if not success:
        return {"path": None, "cost": None, "runtime_ms": round(runtime_ms, 4),
                "meta": {"note": "kebijakan belum konvergen ke target, coba tambah episodes"}}

    cost = sum(G[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))
    return {
        "path": path,
        "cost": round(cost, 3),
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "episode_dijalankan": episodes,
            "episode_sukses_pertama": first_success_episode,
            "langkah_kebijakan_akhir": len(path) - 1,
        },
    }
