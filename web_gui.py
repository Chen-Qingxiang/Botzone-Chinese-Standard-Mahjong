import copy
import math
import random

import torch
from flask import Flask, jsonify, request, Response

from mahjong_agent.engine.env import MahjongEnv
from mahjong_agent.engine.actions import ActionType
from mahjong_agent.features import encode_action_v2, encode_observation_v2
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_model_from_checkpoint


HUMAN = 0

app = Flask(__name__)

print("正在加载模型……")
model, metadata = load_model_from_checkpoint("artifacts/botzone_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
coach = ModelPolicy(model)
print("模型已加载：", device)

env = None


def tile_name(tile):
    if tile < 9:
        return f"{tile + 1}万"
    if tile < 18:
        return f"{tile - 8}条"
    if tile < 27:
        return f"{tile - 17}筒"

    names = ["东", "南", "西", "北", "中", "发", "白"]
    return names[tile - 27]


def action_name(action):
    if action.kind == ActionType.PASS:
        return "过"

    if action.kind == ActionType.PLAY:
        return f"打 {tile_name(action.tile)}"

    if action.kind == ActionType.CHI:
        seq = " ".join(tile_name(x) for x in action.sequence)
        return f"吃 [{seq}] → 打 {tile_name(action.discard)}"

    if action.kind == ActionType.PENG:
        return f"碰 {tile_name(action.tile)} → 打 {tile_name(action.discard)}"

    if action.kind == ActionType.GANG:
        return f"杠 {tile_name(action.tile)}"

    if action.kind == ActionType.BUGANG:
        return f"补杠 {tile_name(action.tile)}"

    if action.kind == ActionType.HU:
        return "胡"

    return str(action.kind)



def _model_outputs(observation, legal):
    """
    对当前所有合法动作一次性做模型前向计算。
    返回模型偏好以及内部打分拆解。
    """
    model = coach.model
    model.eval()
    device = next(model.parameters()).device

    feature_tokens, feature_mask = encode_observation_v2(observation)

    encoded_actions = [
        encode_action_v2(action)
        for action in legal
    ]

    action_tokens = [
        item[0]
        for item in encoded_actions
    ]

    action_token_masks = [
        item[1]
        for item in encoded_actions
    ]

    with torch.no_grad():
        output = model(
            torch.tensor(
                [feature_tokens],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [action_tokens],
                dtype=torch.float32,
                device=device,
            ),
            action_mask=torch.ones(
                (1, len(legal)),
                dtype=torch.bool,
                device=device,
            ),
            feature_mask=torch.tensor(
                [feature_mask],
                dtype=torch.bool,
                device=device,
            ),
            action_token_mask=torch.tensor(
                [action_token_masks],
                dtype=torch.bool,
                device=device,
            ),
        )

        logits = output["logits"][0, :len(legal)]
        probabilities = torch.softmax(logits, dim=-1)

        family = output["family_logits"][0, :len(legal)]
        tactical = output["tactical_logits"][0, :len(legal)]

        action_outcome = output["action_outcome"][0, :len(legal)]
        action_fan_logits = output["action_fan_logits"][0, :len(legal)]

        fan_values = torch.arange(
            5,
            dtype=action_fan_logits.dtype,
            device=device,
        )

        expected_fan_bucket = (
            action_fan_logits.softmax(-1)
            * fan_values
        ).sum(-1)

        auxiliary = torch.stack(
            (
                action_outcome[:, 0],
                -action_outcome[:, 1],
                action_outcome[:, 2],
                action_outcome[:, 3],
                expected_fan_bucket,
            ),
            dim=-1,
        )

        aux_correction = (
            auxiliary
            * model.auxiliary_logit_gate
        ).sum(-1)

        win_tendency = torch.sigmoid(
            action_outcome[:, 0]
        )

        dealin_tendency = torch.sigmoid(
            action_outcome[:, 1]
        )

        eightfan_tendency = torch.sigmoid(
            action_outcome[:, 3]
        )

    rows = []

    hu_index = next(
        (
            i
            for i, action in enumerate(legal)
            if action.kind == ActionType.HU
        ),
        None,
    )

    for i, action in enumerate(legal):
        probability = float(probabilities[i])

        # ModelPolicy 本身遇到合法 HU 会强制胡。
        if hu_index is not None:
            probability = 1.0 if i == hu_index else 0.0

        rows.append({
            "index": i,
            "name": action_name(action),
            "kind": action.kind.name,
            "probability": probability,
            "logit": float(logits[i]),
            "family": float(family[i]),
            "tactical": float(tactical[i]),
            "aux": float(aux_correction[i]),
            "win": float(win_tendency[i]),
            "dealin": float(dealin_tendency[i]),
            "score": float(action_outcome[i, 2]),
            "eightfan": float(eightfan_tendency[i]),
            "fan_bucket": float(expected_fan_bucket[i]),
        })

    rows.sort(
        key=lambda row: row["probability"],
        reverse=True,
    )

    return rows


def current_ai_rows():
    if env is None:
        return []

    if env.is_terminal():
        return []

    if env.current_player != HUMAN:
        return []

    legal = env.legal_actions(HUMAN)

    if not legal:
        return []

    return _model_outputs(
        env.observe(HUMAN),
        legal,
    )


def _determinize_hidden(work_env, seed):
    """
    保持 HUMAN 可见信息不变，
    将三家暗手 + 牌墙重新随机分配。
    """
    rng = random.Random(seed)

    pool = list(work_env.wall)
    sizes = {}

    for player in range(1, 4):
        sizes[player] = len(work_env.hands[player])
        pool.extend(work_env.hands[player])

    rng.shuffle(pool)

    pos = 0

    for player in range(1, 4):
        size = sizes[player]

        work_env.hands[player] = sorted(
            pool[pos:pos + size]
        )

        pos += size

    work_env.wall = list(pool[pos:])


def _rollout_many_to_end(envs, max_steps=512):
    """
    把大量反事实环境一起 batch 到 GPU。
    """
    steps = [0] * len(envs)

    while True:
        active = [
            i
            for i, item in enumerate(envs)
            if not item.is_terminal()
            and steps[i] < max_steps
        ]

        if not active:
            break

        observations = []
        legal_batches = []

        for i in active:
            item = envs[i]
            player = item.current_player

            legal = item.legal_actions(player)

            if not legal:
                raise RuntimeError(
                    f"rollout {i}: "
                    f"player={player}, "
                    f"phase={item.phase}, "
                    "没有合法动作"
                )

            observations.append(
                item.observe(player)
            )

            legal_batches.append(legal)

        actions = coach.batch_act(
            observations,
            legal_batches,
        )

        for i, action in zip(active, actions):
            envs[i].step(action)
            steps[i] += 1

    for i, item in enumerate(envs):
        if not item.is_terminal():
            raise RuntimeError(
                f"rollout {i} 超过最大步数"
            )

    return [
        item.result()
        for item in envs
    ]


def _mean(values):
    return sum(values) / max(1, len(values))


def _paired_ci95(values):
    if len(values) < 2:
        value = _mean(values)
        return value, value

    mean = _mean(values)

    variance = sum(
        (x - mean) ** 2
        for x in values
    ) / (len(values) - 1)

    se = math.sqrt(
        variance / len(values)
    )

    return (
        mean - 1.96 * se,
        mean + 1.96 * se,
    )


def compare_actions_json(idx_a, idx_b, samples=64):
    if env.is_terminal():
        raise ValueError("本局已经结束")

    if env.current_player != HUMAN:
        raise ValueError("当前不是你的决策点")

    if env.phase not in ("discard", "claim"):
        raise ValueError(
            f"当前阶段 {env.phase} 暂不支持比较"
        )

    if (
        env.phase == "claim"
        and (
            env.pending_bugang is not None
            or env.claim_hu_only
        )
    ):
        raise ValueError(
            "目前暂不支持抢补杠阶段的反事实比较"
        )

    legal = env.legal_actions(HUMAN)

    if not (
        0 <= idx_a < len(legal)
        and 0 <= idx_b < len(legal)
    ):
        raise ValueError("动作编号超出范围")

    if idx_a == idx_b:
        raise ValueError("请选择两个不同动作")

    action_a = legal[idx_a]
    action_b = legal[idx_b]

    worlds_a = []
    worlds_b = []

    base_seed = (
        730000
        + len(env.events) * 1000
        + len(env.wall)
    )

    for sample in range(samples):
        base = copy.deepcopy(env)

        if env.phase == "claim":
            # 清除 HUMAN 不应该知道的、
            # 尚未裁决的其他 AI claim。
            base.claim_responses = {}

            if HUMAN not in base.pending_claimers:
                raise RuntimeError(
                    "HUMAN 不在 pending_claimers 中"
                )

            base.current_player = (
                base.pending_claimers[0]
            )

        _determinize_hidden(
            base,
            base_seed + sample,
        )

        if env.phase == "claim":

            safety = 0

            # 重新让 HUMAN 之前的 responder
            # 在新的隐藏世界里作答。
            while (
                not base.is_terminal()
                and base.phase == "claim"
                and base.current_player != HUMAN
            ):
                player = base.current_player
                obs = base.observe(player)
                pre_legal = base.legal_actions(player)

                ai_action = coach.act(
                    obs,
                    pre_legal,
                )

                base.step(ai_action)

                safety += 1

                if safety > 3:
                    raise RuntimeError(
                        "claim replay 未能轮到 HUMAN"
                    )

            if base.is_terminal():
                raise RuntimeError(
                    "claim replay 在 HUMAN 决策前终局"
                )

            if base.current_player != HUMAN:
                raise RuntimeError(
                    "claim replay 未恢复到 HUMAN"
                )

        legal_now = {
            action.key(): action
            for action
            in base.legal_actions(HUMAN)
        }

        if action_a.key() not in legal_now:
            raise RuntimeError(
                "动作 A 在随机世界中不再合法"
            )

        if action_b.key() not in legal_now:
            raise RuntimeError(
                "动作 B 在随机世界中不再合法"
            )

        world_a = copy.deepcopy(base)
        world_b = copy.deepcopy(base)

        legal_a = {
            action.key(): action
            for action
            in world_a.legal_actions(HUMAN)
        }

        legal_b = {
            action.key(): action
            for action
            in world_b.legal_actions(HUMAN)
        }

        world_a.step(
            legal_a[action_a.key()]
        )

        world_b.step(
            legal_b[action_b.key()]
        )

        worlds_a.append(world_a)
        worlds_b.append(world_b)

    results = _rollout_many_to_end(
        worlds_a + worlds_b
    )

    results_a = results[:samples]
    results_b = results[samples:]

    def summarize(result):
        return {
            "score": result["scores"][HUMAN],
            "win": int(
                result["winner"] == HUMAN
            ),
            "dealin": int(
                result["loser"] == HUMAN
            ),
            "draw": int(
                result["winner"] is None
            ),
            "fan": (
                result["fan_count"]
                if result["winner"] == HUMAN
                else 0
            ),
        }

    records_a = [
        summarize(x)
        for x in results_a
    ]

    records_b = [
        summarize(x)
        for x in results_b
    ]

    def aggregate(records):
        wins = [
            row["fan"]
            for row in records
            if row["win"]
        ]

        return {
            "score": _mean(
                [x["score"] for x in records]
            ),
            "win": _mean(
                [x["win"] for x in records]
            ),
            "dealin": _mean(
                [x["dealin"] for x in records]
            ),
            "draw": _mean(
                [x["draw"] for x in records]
            ),
            "fan": (
                _mean(wins)
                if wins
                else 0.0
            ),
        }

    a = aggregate(records_a)
    b = aggregate(records_b)

    deltas = [
        x["score"] - y["score"]
        for x, y in zip(
            records_a,
            records_b,
        )
    ]

    delta = _mean(deltas)
    ci_low, ci_high = _paired_ci95(deltas)

    if ci_low > 0:
        interpretation = (
            "这批可能世界中，A 的 rollout "
            "表现较稳定地高于 B。"
        )
    elif ci_high < 0:
        interpretation = (
            "这批可能世界中，B 的 rollout "
            "表现较稳定地高于 A。"
        )
    else:
        interpretation = (
            "目前样本下差异仍有较大噪声，"
            "暂时不能仅凭 rollout 区分两者。"
        )

    return {
        "samples": samples,
        "action_a": action_name(action_a),
        "action_b": action_name(action_b),
        "a": a,
        "b": b,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "interpretation": interpretation,
    }


def advance_ai():
    """
    让 AI 一直打，直到：
    - 轮到 HUMAN 决策
    - 或本局结束
    """
    safety = 0

    while not env.is_terminal() and env.current_player != HUMAN:
        player = env.current_player
        obs = env.observe(player)
        legal = env.legal_actions(player)

        if not legal:
            raise RuntimeError(
                f"玩家 {player} 没有合法动作，phase={env.phase}"
            )

        action = coach.act(obs, legal)
        env.step(action)

        safety += 1
        if safety > 500:
            raise RuntimeError("AI 自动推进超过安全步数")


def meld_json(meld):
    return {
        "kind": meld.kind.name,
        "tiles": [tile_name(x) for x in meld.tiles],
        "from_player": meld.from_player,
    }


def state_json():
    legal = []

    if (
        not env.is_terminal()
        and env.current_player == HUMAN
    ):
        actions = env.legal_actions(HUMAN)

        for i, action in enumerate(actions):
            legal.append({
                "index": i,
                "name": action_name(action),
                "kind": action.kind.name,
                "tile": (
                    action.tile
                    if action.tile is not None
                    else -1
                ),
                "tile_name": (
                    tile_name(action.tile)
                    if action.tile is not None
                    and action.tile >= 0
                    else None
                ),
            })

    last_discard = None
    if env.last_discard is not None:
        p, t = env.last_discard
        last_discard = {
            "player": p,
            "tile": tile_name(t),
        }

    players = []

    for p in range(4):
        players.append({
            "id": p,
            "hand_count": len(env.hands[p]),
            "river": [
                tile_name(x)
                for x in env.discards[p]
            ],
            "melds": [
                meld_json(m)
                for m in env.melds[p]
            ],
        })

    result = env.result() if env.is_terminal() else None

    return {
        "terminal": env.is_terminal(),
        "phase": env.phase,
        "current_player": env.current_player,
        "wall_remaining": len(env.wall),
        "last_discard": last_discard,
        "players": players,

        # 只把 HUMAN 的暗手送给浏览器。
        "hand": [
            {
                "tile": tile,
                "name": tile_name(tile),
            }
            for tile in env.hands[HUMAN]
        ],

        "legal_actions": legal,
        "ai_rankings": current_ai_rows(),
        "result": result,
    }


def new_game():
    global env
    seed = random.randrange(1, 10**9)
    env = MahjongEnv()
    env.reset(seed=seed)

    # 当前实现通常庄家就是玩家0。
    # 如果以后环境改变，也自动推进到玩家0。
    advance_ai()


HTML = r'''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>麻将 AI 教练</title>

<style>
    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        background: #123b2b;
        color: #f3f3f3;
        font-family:
            -apple-system, BlinkMacSystemFont,
            "Segoe UI", "Microsoft YaHei",
            sans-serif;
    }

    button {
        font: inherit;
    }

    #app {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }

    .topbar {
        height: 58px;
        display: flex;
        align-items: center;
        padding: 0 22px;
        background: rgba(0,0,0,.25);
        border-bottom: 1px solid rgba(255,255,255,.12);
        gap: 22px;
    }

    .title {
        font-size: 20px;
        font-weight: 700;
    }

    .status {
        color: #d3ddd8;
        font-size: 14px;
    }

    .new-game {
        margin-left: auto;
        border: 0;
        padding: 9px 16px;
        border-radius: 8px;
        cursor: pointer;
        background: #f3f3f3;
        color: #173428;
        font-weight: 600;
    }

    .table-area {
        flex: 1;
        display: grid;
        grid-template-columns: 190px 1fr 190px;
        grid-template-rows: 170px 1fr 210px;
        gap: 12px;
        padding: 15px 390px 15px 15px;
        min-height: 670px;
    }

    .player {
        background: rgba(0,0,0,.13);
        border-radius: 12px;
        padding: 10px;
        overflow: auto;
    }

    .p2 {
        grid-column: 2;
        grid-row: 1;
    }

    .p3 {
        grid-column: 1;
        grid-row: 2;
    }

    .p1 {
        grid-column: 3;
        grid-row: 2;
    }

    .center {
        grid-column: 2;
        grid-row: 2;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .center-box {
        min-width: 280px;
        text-align: center;
        padding: 28px;
        border-radius: 18px;
        background: rgba(0,0,0,.22);
        box-shadow: 0 10px 35px rgba(0,0,0,.18);
    }

    .human {
        grid-column: 1 / 4;
        grid-row: 3;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
    }

    .player-title {
        font-weight: 700;
        margin-bottom: 8px;
    }

    .small {
        font-size: 12px;
        color: #b9cac2;
    }

    .river,
    .meld-row {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        margin-top: 6px;
    }

    .mini-tile {
        background: #f6f3e8;
        color: #1e2522;
        border-radius: 3px;
        min-width: 28px;
        height: 38px;
        padding: 0 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 650;
        font-size: 13px;
        box-shadow:
            0 2px 0 #b8b3a5,
            0 3px 5px rgba(0,0,0,.22);
    }

    .meld {
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 6px;
        padding: 4px;
        margin-top: 5px;
    }

    .hand {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 4px;
        min-height: 86px;
    }

    .tile {
        width: 52px;
        height: 72px;
        border: 0;
        border-radius: 5px;
        background: linear-gradient(#fffdf6, #e9e5d8);
        color: #161b18;
        font-size: 18px;
        font-weight: 700;
        cursor: default;
        box-shadow:
            0 4px 0 #aaa595,
            0 6px 9px rgba(0,0,0,.28);
        transition:
            transform .11s ease,
            box-shadow .11s ease;
    }

    .tile.playable {
        cursor: pointer;
    }

    .tile.playable:hover {
        transform: translateY(-10px);
        box-shadow:
            0 4px 0 #aaa595,
            0 12px 15px rgba(0,0,0,.30);
    }

    .actions {
        min-height: 56px;
        display: flex;
        gap: 8px;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }

    .action-btn {
        padding: 8px 13px;
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 7px;
        background: rgba(255,255,255,.13);
        color: white;
        cursor: pointer;
    }

    .action-btn:hover {
        background: rgba(255,255,255,.25);
    }

    .human-label {
        margin: 6px 0 8px;
        font-weight: 700;
    }

    .terminal {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
    }

    .last {
        color: #ffdd8b;
        margin-top: 8px;
    }


    .coach-panel {
        position: fixed;
        top: 72px;
        right: 14px;
        bottom: 14px;
        width: 355px;
        padding: 15px;
        border-radius: 14px;
        background: rgba(4, 32, 23, .96);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 10px 30px rgba(0,0,0,.28);
        overflow-y: auto;
        z-index: 20;
    }

    .coach-title {
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 3px;
    }

    .coach-subtitle {
        font-size: 12px;
        color: #9db8ac;
        margin-bottom: 14px;
    }

    .coach-row {
        display: grid;
        grid-template-columns: 24px 1fr 62px;
        gap: 7px;
        align-items: center;
        padding: 7px 5px;
        border-bottom: 1px solid rgba(255,255,255,.07);
    }

    .coach-row:first-child {
        background: rgba(255,255,255,.07);
        border-radius: 7px;
    }

    .coach-action {
        font-size: 13px;
        line-height: 1.25;
    }

    .coach-prob {
        font-size: 13px;
        font-weight: 700;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .coach-check {
        width: 17px;
        height: 17px;
        cursor: pointer;
    }

    .coach-controls {
        display: flex;
        gap: 7px;
        margin-top: 14px;
    }

    .coach-btn {
        flex: 1;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 7px;
        padding: 8px 7px;
        background: rgba(255,255,255,.11);
        color: white;
        cursor: pointer;
    }

    .coach-btn:hover {
        background: rgba(255,255,255,.20);
    }

    .coach-btn:disabled {
        opacity: .45;
        cursor: wait;
    }

    .coach-details {
        margin-top: 15px;
        font-size: 12px;
        line-height: 1.5;
    }

    .analysis-table,
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        font-size: 11px;
    }

    .analysis-table th,
    .analysis-table td,
    .compare-table th,
    .compare-table td {
        padding: 4px 3px;
        border-bottom: 1px solid rgba(255,255,255,.07);
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .analysis-table th:first-child,
    .analysis-table td:first-child,
    .compare-table th:first-child,
    .compare-table td:first-child {
        text-align: left;
    }

    .coach-section-title {
        margin-top: 12px;
        font-weight: 700;
        color: #ecf5f1;
    }

    .coach-warning {
        margin-top: 9px;
        color: #a9bdb4;
        font-size: 11px;
    }

    .coach-loading {
        padding: 16px 0;
        color: #d2e4dc;
    }

    @media (max-width: 900px) {
        .table-area {
            grid-template-columns: 130px 1fr 130px;
        }

        .tile {
            width: 43px;
            height: 64px;
            font-size: 15px;
        }
    }
</style>
</head>

<body>
<div id="app">

    <div class="topbar">
        <div class="title">麻将 AI 教练</div>
        <div class="status" id="topStatus"></div>
        <button class="new-game" onclick="newGame()">新一局</button>
    </div>

    <div class="table-area">

        <div class="player p2" id="player2"></div>

        <div class="player p3" id="player3"></div>

        <div class="center">
            <div class="center-box">
                <div id="centerInfo"></div>
            </div>
        </div>

        <div class="player p1" id="player1"></div>

        <div class="human">

            <div class="actions" id="actions"></div>

            <div class="human-label">你的手牌</div>

            <div class="hand" id="hand"></div>

            <div class="small" style="margin-top:12px">
                玩家 0 · 你
            </div>

        </div>

    </div>

    <aside class="coach-panel">
        <div class="coach-title">AI 教练</div>

        <div class="coach-subtitle">
            百分比是模型策略偏好，不是和牌率或胜率。
            勾选两个动作可以做反事实比较。
        </div>

        <div id="coachRankings"></div>

        <div class="coach-controls">
            <button
                class="coach-btn"
                id="analysisBtn"
                onclick="loadAnalysis()"
            >
                模型拆解
            </button>

            <button
                class="coach-btn"
                id="compareBtn"
                onclick="runCompare()"
            >
                比较选中
            </button>
        </div>

        <div
            class="coach-details"
            id="coachDetails"
        ></div>
    </aside>

</div>

<script>

let state = null;
let compareSelection = [];


async function api(url, options={}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json"
        },
        ...options
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "请求失败");
    }

    return data;
}


function miniTile(name) {
    const d = document.createElement("div");
    d.className = "mini-tile";
    d.textContent = name;
    return d;
}


function renderPlayer(player) {

    const box = document.getElementById(
        "player" + player.id
    );

    box.innerHTML = "";

    const title = document.createElement("div");
    title.className = "player-title";
    title.textContent =
        `玩家 ${player.id} · 暗牌 ${player.hand_count} 张`;

    box.appendChild(title);

    const riverTitle = document.createElement("div");
    riverTitle.className = "small";
    riverTitle.textContent = "牌河";
    box.appendChild(riverTitle);

    const river = document.createElement("div");
    river.className = "river";

    for (const tile of player.river) {
        river.appendChild(miniTile(tile));
    }

    if (player.river.length === 0) {
        river.innerHTML =
            '<span class="small">（空）</span>';
    }

    box.appendChild(river);

    if (player.melds.length > 0) {
        const label = document.createElement("div");
        label.className = "small";
        label.style.marginTop = "9px";
        label.textContent = "副露";
        box.appendChild(label);

        for (const meld of player.melds) {
            const wrap = document.createElement("div");
            wrap.className = "meld";

            const row = document.createElement("div");
            row.className = "meld-row";

            for (const tile of meld.tiles) {
                row.appendChild(miniTile(tile));
            }

            wrap.appendChild(row);
            box.appendChild(wrap);
        }
    }
}


function playAction(index) {
    api("/api/action", {
        method: "POST",
        body: JSON.stringify({index})
    })
    .then(data => {
        state = data;
        compareSelection = [];
        document.getElementById("coachDetails").innerHTML = "";
        render();
    })
    .catch(err => alert(err.message));
}


function renderHand() {

    const hand = document.getElementById("hand");
    hand.innerHTML = "";

    const playMap = new Map();

    for (const action of state.legal_actions) {
        if (action.kind === "PLAY") {
            playMap.set(action.tile, action.index);
        }
    }

    for (const item of state.hand) {

        const btn = document.createElement("button");
        btn.className = "tile";
        btn.textContent = item.name;

        if (playMap.has(item.tile)) {
            btn.classList.add("playable");

            const index = playMap.get(item.tile);

            btn.onclick = () => playAction(index);
        }

        hand.appendChild(btn);
    }
}


function renderActions() {

    const box = document.getElementById("actions");
    box.innerHTML = "";

    for (const action of state.legal_actions) {

        // 普通 PLAY 已经可以直接点牌，不再重复显示按钮。
        if (action.kind === "PLAY") {
            continue;
        }

        const btn = document.createElement("button");
        btn.className = "action-btn";
        btn.textContent = action.name;

        btn.onclick = () =>
            playAction(action.index);

        box.appendChild(btn);
    }
}



function toggleCompare(index, checked) {

    if (checked) {

        if (!compareSelection.includes(index)) {

            if (compareSelection.length >= 2) {
                alert("一次选择两个动作进行比较。");
                renderCoach();
                return;
            }

            compareSelection.push(index);
        }

    } else {

        compareSelection =
            compareSelection.filter(
                x => x !== index
            );
    }
}


function renderCoach() {

    const box =
        document.getElementById("coachRankings");

    box.innerHTML = "";

    if (
        !state
        || state.terminal
        || !state.ai_rankings
        || state.ai_rankings.length === 0
    ) {
        box.innerHTML =
            '<div class="small">当前没有需要你决策的动作。</div>';
        return;
    }

    state.ai_rankings.forEach(
        (row, rank) => {

            const div =
                document.createElement("div");

            div.className = "coach-row";

            const checked =
                compareSelection.includes(
                    row.index
                );

            div.innerHTML = `
                <input
                    class="coach-check"
                    type="checkbox"
                    ${checked ? "checked" : ""}
                >

                <div class="coach-action">
                    ${rank + 1}. ${row.name}
                    ${rank === 0 ? " ← 推荐" : ""}
                </div>

                <div class="coach-prob">
                    ${(row.probability * 100).toFixed(2)}%
                </div>
            `;

            const checkbox =
                div.querySelector("input");

            checkbox.onchange = () => {
                toggleCompare(
                    row.index,
                    checkbox.checked
                );
            };

            box.appendChild(div);
        }
    );
}


async function loadAnalysis() {

    const details =
        document.getElementById("coachDetails");

    const button =
        document.getElementById("analysisBtn");

    button.disabled = true;

    details.innerHTML =
        '<div class="coach-loading">正在读取模型内部打分……</div>';

    try {

        const data =
            await api("/api/analysis");

        let html = `
            <div class="coach-section-title">
                模型内部决策拆解
            </div>

            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>动作</th>
                        <th>偏好</th>
                        <th>总分</th>
                        <th>战术</th>
                        <th>类别</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.rows.slice(0, 8).forEach(row => {
            html += `
                <tr>
                    <td>${row.name}</td>
                    <td>${(row.probability * 100).toFixed(1)}%</td>
                    <td>${row.logit.toFixed(2)}</td>
                    <td>${row.tactical.toFixed(2)}</td>
                    <td>${row.family.toFixed(2)}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;

        if (data.rows.length > 0) {

            const best = data.rows[0];

            html += `
                <div class="coach-section-title">
                    第一名辅助预测
                </div>

                <div>
                    Win 倾向：
                    ${(best.win * 100).toFixed(1)}%
                    <br>

                    放铳倾向：
                    ${(best.dealin * 100).toFixed(1)}%
                    <br>

                    8番倾向：
                    ${(best.eightfan * 100).toFixed(1)}%
                    <br>

                    得分预测：
                    ${best.score.toFixed(3)}
                </div>
            `;
        }

        html += `
            <div class="coach-warning">
                内部预测头并非校准后的真实概率；
                tactical score 是神经网络学到的综合表征。
            </div>
        `;

        details.innerHTML = html;

    } catch (err) {

        details.innerHTML =
            `<div>${err.message}</div>`;

    } finally {

        button.disabled = false;
    }
}


async function runCompare() {

    if (compareSelection.length !== 2) {
        alert("请先勾选两个候选动作。");
        return;
    }

    const details =
        document.getElementById("coachDetails");

    const button =
        document.getElementById("compareBtn");

    button.disabled = true;

    const [a, b] = compareSelection;

    details.innerHTML = `
        <div class="coach-loading">
            正在跑 64 个配对可能世界……<br>
            GPU 会同时推进 128 条后续路线。
        </div>
    `;

    try {

        const data = await api(
            "/api/compare",
            {
                method: "POST",
                body: JSON.stringify({
                    a,
                    b,
                    samples: 64
                })
            }
        );

        details.innerHTML = `
            <div class="coach-section-title">
                反事实比较
            </div>

            <div style="margin-top:7px">
                A：${data.action_a}<br>
                B：${data.action_b}
            </div>

            <table class="compare-table">
                <thead>
                    <tr>
                        <th></th>
                        <th>A</th>
                        <th>B</th>
                    </tr>
                </thead>

                <tbody>
                    <tr>
                        <td>平均得分</td>
                        <td>${data.a.score.toFixed(2)}</td>
                        <td>${data.b.score.toFixed(2)}</td>
                    </tr>

                    <tr>
                        <td>和牌率</td>
                        <td>${(data.a.win * 100).toFixed(1)}%</td>
                        <td>${(data.b.win * 100).toFixed(1)}%</td>
                    </tr>

                    <tr>
                        <td>放铳率</td>
                        <td>${(data.a.dealin * 100).toFixed(1)}%</td>
                        <td>${(data.b.dealin * 100).toFixed(1)}%</td>
                    </tr>

                    <tr>
                        <td>流局率</td>
                        <td>${(data.a.draw * 100).toFixed(1)}%</td>
                        <td>${(data.b.draw * 100).toFixed(1)}%</td>
                    </tr>

                    <tr>
                        <td>和牌平均番</td>
                        <td>${data.a.fan.toFixed(1)}</td>
                        <td>${data.b.fan.toFixed(1)}</td>
                    </tr>
                </tbody>
            </table>

            <div class="coach-section-title">
                A − B = ${data.delta >= 0 ? "+" : ""}
                ${data.delta.toFixed(2)}
            </div>

            <div>
                粗略 95% CI：
                [${data.ci_low.toFixed(2)},
                ${data.ci_high.toFixed(2)}]
            </div>

            <div style="margin-top:8px">
                ${data.interpretation}
            </div>

            <div class="coach-warning">
                隐藏牌目前根据公开信息无条件随机化，
                尚未根据对手历史打法建立后验分布。
            </div>
        `;

    } catch (err) {

        details.innerHTML =
            `<div>${err.message}</div>`;

    } finally {

        button.disabled = false;
    }
}


function renderCenter() {

    const box = document.getElementById("centerInfo");

    if (state.terminal) {

        const r = state.result;

        if (r.winner === null) {
            box.innerHTML = `
                <div class="terminal">流局</div>
            `;
        } else {
            box.innerHTML = `
                <div class="terminal">
                    玩家 ${r.winner} 胡牌
                </div>
                <div style="margin-top:8px">
                    ${r.fan_count} 番
                </div>
            `;
        }

        return;
    }

    let last = "";

    if (state.last_discard) {
        last = `
            <div class="last">
                玩家 ${state.last_discard.player}
                打 ${state.last_discard.tile}
            </div>
        `;
    }

    box.innerHTML = `
        <div style="font-size:28px;font-weight:700">
            ${state.wall_remaining}
        </div>
        <div class="small">牌墙剩余</div>

        ${last}

        <div style="margin-top:14px" class="small">
            当前：玩家 ${state.current_player}
            · ${state.phase}
        </div>
    `;
}


function render() {

    document.getElementById("topStatus").textContent =
        `牌墙 ${state.wall_remaining} 张`;

    renderPlayer(state.players[1]);
    renderPlayer(state.players[2]);
    renderPlayer(state.players[3]);

    renderHand();
    renderActions();
    renderCenter();
    renderCoach();
}


async function loadState() {
    state = await api("/api/state");
    render();
}


async function newGame() {
    state = await api("/api/new", {
        method: "POST",
        body: "{}"
    });

    compareSelection = [];

    document.getElementById(
        "coachDetails"
    ).innerHTML = "";

    render();
}


loadState();

</script>

</body>
</html>
'''


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/state")
def api_state():
    return jsonify(state_json())


@app.route("/api/new", methods=["POST"])
def api_new():
    try:
        new_game()
        return jsonify(state_json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/action", methods=["POST"])
def api_action():
    try:
        if env.is_terminal():
            raise ValueError("本局已经结束")

        if env.current_player != HUMAN:
            raise ValueError("当前还没有轮到你")

        data = request.get_json(force=True)
        index = int(data["index"])

        legal = env.legal_actions(HUMAN)

        if index < 0 or index >= len(legal):
            raise ValueError("非法动作编号")

        action = legal[index]

        env.step(action)

        advance_ai()

        return jsonify(state_json())

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400



@app.route("/api/analysis")
def api_analysis():
    try:
        rows = current_ai_rows()

        if not rows:
            raise ValueError(
                "当前没有需要分析的决策"
            )

        return jsonify({
            "rows": rows,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 400


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        data = request.get_json(force=True)

        idx_a = int(data["a"])
        idx_b = int(data["b"])

        samples = int(
            data.get("samples", 64)
        )

        if samples < 2 or samples > 200:
            raise ValueError(
                "模拟世界数必须在 2～200 之间"
            )

        result = compare_actions_json(
            idx_a,
            idx_b,
            samples,
        )

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    new_game()

    print()
    print("GUI 已启动：")
    print("http://127.0.0.1:8000")
    print()

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
        use_reloader=False,
        threaded=False,
    )
