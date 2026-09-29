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
            "tile_name": (
                tile_name(action.tile)
                if action.tile is not None and action.tile >= 0
                else None
            ),
            "discard_name": (
                tile_name(action.discard)
                if action.discard is not None and action.discard >= 0
                else None
            ),
            "sequence_names": [
                tile_name(item)
                for item in action.sequence
            ],
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
    兼容旧调用：让 AI 一直推进到 HUMAN 或终局。
    Web GUI 使用 advance_ai_once()，这样前端可以逐步播放。
    """
    safety = 0

    while not env.is_terminal() and env.current_player != HUMAN:
        advance_ai_once()
        safety += 1
        if safety > 500:
            raise RuntimeError("AI 自动推进超过安全步数")


def auto_pass_human_if_forced():
    """
    训练模式下，如果 HUMAN 当前唯一合法动作就是 PASS，则自动执行。

    这只跳过“没有任何选择”的响应，不会自动替玩家决定
    吃 / 碰 / 杠 / 胡 / 出牌等真正存在分支的局面。
    """
    if env.is_terminal() or env.current_player != HUMAN:
        return False

    legal = env.legal_actions(HUMAN)

    if (
        len(legal) == 1
        and legal[0].kind == ActionType.PASS
    ):
        env.step(legal[0])
        return True

    return False


def advance_ai_once():
    """
    只推进一个 AI 决策。

    注意：不返回 AI 具体 claim 动作，避免在声明尚未统一裁决前
    把隐藏的吃/碰/杠意图泄漏给 HUMAN。
    """
    if env.is_terminal() or env.current_player == HUMAN:
        return False

    player = env.current_player
    obs = env.observe(player)
    legal = env.legal_actions(player)

    if not legal:
        raise RuntimeError(
            f"玩家 {player} 没有合法动作，phase={env.phase}"
        )

    old_event_count = len(env.events)
    action = coach.act(obs, legal)
    env.step(action)

    # 若 AI 动作后轮到 HUMAN，但唯一选择只是“过”，直接自动过。
    # 这样训练时不会被大量无意义的 PASS 点击打断。
    auto_pass_human_if_forced()

    # 只告诉前端“是否产生了新的公开事件”，不暴露未裁决 claim。
    return len(env.events) > old_event_count


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
                "discard": (
                    action.discard
                    if action.discard is not None
                    else -1
                ),
                "discard_name": (
                    tile_name(action.discard)
                    if action.discard is not None
                    and action.discard >= 0
                    else None
                ),
                "sequence": list(action.sequence),
                "sequence_names": [
                    tile_name(item)
                    for item in action.sequence
                ],
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
        "compute_device": str(device),
        "compare_default": 64 if device.type == "cuda" else 8,

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
    :root {
        --bg-deep: #071913;
        --bg-panel: #0a261d;
        --bg-panel-2: #0e3125;
        --felt-1: #0f5a42;
        --felt-2: #0a4634;
        --felt-3: #073326;
        --wood-1: #6e3f26;
        --wood-2: #3f2117;
        --wood-3: #27120d;
        --gold: #e7bb67;
        --gold-soft: #ffd992;
        --mint: #4ee0ae;
        --mint-2: #18b985;
        --text: #f4f7f5;
        --muted: #9fb9ae;
        --line: rgba(255,255,255,.10);
        --danger: #ff8c79;
    }

    * {
        box-sizing: border-box;
    }

    html, body {
        min-height: 100%;
    }

    body {
        margin: 0;
        color: var(--text);
        font-family:
            Inter, -apple-system, BlinkMacSystemFont,
            "Segoe UI", "Microsoft YaHei", sans-serif;
        background:
            radial-gradient(circle at 38% 18%, rgba(43,124,92,.18), transparent 30%),
            linear-gradient(145deg, #061611 0%, #092219 52%, #061711 100%);
    }

    button, select {
        font: inherit;
    }

    button {
        -webkit-tap-highlight-color: transparent;
    }

    #app {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }

    .topbar {
        min-height: 64px;
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 10px 18px 10px 22px;
        background: rgba(4, 24, 18, .90);
        border-bottom: 1px solid rgba(255,255,255,.08);
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0,0,0,.24);
        position: sticky;
        top: 0;
        z-index: 50;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: max-content;
    }

    .brand-tile {
        width: 34px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 6px;
        color: #b52b22;
        background: linear-gradient(155deg, #fffef8, #ded7c4);
        box-shadow: 0 3px 0 #968e7c, 0 6px 16px rgba(0,0,0,.22);
        font-size: 24px;
        font-weight: 800;
        transform: rotate(-4deg);
    }

    .title {
        font-size: 19px;
        font-weight: 800;
        letter-spacing: .02em;
        white-space: nowrap;
    }

    .title .ai {
        color: var(--mint);
    }

    .status {
        color: #b9cec4;
        font-size: 12px;
        white-space: nowrap;
    }

    .toolbar-spacer {
        flex: 1;
    }

    .toolbar-control {
        display: flex;
        align-items: center;
        gap: 7px;
        color: #bcd0c6;
        font-size: 12px;
        white-space: nowrap;
    }

    .toolbar-control select,
    .coach-setting select {
        color: #eff8f4;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 8px;
        padding: 7px 9px;
        outline: none;
    }

    .toolbar-control option,
    .coach-setting option {
        color: #111;
        background: white;
    }

    .device-badge {
        padding: 6px 9px;
        border-radius: 999px;
        border: 1px solid rgba(78,224,174,.20);
        background: rgba(78,224,174,.08);
        color: #9fdec7;
        font-size: 11px;
        white-space: nowrap;
    }

    .new-game {
        border: 1px solid rgba(78,224,174,.55);
        border-radius: 9px;
        padding: 9px 17px;
        color: white;
        background: linear-gradient(180deg, #18a97d, #0c7658);
        cursor: pointer;
        font-weight: 750;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.18);
    }

    .new-game:hover {
        filter: brightness(1.08);
    }

    .workspace {
        flex: 1;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 390px;
        gap: 14px;
        padding: 14px;
        min-height: 0;
    }

    .game-shell {
        min-width: 0;
        min-height: calc(100vh - 92px);
        display: flex;
    }

    .table-rim {
        flex: 1;
        min-width: 0;
        position: relative;
        border-radius: 44px;
        padding: 22px;
        background:
            linear-gradient(145deg, rgba(255,255,255,.08), transparent 20%),
            linear-gradient(145deg, var(--wood-1), var(--wood-2) 50%, var(--wood-3));
        box-shadow:
            inset 0 0 0 2px rgba(255,219,160,.11),
            inset 0 0 0 8px rgba(41,19,12,.28),
            0 24px 60px rgba(0,0,0,.34);
        overflow: hidden;
    }

    .table-rim::before,
    .table-rim::after {
        content: "";
        position: absolute;
        inset: 9px;
        border-radius: 38px;
        pointer-events: none;
    }

    .table-rim::before {
        border: 1px solid rgba(255,220,165,.15);
    }

    .table-rim::after {
        inset: auto 12% 7px;
        height: 8px;
        background: rgba(18,8,6,.40);
        border-radius: 50%;
        filter: blur(7px);
    }

    .felt {
        position: relative;
        height: 100%;
        min-height: 690px;
        border-radius: 30px;
        overflow: hidden;
        background:
            radial-gradient(circle at 50% 45%, rgba(48,134,98,.28), transparent 38%),
            radial-gradient(circle at 50% 55%, rgba(4,43,31,.0) 0 52%, rgba(1,24,18,.32) 100%),
            linear-gradient(145deg, var(--felt-1), var(--felt-2) 58%, var(--felt-3));
        box-shadow:
            inset 0 0 80px rgba(0,0,0,.22),
            inset 0 0 0 1px rgba(255,255,255,.08);
    }

    .felt::before {
        content: "";
        position: absolute;
        inset: 20% 28%;
        border-radius: 50%;
        background:
            repeating-radial-gradient(
                circle,
                rgba(255,255,255,.022) 0 1px,
                transparent 1px 12px
            );
        opacity: .55;
        pointer-events: none;
    }

    .player {
        position: absolute;
        z-index: 3;
        color: var(--text);
        min-width: 150px;
    }

    .p2 {
        top: 18px;
        left: 50%;
        transform: translateX(-50%);
        width: min(58%, 650px);
    }

    .p3 {
        left: 18px;
        top: 50%;
        transform: translateY(-50%);
        width: min(230px, 22%);
    }

    .p1 {
        right: 18px;
        top: 50%;
        transform: translateY(-50%);
        width: min(230px, 22%);
    }

    .player-head {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
        background: rgba(4, 34, 24, .72);
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
        backdrop-filter: blur(7px);
        margin-bottom: 7px;
    }

    .p2 .player-head {
        margin-left: 50%;
        transform: translateX(-50%);
    }

    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        color: #3c2a1d;
        background: linear-gradient(145deg, #ffe0a8, #c99a56);
        border: 1px solid rgba(255,255,255,.38);
        font-size: 16px;
    }

    .player-name {
        font-size: 12px;
        font-weight: 800;
        line-height: 1.15;
    }

    .player-count {
        margin-top: 2px;
        color: #bdd0c7;
        font-size: 10px;
    }

    .concealed-row {
        display: flex;
        gap: 2px;
        margin: 0 auto 8px;
        justify-content: center;
    }

    .p3 .concealed-row,
    .p1 .concealed-row {
        display: none;
    }

    .tile-back {
        width: 27px;
        height: 39px;
        border-radius: 4px;
        background:
            linear-gradient(180deg, rgba(255,255,255,.08), transparent 25%),
            linear-gradient(145deg, #2f784f, #135536 62%, #0d3926);
        border: 1px solid rgba(255,255,255,.17);
        box-shadow: 0 2px 0 #ddd4bd, 0 4px 7px rgba(0,0,0,.18);
    }

    .player-public {
        padding: 7px 8px;
        border-radius: 10px;
        background: rgba(3, 30, 21, .24);
        border: 1px solid rgba(255,255,255,.05);
    }

    .player-public-label {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        color: rgba(220,235,228,.67);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 5px;
    }

    .river,
    .meld-row {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
    }

    .p2 .river,
    .p2 .meld-row {
        justify-content: center;
    }

    .mini-tile {
        width: 32px;
        min-width: 32px;
        height: 44px;
        padding: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 4px;
        border: 1px solid rgba(64,53,35,.20);
        background: linear-gradient(155deg, #fffef9, #eee8d8);
        box-shadow: 0 2px 0 #aca38e, 0 4px 7px rgba(0,0,0,.19);
        color: #15211b;
        font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", "Apple Symbols", sans-serif;
        font-size: 25px;
        line-height: 1;
    }

    .mini-tile img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }

    .meld {
        display: inline-block;
        margin-top: 5px;
        margin-right: 5px;
        padding: 3px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(0,0,0,.12);
    }

    .center {
        position: absolute;
        inset: 50% auto auto 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
    }

    .center-box {
        min-width: 230px;
        text-align: center;
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(231,187,103,.35);
        background:
            linear-gradient(180deg, rgba(7,47,35,.94), rgba(4,32,24,.92));
        box-shadow: 0 18px 36px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.06);
    }

    .round-title {
        color: var(--gold-soft);
        font-size: 20px;
        font-weight: 850;
        letter-spacing: .04em;
    }

    .wall-number {
        margin-top: 8px;
        font-size: 30px;
        font-weight: 850;
        font-variant-numeric: tabular-nums;
    }

    .center-label {
        color: #b6cbc1;
        font-size: 10px;
        margin-top: -2px;
    }

    .center-current {
        margin-top: 9px;
        font-size: 11px;
        color: #e8f0ec;
    }

    .last-play {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        margin-top: 9px;
        padding-top: 8px;
        border-top: 1px solid rgba(255,255,255,.08);
        color: #c7d7cf;
        font-size: 10px;
    }

    .last-play .mini-tile {
        width: 28px;
        min-width: 28px;
        height: 39px;
    }

    .human {
        position: absolute;
        z-index: 5;
        left: 6%;
        right: 6%;
        bottom: 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .human-public {
        width: min(780px, 82%);
        min-height: 48px;
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 12px;
        margin-bottom: 8px;
    }

    .human-river-wrap,
    .human-meld-wrap {
        padding: 6px 8px;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(3,31,22,.28);
        backdrop-filter: blur(5px);
    }

    .human-river-wrap {
        min-width: 260px;
        flex: 1;
    }

    .human-meld-wrap {
        flex: 0 0 auto;
    }

    .small {
        color: #aac0b6;
        font-size: 10px;
    }

    .actions {
        min-height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        flex-wrap: wrap;
        margin-bottom: 5px;
    }

    .action-btn,
    .claim-cancel {
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,.15);
        color: white;
        background: rgba(8,53,39,.86);
        cursor: pointer;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.07);
    }

    .action-btn:hover,
    .claim-cancel:hover {
        background: rgba(17,91,68,.95);
    }

    .claim-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 11px;
        border-radius: 9px;
        color: #ffeab7;
        border: 1px solid rgba(231,187,103,.28);
        background: rgba(72,48,16,.30);
        font-size: 11px;
    }

    .claim-step strong {
        color: #fff6d6;
    }

    .human-label {
        margin: 1px 0 6px;
        color: #dbe8e2;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .06em;
    }

    .hand {
        display: flex;
        align-items: flex-end;
        justify-content: center;
        gap: 4px;
        min-height: 96px;
        padding: 5px 9px 7px;
        border-radius: 14px;
        background: rgba(2, 27, 19, .20);
        border: 1px solid rgba(255,255,255,.045);
    }

    .tile {
        position: relative;
        width: 58px;
        height: 82px;
        border-radius: 7px;
        border: 1px solid rgba(65,51,34,.18);
        padding: 2px;
        overflow: visible;
        color: #15211b;
        background: linear-gradient(155deg, #fffef9 0%, #f4efe2 60%, #d9d1bd 100%);
        box-shadow: 0 4px 0 #a79d86, 0 7px 12px rgba(0,0,0,.30);
        cursor: default;
        transition: transform .12s ease, filter .12s ease, box-shadow .12s ease;
    }

    .tile img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        pointer-events: none;
    }

    .tile.playable {
        cursor: pointer;
    }

    .tile.playable:hover {
        transform: translateY(-9px);
        filter: brightness(1.04);
        box-shadow: 0 4px 0 #a79d86, 0 14px 18px rgba(0,0,0,.32);
    }

    .tile.claim-discard {
        outline: 2px solid var(--gold);
        outline-offset: 2px;
    }

    .tile.recommended-primary {
        outline: 2px solid #57f0ae;
        outline-offset: 3px;
        box-shadow:
            0 4px 0 #a79d86,
            0 0 0 5px rgba(87,240,174,.09),
            0 0 23px rgba(87,240,174,.33);
    }

    .tile.recommended-secondary {
        outline: 2px solid #73aefb;
        outline-offset: 2px;
    }

    .recommend-badge {
        position: absolute;
        left: 50%;
        bottom: -27px;
        transform: translateX(-50%);
        min-width: 40px;
        padding: 3px 5px;
        border-radius: 7px;
        font-size: 9px;
        font-weight: 850;
        color: #f3fff9;
        background: rgba(4,37,27,.95);
        border: 1px solid rgba(87,240,174,.30);
        box-shadow: 0 4px 10px rgba(0,0,0,.18);
        pointer-events: none;
    }

    .recommend-badge.secondary {
        border-color: rgba(115,174,251,.38);
        color: #d9ebff;
    }

    .human-seat {
        margin-top: 31px;
        padding: 5px 10px;
        border-radius: 999px;
        color: #b9d0c5;
        background: rgba(3,28,20,.33);
        font-size: 10px;
    }

    .coach-panel {
        min-height: calc(100vh - 92px);
        max-height: calc(100vh - 92px);
        overflow-y: auto;
        padding: 16px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.10);
        background:
            radial-gradient(circle at 70% 0%, rgba(48,143,106,.12), transparent 30%),
            linear-gradient(180deg, rgba(7,38,29,.98), rgba(4,27,21,.98));
        box-shadow: 0 20px 50px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
        scrollbar-width: thin;
        scrollbar-color: rgba(255,255,255,.16) transparent;
    }

    .coach-header {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        margin-bottom: 13px;
    }

    .coach-spark {
        color: var(--gold-soft);
        font-size: 20px;
        line-height: 1;
        margin-top: 2px;
    }

    .coach-title {
        font-size: 18px;
        font-weight: 850;
    }

    .coach-subtitle {
        color: #a7beb3;
        font-size: 10px;
        line-height: 1.55;
        margin-top: 3px;
    }

    .featured-list {
        display: grid;
        gap: 10px;
    }

    .featured-card {
        position: relative;
        display: grid;
        grid-template-columns: 26px 50px minmax(0,1fr) auto;
        align-items: center;
        gap: 9px;
        padding: 12px;
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.035);
    }

    .featured-card.primary {
        border-color: rgba(87,240,174,.70);
        background:
            radial-gradient(circle at 100% 0%, rgba(87,240,174,.12), transparent 42%),
            rgba(17,90,67,.24);
        box-shadow: inset 0 0 0 1px rgba(87,240,174,.08), 0 10px 22px rgba(0,0,0,.16);
    }

    .featured-card.secondary {
        border-color: rgba(115,174,251,.25);
    }

    .featured-card input,
    .other-row input {
        width: 16px;
        height: 16px;
        accent-color: #44dba5;
        cursor: pointer;
    }

    .coach-tile-icon {
        width: 45px;
        height: 62px;
        padding: 2px;
        border-radius: 6px;
        background: linear-gradient(155deg, #fffef9, #eee8d8);
        box-shadow: 0 3px 0 #a99f89, 0 5px 10px rgba(0,0,0,.20);
        display: grid;
        place-items: center;
        overflow: hidden;
        font-size: 31px;
        color: #16221c;
    }

    .coach-tile-icon img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .featured-rank {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-bottom: 5px;
        color: #ffe3a1;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .08em;
    }

    .featured-action {
        font-size: 19px;
        font-weight: 850;
        line-height: 1.2;
    }

    .featured-meta {
        margin-top: 5px;
        color: #afc4ba;
        font-size: 9px;
        line-height: 1.4;
    }

    .featured-prob {
        text-align: right;
        font-size: 26px;
        font-weight: 900;
        font-variant-numeric: tabular-nums;
        color: #fff4ce;
        white-space: nowrap;
    }

    .featured-prob span {
        font-size: 11px;
        font-weight: 650;
        color: #a9c0b5;
        margin-left: 2px;
    }

    .coach-controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 11px;
    }

    .coach-btn {
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 9px;
        padding: 9px 8px;
        color: white;
        background: rgba(255,255,255,.07);
        cursor: pointer;
        font-size: 11px;
        font-weight: 700;
    }

    .coach-btn.primary {
        color: #3d2b13;
        border-color: #f4d28c;
        background: linear-gradient(180deg, #ffe1a5, #e8bd72);
    }

    .coach-btn:hover {
        filter: brightness(1.08);
    }

    .coach-btn:disabled {
        opacity: .45;
        cursor: wait;
    }

    .coach-setting {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-top: 10px;
        padding: 9px 0;
        border-top: 1px solid rgba(255,255,255,.07);
        border-bottom: 1px solid rgba(255,255,255,.07);
        color: #a9beb4;
        font-size: 10px;
    }

    .other-candidates {
        margin-top: 10px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(255,255,255,.025);
        overflow: hidden;
    }

    .other-candidates summary {
        cursor: pointer;
        list-style: none;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 11px;
        color: #c1d3ca;
        font-size: 10px;
        font-weight: 750;
    }

    .other-candidates summary::-webkit-details-marker {
        display: none;
    }

    .other-candidates summary::after {
        content: "⌄";
        color: #9eb7ab;
        font-size: 14px;
    }

    .other-candidates[open] summary::after {
        transform: rotate(180deg);
    }

    .other-list {
        border-top: 1px solid rgba(255,255,255,.06);
    }

    .other-row {
        display: grid;
        grid-template-columns: 24px minmax(0,1fr) 58px;
        gap: 7px;
        align-items: center;
        padding: 7px 10px;
        border-bottom: 1px solid rgba(255,255,255,.045);
        font-size: 10px;
    }

    .other-row:last-child {
        border-bottom: 0;
    }

    .other-action {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: #dce8e2;
    }

    .other-prob {
        text-align: right;
        font-variant-numeric: tabular-nums;
        color: #d8e8e0;
        font-weight: 750;
    }

    .coach-details {
        margin-top: 11px;
        font-size: 10px;
        line-height: 1.55;
        color: #d6e4dd;
    }

    .analysis-table,
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        font-size: 9px;
    }

    .analysis-table th,
    .analysis-table td,
    .compare-table th,
    .compare-table td {
        padding: 5px 3px;
        border-bottom: 1px solid rgba(255,255,255,.06);
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
        margin-top: 11px;
        color: #eef7f2;
        font-weight: 850;
        font-size: 10px;
    }

    .coach-warning {
        margin-top: 8px;
        color: #8ea89c;
        font-size: 9px;
    }

    .coach-loading {
        padding: 12px 0;
        color: #c5d9cf;
    }

    .coach-action-glyph {
        font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", "Apple Symbols", sans-serif;
        font-size: 17px;
        white-space: nowrap;
    }

    .terminal {
        font-size: 22px;
        font-weight: 850;
        color: #fff0c9;
    }

    @media (max-width: 1350px) {
        .workspace {
            grid-template-columns: minmax(0,1fr) 330px;
        }

        .featured-action {
            font-size: 16px;
        }

        .featured-prob {
            font-size: 22px;
        }

        .tile {
            width: 52px;
            height: 74px;
        }

        .mini-tile {
            width: 29px;
            min-width: 29px;
            height: 40px;
        }
    }

    @media (max-width: 1050px) {
        .workspace {
            display: block;
        }

        .game-shell {
            min-height: 680px;
        }

        .coach-panel {
            min-height: auto;
            max-height: none;
            margin-top: 14px;
        }

        .topbar {
            flex-wrap: wrap;
        }

        .toolbar-spacer {
            display: none;
        }
    }
</style></style>
</head>

<body>
<div id="app">
    <header class="topbar">
        <div class="brand">
            <div class="brand-tile">中</div>
            <div class="title">麻将 <span class="ai">AI</span> 教练</div>
        </div>

        <div class="status" id="topStatus"></div>

        <div class="toolbar-spacer"></div>

        <div class="toolbar-control">
            AI 节奏
            <select id="speedSelect">
                <option value="250">快</option>
                <option value="650" selected>正常</option>
                <option value="1100">慢</option>
            </select>
        </div>

        <div class="device-badge" id="deviceBadge">AI device</div>
        <button class="new-game" onclick="newGame()">↻&nbsp; 新一局</button>
    </header>

    <main class="workspace">
        <section class="game-shell">
            <div class="table-rim">
                <div class="felt">
                    <div class="player p2" id="player2"></div>
                    <div class="player p3" id="player3"></div>
                    <div class="player p1" id="player1"></div>

                    <div class="center">
                        <div class="center-box">
                            <div id="centerInfo"></div>
                        </div>
                    </div>

                    <div class="human">
                        <div class="human-public">
                            <div class="human-river-wrap">
                                <div class="small">你的牌河</div>
                                <div class="river" id="humanRiver"></div>
                            </div>

                            <div class="human-meld-wrap" id="humanMeldWrap" style="display:none">
                                <div class="small">你的副露</div>
                                <div class="meld-row" id="humanMelds"></div>
                            </div>
                        </div>

                        <div class="actions" id="actions"></div>
                        <div class="human-label">你的手牌</div>
                        <div class="hand" id="hand"></div>
                        <div class="human-seat">你 · 玩家 0</div>
                    </div>
                </div>
            </div>
        </section>

        <aside class="coach-panel">
            <div class="coach-header">
                <div class="coach-spark">✦</div>
                <div>
                    <div class="coach-title">AI 教练</div>
                    <div class="coach-subtitle">
                        基于当前公开局面给出策略偏好。百分比是模型策略偏好，
                        不是校准后的真实胜率。
                    </div>
                </div>
            </div>

            <div class="featured-list" id="coachFeatured"></div>

            <div class="coach-controls">
                <button
                    class="coach-btn primary"
                    id="compareBtn"
                    onclick="runCompare()"
                >
                    比较两个选择
                </button>

                <button
                    class="coach-btn"
                    id="analysisBtn"
                    onclick="loadAnalysis()"
                >
                    查看详细分析
                </button>
            </div>

            <div class="coach-setting">
                <span>反事实模拟</span>
                <select id="compareSamples">
                    <option value="0">关闭</option>
                    <option value="8">快速 · 8 worlds</option>
                    <option value="16">标准 · 16 worlds</option>
                    <option value="64">深入 · 64 worlds</option>
                </select>
            </div>

            <details class="other-candidates" id="otherCandidates">
                <summary>其他候选</summary>
                <div class="other-list" id="coachOthers"></div>
            </details>

            <div class="coach-details" id="coachDetails"></div>
        </aside>
    </main>
</div>

<script><script>

let state = null;
let compareSelection = [];
let advancing = false;
let compareConfigured = false;
let gameGeneration = 0;
let pendingClaim = null;


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


function tileGlyph(name) {
    if (!name) return "";

    const rank = parseInt(name, 10);

    if (name.endsWith("万") && rank >= 1 && rank <= 9) {
        return String.fromCodePoint(0x1F007 + rank - 1);
    }

    if (name.endsWith("条") && rank >= 1 && rank <= 9) {
        return String.fromCodePoint(0x1F010 + rank - 1);
    }

    if (name.endsWith("筒") && rank >= 1 && rank <= 9) {
        return String.fromCodePoint(0x1F019 + rank - 1);
    }

    const honors = {
        "东": "🀀",
        "南": "🀁",
        "西": "🀂",
        "北": "🀃",
        "中": "🀄",
        "发": "🀅",
        "白": "🀆"
    };

    return honors[name] || name;
}


function tileAssetPath(name) {
    if (!name) return null;

    const rank = parseInt(name, 10);

    if (name.endsWith("万") && rank >= 1 && rank <= 9) {
        return `/static/tiles/m${rank}.svg`;
    }

    if (name.endsWith("条") && rank >= 1 && rank <= 9) {
        return `/static/tiles/s${rank}.svg`;
    }

    if (name.endsWith("筒") && rank >= 1 && rank <= 9) {
        return `/static/tiles/p${rank}.svg`;
    }

    const honors = {
        "东": "east.svg",
        "南": "south.svg",
        "西": "west.svg",
        "北": "north.svg",
        "中": "red.svg",
        "发": "green.svg",
        "白": "white.svg"
    };

    return honors[name]
        ? `/static/tiles/${honors[name]}`
        : null;
}


function actionGlyphs(row) {
    if (!row) return "";

    if (row.kind === "CHI" && row.sequence_names?.length) {
        const claim = row.sequence_names.map(tileGlyph).join("");
        return row.discard_name
            ? `${claim} → ${tileGlyph(row.discard_name)}`
            : claim;
    }

    if (row.kind === "PENG") {
        const claim = tileGlyph(row.tile_name);
        return row.discard_name
            ? `${claim} → ${tileGlyph(row.discard_name)}`
            : claim;
    }

    if (row.kind === "PLAY"
        || row.kind === "GANG"
        || row.kind === "BUGANG") {
        return tileGlyph(row.tile_name);
    }

    if (row.kind === "HU") {
        return "胡";
    }

    if (row.kind === "PASS") {
        return "过";
    }

    return "";
}


function tileImageElement(name) {
    const img = document.createElement("img");
    const src = tileAssetPath(name);

    if (src) {
        img.src = src;
        img.alt = name;
        img.title = name;
    }

    return img;
}


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
    d.title = name;

    const img = tileImageElement(name);

    if (img.src) {
        img.onerror = () => {
            d.replaceChildren(document.createTextNode(tileGlyph(name)));
        };
        d.appendChild(img);
    } else {
        d.textContent = tileGlyph(name);
    }

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


async function playAction(index) {
    if (advancing) return;

    try {
        advancing = true;

        state = await api("/api/action", {
            method: "POST",
            body: JSON.stringify({index})
        });

        compareSelection = [];
        pendingClaim = null;
        document.getElementById("coachDetails").innerHTML = "";
        render();

        await advanceUntilHuman();

    } catch (err) {
        alert(err.message);
    } finally {
        advancing = false;
        render();
    }
}


async function advanceUntilHuman() {
    const generation = gameGeneration;

    while (
        state
        && !state.terminal
        && state.current_player !== 0
        && generation === gameGeneration
    ) {
        const delay = Number(
            document.getElementById("speedSelect").value
        ) || 650;

        await sleep(delay);

        if (generation !== gameGeneration) {
            return;
        }

        const result = await api("/api/advance", {
            method: "POST",
            body: "{}"
        });

        state = result.state;
        pendingClaim = null;
        render();
    }
}


function renderHand() {

    const hand = document.getElementById("hand");
    hand.innerHTML = "";

    const actionMap = new Map();

    if (pendingClaim) {
        // 第二步：碰/吃已经由玩家选定，现在只允许选择该声明后
        // 真正合法的弃牌。每张弃牌对应引擎里的一个组合 Action。
        for (const index of pendingClaim.indices) {
            const action = state.legal_actions.find(
                item => item.index === index
            );

            if (action && action.discard >= 0) {
                actionMap.set(action.discard, action.index);
            }
        }
    } else {
        for (const action of state.legal_actions) {
            if (action.kind === "PLAY") {
                actionMap.set(action.tile, action.index);
            }
        }
    }

    for (const item of state.hand) {

        const btn = document.createElement("button");
        btn.className = "tile";
        btn.title = item.name;

        const img = tileImageElement(item.name);

        if (img.src) {
            img.onerror = () => {
                btn.replaceChildren(
                    document.createTextNode(tileGlyph(item.name))
                );
            };
            btn.appendChild(img);
        } else {
            btn.textContent = tileGlyph(item.name);
        }

        if (actionMap.has(item.tile)) {
            btn.classList.add("playable");

            if (pendingClaim) {
                btn.classList.add("claim-discard");
            }

            const index = actionMap.get(item.tile);

            btn.onclick = () => playAction(index);
        }

        hand.appendChild(btn);
    }
}


function renderHumanPublic() {
    const player = state.players[0];

    const river = document.getElementById("humanRiver");
    river.innerHTML = "";

    for (const tile of player.river) {
        river.appendChild(miniTile(tile));
    }

    if (player.river.length === 0) {
        river.innerHTML = '<span class="small">（还没有弃牌）</span>';
    }

    const meldWrap = document.getElementById("humanMeldWrap");
    const meldBox = document.getElementById("humanMelds");
    meldBox.innerHTML = "";

    const meldTiles = [];

    for (const meld of player.melds) {
        for (const tile of meld.tiles) {
            meldTiles.push(tile);
        }
    }

    if (meldTiles.length > 0) {
        meldWrap.style.display = "";

        for (const tile of meldTiles) {
            meldBox.appendChild(miniTile(tile));
        }
    } else {
        meldWrap.style.display = "none";
    }
}


function claimGroupKey(action) {
    if (action.kind === "PENG") {
        return `PENG:${action.tile}`;
    }

    if (action.kind === "CHI") {
        return `CHI:${(action.sequence || []).join(",")}`;
    }

    return null;
}


function claimGroupLabel(action) {
    if (action.kind === "PENG") {
        return `${tileGlyph(action.tile_name)} 碰 ${action.tile_name}`;
    }

    if (action.kind === "CHI") {
        const glyphs = (action.sequence_names || [])
            .map(tileGlyph)
            .join("");
        const names = (action.sequence_names || []).join(" ");
        return `${glyphs} 吃 ${names}`;
    }

    return action.name;
}


function startClaimStep(actions) {
    if (!actions.length) return;

    const first = actions[0];

    pendingClaim = {
        kind: first.kind,
        indices: actions.map(item => item.index),
        label: claimGroupLabel(first),
    };

    renderActions();
    renderHand();
}


function cancelClaimStep() {
    pendingClaim = null;
    renderActions();
    renderHand();
}


function renderActions() {

    const box = document.getElementById("actions");
    box.innerHTML = "";

    if (pendingClaim) {
        const step = document.createElement("div");
        step.className = "claim-step";
        step.innerHTML = `
            <span>
                已选择 <strong>${pendingClaim.label}</strong>，
                现在请选择一张手牌打出
            </span>
        `;

        const cancel = document.createElement("button");
        cancel.className = "claim-cancel";
        cancel.textContent = "取消";
        cancel.onclick = cancelClaimStep;

        step.appendChild(cancel);
        box.appendChild(step);
        return;
    }

    const groups = new Map();

    for (const action of state.legal_actions) {

        // 普通 PLAY 已经可以直接点手牌，不重复显示按钮。
        if (action.kind === "PLAY") {
            continue;
        }

        const key = claimGroupKey(action);

        if (key) {
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push(action);
            continue;
        }

        // PASS / HU / GANG / BUGANG 等没有“声明后再弃牌”的第二步，
        // 保持一步操作。
        const btn = document.createElement("button");
        btn.className = "action-btn";
        btn.textContent = action.name;
        btn.onclick = () => playAction(action.index);
        box.appendChild(btn);
    }

    for (const actions of groups.values()) {
        const btn = document.createElement("button");
        btn.className = "action-btn";
        btn.textContent = claimGroupLabel(actions[0]);
        btn.onclick = () => startClaimStep(actions);
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
                    <span>${rank + 1}.</span>
                    <span class="coach-action-glyph">
                        ${actionGlyphs(row)}
                    </span>
                    <span>
                        ${row.name}
                        ${rank === 0 ? " ← 推荐" : ""}
                    </span>
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
                        <th>策略总分</th>
                        <th>具体动作分</th>
                        <th>类型基准</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.rows.slice(0, 8).forEach(row => {
            html += `
                <tr>
                    <td><span class="coach-action-glyph">${actionGlyphs(row)}</span> ${row.name}</td>
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

    const samples = Number(
        document.getElementById("compareSamples").value
    );

    if (!samples) {
        alert("反事实模拟当前已关闭。请选择快速、标准或深入模式。");
        button.disabled = false;
        return;
    }

    details.innerHTML = `
        <div class="coach-loading">
            正在跑 ${samples} 个配对可能世界……<br>
            将推进 ${samples * 2} 条后续路线 · ${state.compute_device}
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
                    samples
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
        advancing
        ? `AI 思考中 · 牌墙 ${state.wall_remaining} 张`
        : `牌墙 ${state.wall_remaining} 张`;

    document.getElementById("deviceBadge").textContent =
        `AI · ${state.compute_device}`;

    if (!compareConfigured) {
        document.getElementById("compareSamples").value =
            String(state.compare_default || 8);
        compareConfigured = true;
    }

    renderPlayer(state.players[1]);
    renderPlayer(state.players[2]);
    renderPlayer(state.players[3]);

    renderHumanPublic();
    renderHand();
    renderActions();
    renderCenter();
    renderCoach();
}


async function loadState() {
    state = await api("/api/state");
    pendingClaim = null;
    render();
}


async function newGame() {
    gameGeneration += 1;
    advancing = false;

    state = await api("/api/new", {
        method: "POST",
        body: "{}"
    });

    compareSelection = [];
    pendingClaim = null;

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

        # 极少数状态迁移可能立即回到 HUMAN；若此时唯一选择为“过”，
        # 同样自动跳过。
        auto_pass_human_if_forced()

        # Web 端逐个请求 AI 决策，以便把每一步实际播放出来。
        return jsonify(state_json())

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400



@app.route("/api/advance", methods=["POST"])
def api_advance():
    try:
        if env.is_terminal() or env.current_player == HUMAN:
            return jsonify({
                "state": state_json(),
                "public_changed": False,
            })

        public_changed = advance_ai_once()

        return jsonify({
            "state": state_json(),
            "public_changed": public_changed,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 400


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
