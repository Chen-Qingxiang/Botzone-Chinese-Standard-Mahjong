import copy
import math
import random

import torch

from mahjong_agent.engine import MahjongEnv
from mahjong_agent.engine.actions import ActionType
from mahjong_agent.engine.tiles import tile_to_name
from mahjong_agent.training.checkpoint import load_model_from_checkpoint
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.features import encode_action_v2, encode_observation_v2
from mahjong_agent.policies.analysis import hand_potential, action_deal_in_risk


HUMAN = 0

CN = {
    "F1": "东", "F2": "南", "F3": "西", "F4": "北",
    "J1": "中", "J2": "发", "J3": "白",
}


def tile_name(tile):
    if tile < 0:
        return "-"
    x = tile_to_name(tile)
    if x in CN:
        return CN[x]
    suit = {"W": "万", "T": "条", "B": "筒"}[x[0]]
    return x[1:] + suit


def tiles_name(tiles):
    return " ".join(tile_name(x) for x in tiles)


def action_name(a):
    if a.kind == ActionType.PASS:
        return "过"
    if a.kind == ActionType.PLAY:
        return "打 " + tile_name(a.tile)
    if a.kind == ActionType.HU:
        return "胡"
    if a.kind == ActionType.GANG:
        return "杠 " + tile_name(a.tile)
    if a.kind == ActionType.BUGANG:
        return "补杠 " + tile_name(a.tile)
    if a.kind == ActionType.PENG:
        return f"碰 {tile_name(a.tile)} → 打 {tile_name(a.discard)}"
    if a.kind == ActionType.CHI:
        return f"吃 [{tiles_name(a.sequence)}] → 打 {tile_name(a.discard)}"
    return str(a)


def meld_name(m):
    if m.kind == ActionType.CHI:
        kind = "吃"
    elif m.kind == ActionType.PENG:
        kind = "碰"
    elif m.kind == ActionType.GANG:
        kind = "杠"
    else:
        kind = m.kind.name
    return f"{kind}[{tiles_name(m.tiles)}]"



def coach_rank(coach, observation, legal):
    """返回 [(合法动作编号, action, 模型偏好概率), ...]。"""

    # ModelPolicy 遇到可胡时会强制胡，这里保持同样逻辑。
    for i, action in enumerate(legal):
        if action.kind == ActionType.HU:
            return [(i, action, 1.0)]

    model = coach.model
    model.eval()
    device = next(model.parameters()).device

    feature_tokens, feature_mask = encode_observation_v2(observation)

    encoded_actions = [encode_action_v2(a) for a in legal]
    action_tokens = [x[0] for x in encoded_actions]
    action_token_masks = [x[1] for x in encoded_actions]

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
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

    ranked = list(enumerate(zip(legal, probabilities)))
    ranked.sort(key=lambda x: x[1][1], reverse=True)

    return [
        (index, action, probability)
        for index, (action, probability) in ranked
    ]



def coach_analyze(coach, observation, legal):
    """
    拆解模型对每个候选动作的内部打分。

    final_logit =
        family_logit
        + tactical_logit
        + auxiliary_correction
    """

    model = coach.model
    model.eval()
    device = next(model.parameters()).device

    feature_tokens, feature_mask = encode_observation_v2(observation)
    encoded_actions = [encode_action_v2(a) for a in legal]

    action_tokens = [x[0] for x in encoded_actions]
    action_token_masks = [x[1] for x in encoded_actions]

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
        probs = torch.softmax(logits, dim=-1)

        family = output["family_logits"][0, :len(legal)]
        tactical = output["tactical_logits"][0, :len(legal)]

        action_outcome = output["action_outcome"][0, :len(legal)]
        action_fan_logits = output["action_fan_logits"][0, :len(legal)]

        # 与模型 forward() 中完全相同的辅助修正。
        fan_values = torch.arange(
            5,
            device=device,
            dtype=action_fan_logits.dtype,
        )

        expected_fan_bucket = (
            action_fan_logits.softmax(-1) * fan_values
        ).sum(-1)

        auxiliary = torch.stack(
            (
                action_outcome[:, 0],       # win logit
                -action_outcome[:, 1],      # negative deal-in logit
                action_outcome[:, 2],       # score regression
                action_outcome[:, 3],       # 8-fan logit
                expected_fan_bucket,
            ),
            dim=-1,
        )

        gate = model.auxiliary_logit_gate
        aux_correction = (auxiliary * gate).sum(-1)

        # 这三个是 BCE 训练出的 logits，可 sigmoid 成 [0,1]，
        # 但这里只能理解为模型预测倾向，不能当成校准后的真实概率。
        win_tendency = torch.sigmoid(action_outcome[:, 0])
        dealin_tendency = torch.sigmoid(action_outcome[:, 1])
        eightfan_tendency = torch.sigmoid(action_outcome[:, 3])

    rows = []

    for i, action in enumerate(legal):
        rows.append({
            "idx": i,
            "action": action,
            "prob": float(probs[i]),
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

    rows.sort(key=lambda x: x["prob"], reverse=True)

    print("\n===== 模型内部决策拆解 =====")
    print("偏好 = softmax(final logit)，不是客观胜率。")
    print("Win/放铳/8番也是模型辅助头的预测倾向，不是校准后的真实概率。\n")

    print(
        f"{'排名':<4}"
        f"{'动作':<27}"
        f"{'偏好':>8}"
        f"{'总分':>9}"
        f"{'战术':>9}"
        f"{'类别':>9}"
        f"{'Aux':>9}"
    )
    print("-" * 84)

    for rank, row in enumerate(rows[:8], 1):
        action_text = f"[{row['idx']}] {action_name(row['action'])}"
        if len(action_text) > 25:
            action_text = action_text[:24] + "…"

        print(
            f"{rank:<4}"
            f"{action_text:<27}"
            f"{row['prob'] * 100:>7.2f}%"
            f"{row['logit']:>9.3f}"
            f"{row['tactical']:>9.3f}"
            f"{row['family']:>9.3f}"
            f"{row['aux']:>9.3f}"
        )

    best = rows[0]

    print("\n--- 第一名的辅助预测 ---")
    print(f"动作：{action_name(best['action'])}")
    print(f"Win 倾向：      {best['win'] * 100:6.2f}%")
    print(f"放铳倾向：      {best['dealin'] * 100:6.2f}%")
    print(f"8番倾向：       {best['eightfan'] * 100:6.2f}%")
    print(f"得分预测值：    {best['score']:8.3f}")
    print(f"番数档位期望：  {best['fan_bucket']:8.3f}")

    if len(rows) >= 2:
        second = rows[1]

        print("\n--- 第一名 vs 第二名 ---")
        print(
            f"{action_name(best['action'])}  vs  "
            f"{action_name(second['action'])}"
        )

        print(
            f"总分差：    "
            f"{best['logit'] - second['logit']:+.3f}"
        )
        print(
            f"战术分差：  "
            f"{best['tactical'] - second['tactical']:+.3f}"
        )
        print(
            f"类别分差：  "
            f"{best['family'] - second['family']:+.3f}"
        )
        print(
            f"Aux 修正差："
            f"{best['aux'] - second['aux']:+.3f}"
        )

        tactical_gap = abs(best["tactical"] - second["tactical"])
        family_gap = abs(best["family"] - second["family"])
        aux_gap = abs(best["aux"] - second["aux"])

        largest = max(
            [
                ("战术网络", tactical_gap),
                ("动作类别", family_gap),
                ("辅助预测", aux_gap),
            ],
            key=lambda x: x[1],
        )

        print(
            f"\n当前可确认：两者最大的直接打分差来自"
            f"「{largest[0]}」。"
        )

        if largest[0] == "战术网络":
            print(
                "但 tactical score 是神经网络学出的综合表征，"
                "这一层本身不能继续翻译成唯一的人类规则。"
            )

    print()

def print_board(env):
    print("\n" + "=" * 68)
    print(f"牌墙剩余：{len(env.wall)}")

    if env.last_discard is not None:
        p, t = env.last_discard
        print(f"上一张：玩家 {p} 打 {tile_name(t)}")

    print("\n--- 牌河 ---")
    for p in range(4):
        river = tiles_name(env.discards[p]) or "（空）"
        print(f"玩家 {p}: {river}")

    print("\n--- 副露 ---")
    for p in range(4):
        ms = "  ".join(meld_name(m) for m in env.melds[p]) or "（无）"
        print(f"玩家 {p}: {ms}")

    print("\n--- 你的手牌 ---")
    print(tiles_name(env.hands[HUMAN]))
    print("=" * 68)



def _determinize_hidden(env, seed):
    """
    随机化玩家0不可见的牌：
    三名对手暗手 + 牌墙重新洗混，再按原来的暗手张数分配。

    保留：
    - 自己手牌
    - 所有牌河
    - 所有副露
    - 当前阶段/轮次
    - 牌墙剩余张数
    """
    rng = random.Random(seed)

    pool = list(env.wall)
    opponent_sizes = {}

    for player in range(1, 4):
        opponent_sizes[player] = len(env.hands[player])
        pool.extend(env.hands[player])

    rng.shuffle(pool)

    pos = 0
    for player in range(1, 4):
        size = opponent_sizes[player]
        env.hands[player] = sorted(pool[pos:pos + size])
        pos += size

    env.wall = list(pool[pos:])


def _rollout_to_end(env, coach, max_steps=512):
    """
    从当前局面继续打到底。
    后续四个座位全部使用同一个 AI。
    """
    steps = 0

    while not env.is_terminal() and steps < max_steps:
        player = env.current_player
        observation = env.observe(player)
        legal = env.legal_actions(player)

        if not legal:
            raise RuntimeError(
                f"没有合法动作：player={player}, phase={env.phase}"
            )

        action = coach.act(observation, legal)
        env.step(action)
        steps += 1

    if not env.is_terminal():
        raise RuntimeError("rollout 超过最大步数")

    return env.result()



def _rollout_many_to_end(envs, coach, max_steps=512):
    """
    同时推进很多个任意中间局面。
    每一轮把所有仍需决策的局面一起送进 coach.batch_act()，
    避免逐局 GPU inference。
    """
    steps = [0] * len(envs)

    while True:
        active = [
            i for i, env in enumerate(envs)
            if not env.is_terminal() and steps[i] < max_steps
        ]

        if not active:
            break

        observations = []
        legal_batches = []

        for i in active:
            env = envs[i]
            player = env.current_player
            legal = env.legal_actions(player)

            if not legal:
                raise RuntimeError(
                    f"没有合法动作：env={i}, "
                    f"player={player}, phase={env.phase}"
                )

            observations.append(env.observe(player))
            legal_batches.append(legal)

        actions = coach.batch_act(observations, legal_batches)

        for i, action in zip(active, actions):
            envs[i].step(action)
            steps[i] += 1

    for i, env in enumerate(envs):
        if not env.is_terminal():
            raise RuntimeError(
                f"rollout {i} 超过最大步数"
            )

    return [env.result() for env in envs]


def _mean(values):
    return sum(values) / max(1, len(values))


def _paired_ci95(values):
    if len(values) < 2:
        m = _mean(values)
        return m, m

    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    return m - 1.96 * se, m + 1.96 * se


def compare_actions(env, coach, legal, idx_a, idx_b, samples=64):
    """
    对两个合法动作做配对反事实 rollout。

    discard 阶段：
        直接强制执行 A / B。

    claim 阶段：
        清除引擎内部尚未公开的 claim_responses，
        回到本次弃牌后的声明循环起点；
        重新随机化隐藏牌；
        让排在 HUMAN 前面的 AI 重新声明；
        到 HUMAN 时分别强制 A / B；
        之后全部交给 AI 打到底。
    """

    if env.phase not in ("discard", "claim"):
        print(
            f"\n当前阶段 {env.phase!r} 暂不支持反事实比较。\n"
        )
        return

    # 抢补杠阶段的 source 暗手里还保留着待补杠牌。
    # 当前 determinization 会随机化对手暗手，因此先不支持这一特殊情况。
    if (
        env.phase == "claim"
        and (
            env.pending_bugang is not None
            or env.claim_hu_only
        )
    ):
        print(
            "\n目前 c 暂不支持“抢补杠”阶段；"
            "普通弃牌后的吃/碰/杠/胡/过已经支持。\n"
        )
        return

    if idx_a == idx_b:
        print("\n两个动作必须不同。\n")
        return

    if not (0 <= idx_a < len(legal) and 0 <= idx_b < len(legal)):
        print("\n动作编号超出范围。\n")
        return

    action_a = legal[idx_a]
    action_b = legal[idx_b]

    print("\n===== 反事实模拟 =====")
    print(f"A [{idx_a}] {action_name(action_a)}")
    print(f"B [{idx_b}] {action_name(action_b)}")
    print(f"随机可能世界：{samples}")

    if env.phase == "claim":
        print(
            "模式：claim 重放。会清除其他 AI 尚未公开的声明，"
            "从这张弃牌后的声明循环起点重新模拟。"
        )
    else:
        print("模式：正常出牌反事实。")

    print(
        "两个动作使用完全相同的一批隐藏世界。"
    )
    print(
        "这不是精确胜率，而是 determinization rollout 的经验比较。\n"
    )

    worlds_a = []
    worlds_b = []

    # 同一公开局面反复调用得到相同的一组可能世界。
    base_seed = (
        730000
        + len(env.events) * 1000
        + len(env.wall)
    )

    for sample in range(samples):
        base = copy.deepcopy(env)

        if env.phase == "claim":
            # --------------------------------------------------
            # 关键：
            # claim_responses 可能已有排在 HUMAN 前面的 AI
            # 私下提交的响应。
            #
            # 这些响应并不是 HUMAN observation 的公开信息，
            # 所以全部清空，并把声明顺序退回第一家。
            # --------------------------------------------------
            base.claim_responses = {}

            if HUMAN not in base.pending_claimers:
                raise RuntimeError(
                    "当前 HUMAN 不在 pending_claimers 中，"
                    "无法重放 claim 阶段"
                )

            base.current_player = base.pending_claimers[0]

        # 只随机化 HUMAN 看不见的：
        # 三家暗手 + 剩余牌墙。
        _determinize_hidden(
            base,
            base_seed + sample,
        )

        if env.phase == "claim":
            # --------------------------------------------------
            # 如果 HUMAN 不是第一位 responder，
            # 让排在前面的 AI 在这个新的可能世界里重新响应。
            #
            # _record_claim() 在所有人响应完成以前只记录声明，
            # 不会提前修改手牌/副露。
            # --------------------------------------------------
            safety = 0

            while (
                not base.is_terminal()
                and base.phase == "claim"
                and base.current_player != HUMAN
            ):
                player = base.current_player
                obs = base.observe(player)
                pre_legal = base.legal_actions(player)

                if not pre_legal:
                    raise RuntimeError(
                        f"claim 重放时玩家 {player} 没有合法动作"
                    )

                ai_action = coach.act(obs, pre_legal)
                base.step(ai_action)

                safety += 1
                if safety > 3:
                    raise RuntimeError(
                        "claim 重放没有在预期步数内轮到 HUMAN"
                    )

            if base.is_terminal():
                # 理论上正常弃牌 claim 不会在未询问 HUMAN 前
                # 就完成裁决；若环境规则以后改变，这里明确报错。
                raise RuntimeError(
                    "claim 重放在轮到 HUMAN 前已经终局"
                )

            if (
                base.phase != "claim"
                or base.current_player != HUMAN
            ):
                raise RuntimeError(
                    "claim 重放未能恢复到 HUMAN 的决策点"
                )

            # 在新的隐藏世界中重新取得 HUMAN 合法动作，
            # 用 key 对应回原来选择，避免复用旧 Action 对象。
            legal_now = {
                action.key(): action
                for action in base.legal_actions(HUMAN)
            }

            if action_a.key() not in legal_now:
                raise RuntimeError(
                    f"A 在重放世界中不再合法："
                    f"{action_name(action_a)}"
                )

            if action_b.key() not in legal_now:
                raise RuntimeError(
                    f"B 在重放世界中不再合法："
                    f"{action_name(action_b)}"
                )

            world_a = copy.deepcopy(base)
            world_b = copy.deepcopy(base)

            legal_a = {
                action.key(): action
                for action in world_a.legal_actions(HUMAN)
            }
            legal_b = {
                action.key(): action
                for action in world_b.legal_actions(HUMAN)
            }

            world_a.step(legal_a[action_a.key()])
            world_b.step(legal_b[action_b.key()])

        else:
            # 正常摸牌后的 discard 阶段。
            world_a = copy.deepcopy(base)
            world_b = copy.deepcopy(base)

            legal_a = {
                action.key(): action
                for action in world_a.legal_actions(HUMAN)
            }
            legal_b = {
                action.key(): action
                for action in world_b.legal_actions(HUMAN)
            }

            if action_a.key() not in legal_a:
                raise RuntimeError(
                    f"A 在随机世界中不再合法："
                    f"{action_name(action_a)}"
                )

            if action_b.key() not in legal_b:
                raise RuntimeError(
                    f"B 在随机世界中不再合法："
                    f"{action_name(action_b)}"
                )

            world_a.step(legal_a[action_a.key()])
            world_b.step(legal_b[action_b.key()])

        worlds_a.append(world_a)
        worlds_b.append(world_b)

    print(
        f"批量推进 {samples * 2} 条后续路线……"
    )

    all_results = _rollout_many_to_end(
        worlds_a + worlds_b,
        coach,
    )

    results_a = all_results[:samples]
    results_b = all_results[samples:]

    def summarize(result):
        return {
            "score": result["scores"][HUMAN],
            "win": int(result["winner"] == HUMAN),
            "dealin": int(result["loser"] == HUMAN),
            "draw": int(result["winner"] is None),
            "fan": (
                result["fan_count"]
                if result["winner"] == HUMAN
                else 0
            ),
        }

    records_a = []
    records_b = []
    paired_deltas = []

    for result_a, result_b in zip(results_a, results_b):
        a = summarize(result_a)
        b = summarize(result_b)

        records_a.append(a)
        records_b.append(b)

        paired_deltas.append(
            a["score"] - b["score"]
        )

    def aggregate(records):
        wins = [r["fan"] for r in records if r["win"]]

        return {
            "score": _mean(
                [r["score"] for r in records]
            ),
            "win": _mean(
                [r["win"] for r in records]
            ),
            "dealin": _mean(
                [r["dealin"] for r in records]
            ),
            "draw": _mean(
                [r["draw"] for r in records]
            ),
            "fan": _mean(wins) if wins else 0.0,
        }

    a = aggregate(records_a)
    b = aggregate(records_b)

    delta = _mean(paired_deltas)
    ci_low, ci_high = _paired_ci95(
        paired_deltas
    )

    print("\n===== 模拟结果 =====")
    print(
        f"{'':<25}"
        f"{'A':>12}"
        f"{'B':>12}"
    )
    print("-" * 49)

    print(
        f"{'平均得分':<25}"
        f"{a['score']:>12.2f}"
        f"{b['score']:>12.2f}"
    )

    print(
        f"{'和牌率':<25}"
        f"{a['win'] * 100:>11.1f}%"
        f"{b['win'] * 100:>11.1f}%"
    )

    print(
        f"{'放铳率':<25}"
        f"{a['dealin'] * 100:>11.1f}%"
        f"{b['dealin'] * 100:>11.1f}%"
    )

    print(
        f"{'流局率':<25}"
        f"{a['draw'] * 100:>11.1f}%"
        f"{b['draw'] * 100:>11.1f}%"
    )

    print(
        f"{'和牌时平均番数':<25}"
        f"{a['fan']:>12.2f}"
        f"{b['fan']:>12.2f}"
    )

    print("\n--- 配对得分差 ---")
    print(f"A - B = {delta:+.2f}")
    print(
        f"粗略 95% CI："
        f"[{ci_low:+.2f}, {ci_high:+.2f}]"
    )

    if ci_low > 0:
        print(
            "这批可能世界里，A 的 rollout 表现"
            "较稳定地高于 B。"
        )
    elif ci_high < 0:
        print(
            "这批可能世界里，B 的 rollout 表现"
            "较稳定地高于 A。"
        )
    else:
        print(
            "目前样本下差异噪声仍较大，"
            "不能仅凭 rollout 认定哪一个长期结果更好。"
        )

    print(
        "\n注意：隐藏牌目前按可见信息做无条件随机化，"
        "还没有根据对手此前的出牌/副露建立后验分布；"
        "因此这是教学用近似，不是严格博弈论估值。\n"
    )


def human_choose(env, coach):
    obs = env.observe(HUMAN)
    legal = env.legal_actions(HUMAN)

    # 如果只有一个选择，例如只能 PASS，就不用烦你。
    if len(legal) == 1:
        return legal[0]

    print_board(env)

    print("\n可选动作：")
    for i, a in enumerate(legal):
        print(f"  {i:2d}. {action_name(a)}")

    print(
        "\n输入数字选择；a = AI 排名；w = 模型拆解；"
        "c i j [n] = 比较两个动作；q = 退出"
    )

    while True:
        s = input("> ").strip().lower()

        if s == "q":
            raise KeyboardInterrupt

        if s == "a":
            ranking = coach_rank(coach, obs, legal)

            print("\nAI 候选：")
            print("（百分比 = 模型策略偏好，不是客观胜率）")

            for rank, (idx, action, probability) in enumerate(ranking, 1):
                marker = "  ← 推荐" if rank == 1 else ""
                print(
                    f"{rank:2d}. [{idx:2d}] "
                    f"{action_name(action):<24} "
                    f"{probability * 100:6.2f}%{marker}"
                )

            continue

        if s == "w":
            coach_analyze(coach, obs, legal)
            continue

        if s.startswith("c "):
            parts = s.split()

            if len(parts) not in (3, 4):
                print(
                    "格式：c 动作A编号 动作B编号 [模拟世界数]\\n"
                    "例如：c 9 2 或 c 9 2 30"
                )
                continue

            try:
                idx_a = int(parts[1])
                idx_b = int(parts[2])
                samples = int(parts[3]) if len(parts) == 4 else 64
            except ValueError:
                print("动作编号和样本数必须是整数。")
                continue

            if samples < 2 or samples > 200:
                print("模拟世界数请选择 2～200。")
                continue

            compare_actions(
                env,
                coach,
                legal,
                idx_a,
                idx_b,
                samples,
            )
            continue

        try:
            idx = int(s)
            if 0 <= idx < len(legal):
                chosen = legal[idx]
                print(f"\n你选择：{action_name(chosen)}")
                return chosen
        except ValueError:
            pass

        print("请输入合法编号、a 或 q。")


def main():
    print("加载麻将模型……")

    model, metadata = load_model_from_checkpoint(
        "artifacts/botzone_model.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    coach = ModelPolicy(model)

    print(f"模型已加载：{device}")
    print("你是 玩家 0。")
    print("这是国标规则，目前至少 8 番才能胡。")

    env = MahjongEnv()
    env.reset()

    try:
        while not env.is_terminal():
            player = env.current_player
            legal = env.legal_actions(player)

            if player == HUMAN:
                action = human_choose(env, coach)
            else:
                obs = env.observe(player)
                action = coach.act(obs, legal)

            # 只显示真正执行完成的公开事件，
            # 不泄露其他 AI 尚未裁决的吃碰杠胡声明。
            old_event_count = len(env.events)
            env.step(action)

            for event in env.events[old_event_count:]:
                if event[0] == "PLAY":
                    _, p, tile = event
                    if p != HUMAN:
                        print(f"玩家 {p}: 打 {tile_name(tile)}")

                elif event[0] == "CHI":
                    _, p, tile, discard = event
                    if p != HUMAN:
                        print(
                            f"玩家 {p}: 吃 {tile_name(tile)}"
                            f" → 打 {tile_name(discard)}"
                        )

                elif event[0] == "PENG":
                    _, p, tile, discard = event
                    if p != HUMAN:
                        print(
                            f"玩家 {p}: 碰 {tile_name(tile)}"
                            f" → 打 {tile_name(discard)}"
                        )

                elif event[0] == "GANG":
                    _, p, tile = event
                    if p != HUMAN:
                        print(f"玩家 {p}: 杠 {tile_name(tile)}")

                elif event[0] == "BUGANG":
                    _, p, tile = event
                    if p != HUMAN:
                        print(f"玩家 {p}: 补杠 {tile_name(tile)}")

    except KeyboardInterrupt:
        print("\n已退出本局。")
        return

    print_board(env)

    result = env.result()

    print("\n===== 本局结束 =====")
    if result["winner"] is None:
        print("流局")
    else:
        winner = result["winner"]
        if result["self_drawn"]:
            print(f"玩家 {winner} 自摸")
        else:
            print(f"玩家 {winner} 胡，玩家 {result['loser']} 点炮")

        print(f"番数：{result['fan_count']}")

    print("得分：")
    for p, score in enumerate(result["scores"]):
        marker = " ← 你" if p == HUMAN else ""
        print(f"  玩家 {p}: {score:+d}{marker}")


if __name__ == "__main__":
    main()
