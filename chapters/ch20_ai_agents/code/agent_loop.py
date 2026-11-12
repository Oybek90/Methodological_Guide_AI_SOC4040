# ci: run
"""A ReAct-style agent loop with tools, step budget and error feedback (no LLM needed).

The 'policy' is any function history -> action. Here a scripted policy stands in for
the language model, so that the LOOP (tools, observations, budget, errors) can be
tested exactly.
"""
import json
import random
import re
from dataclasses import dataclass, field

KB = {   # a tiny 'search' knowledge base: (made-up) facts for the tests
    "population of tashkent": "2.9 million",
    "population of samarkand": "0.6 million",
    "height of registan": "35 m",
}


class ToolError(Exception):
    pass


def calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ToolError("only digits and + - * / ( ) are allowed")
    return str(eval(expression, {"__builtins__": {}}, {}))   # safe: whitelist above


def search(query: str) -> str:
    key = query.lower().strip(" ?.")
    if key in KB:
        return KB[key]
    raise ToolError(f"no result for '{query}'")


TOOLS = {"calculator": calculator, "search": search}


@dataclass
class Trace:
    steps: list = field(default_factory=list)   # (action, observation)
    answer: str = None
    reason: str = ""


def run_agent(policy, question, tools=TOOLS, max_steps=6, feed_errors=True):
    """Loop: policy -> action; run tool; append observation; stop on 'final' or budget."""
    tr = Trace()
    history = [("question", question)]
    for _ in range(max_steps):
        act = policy(history)                       # dict: {"tool": name, "input": str} or {"final": str}
        if "final" in act:
            tr.answer, tr.reason = act["final"], "final"
            return tr
        try:
            obs = tools[act["tool"]](act["input"])
        except (KeyError, ToolError, SyntaxError, ZeroDivisionError) as e:
            obs = f"ERROR: {e}"
            if not feed_errors:
                tr.steps.append((act, obs)); tr.reason = "aborted"
                return tr
        tr.steps.append((act, obs))
        history.append(("action", json.dumps(act)))
        history.append(("observation", obs))
    tr.reason = "budget"
    return tr


def scripted_policy(eps=0.0, seed=0):
    """Solves 'population of A times k': search, then calculator, then final.
    With probability eps a step is malformed (as a real LLM sometimes does)."""
    rng = random.Random(seed)

    def policy(history):
        q = history[0][1].lower()
        city, k = re.search(r"population of (\w+) times (\d+)", q).groups()
        obs = [o for kind, o in history if kind == "observation" and not o.startswith("ERROR")]
        if rng.random() < eps:
            return {"tool": "serach", "input": "oops"}           # malformed tool name
        if len(obs) == 0:
            return {"tool": "search", "input": f"population of {city}"}
        if len(obs) == 1:
            val = obs[0].split()[0]
            return {"tool": "calculator", "input": f"{val} * {k}"}
        return {"final": f"{obs[1]} million"}
    return policy


def success_rate(eps, budget, feed_errors, n=20000, seed=1):
    ok = 0
    for i in range(n):
        tr = run_agent(scripted_policy(eps, seed * 1_000_003 + i),
                       "What is the population of Tashkent times 2?",
                       max_steps=budget, feed_errors=feed_errors)
        ok += tr.answer == "5.8 million"
    return ok / n


if __name__ == "__main__":
    tr = run_agent(scripted_policy(0.0), "What is the population of Tashkent times 2?")
    for a, o in tr.steps:
        print(a, "->", o)
    print("answer:", tr.answer, "| stop reason:", tr.reason)
    assert tr.answer == "5.8 million" and len(tr.steps) == 2

    # tools fail loudly: unknown search and unsafe expression become observations
    assert run_agent(lambda h: {"tool": "search", "input": "mars"} if len(h) < 3 else {"final": "x"},
                     "q").steps[0][1].startswith("ERROR")
    assert "ERROR" in run_agent(lambda h: {"tool": "calculator", "input": "__import__('os')"}
                                if len(h) < 3 else {"final": "x"}, "q").steps[0][1]
    # a policy that never stops is cut off by the budget
    loop = run_agent(lambda h: {"tool": "search", "input": "population of tashkent"}, "q", max_steps=4)
    print("never-ending policy:", loop.reason, "after", len(loop.steps), "steps")
    assert loop.reason == "budget" and len(loop.steps) == 4

    # robustness: the same malformed-step rate, with and without error feedback
    eps = 0.2
    print("eps =", eps, " (each step malformed with this probability; 3 correct steps needed)")
    print("  abort on first error     :", round(success_rate(eps, 6, False), 3),
          " closed form (1-eps)^3 =", round((1 - eps) ** 3, 3))
    for B in (3, 4, 5, 6, 8):
        from math import comb
        exact = sum(comb(B, j) * (1 - eps) ** j * eps ** (B - j) for j in range(3, B + 1))
        print(f"  feed errors back, budget {B}: {success_rate(eps, B, True):.3f}   exact {exact:.3f}")
    print("agent_loop self-test passed")
