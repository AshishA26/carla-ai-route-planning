"""Runs one model turn with tools, in fake mode (scripted) or against the OpenAI API.

Both modes call the same tool implementations and produce the same transcript, so the rest of
the pipeline cannot tell them apart. Real mode is used when OPENAI_API_KEY is set and
--fake is not passed.
"""
import json
import os
import time
import envfile  # noqa: F401  (loads .env: OPENAI_API_KEY, OPENAI_MODEL)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

ROUTE_SYSTEM = """You are the route-planning assistant inside an autonomous car.
Turn the rider's request into a route using your tools. Always check the vehicle state and plan
before committing. Prefer the rider's stated priorities (time vs. comfort vs. safety), never exceed
the posted limit, and decline anything that breaks traffic law with report_unsafe_request.
A deterministic validator checks every commit; if it rejects, read the issues and fix your request.
Finish with one or two plain sentences to the rider."""


def run_turn(session, specs, system, user, script=None, fake=None, model=None, max_steps=10):
    """Returns the transcript: a list of {"kind", ...} events with wall-clock offsets."""
    fake = fake if fake is not None else not os.environ.get("OPENAI_API_KEY")
    transcript = [{"kind": "user", "text": user, "mode": "fake" if fake else (model or DEFAULT_MODEL)}]
    t0 = time.time()
    if fake:
        _run_script(session, script, transcript, t0)
    else:
        _run_openai(session, specs, system, user, transcript, t0, model or DEFAULT_MODEL, max_steps)
    return transcript


def _log(transcript, t0, **ev):
    ev["t"] = round(time.time() - t0, 3)
    transcript.append(ev)


def _run_script(session, script, transcript, t0):
    for name, args in script:
        if name == "say":
            _log(transcript, t0, kind="say", text=args(session, transcript) if callable(args) else args)
            continue
        if callable(args):
            args = args(session, transcript)
        _log(transcript, t0, kind="tool_call", name=name, args=args)
        _log(transcript, t0, kind="tool_result", name=name, result=session.call(name, args))


def _run_openai(session, specs, system, user, transcript, t0, model, max_steps):
    from openai import OpenAI

    client = OpenAI()
    tools = [{"type": "function", "function": {**s, "strict": True}} for s in specs]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            _log(transcript, t0, kind="say", text=(msg.content or "").strip())
            return
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            _log(transcript, t0, kind="tool_call", name=tc.function.name, args=args)
            result = session.call(tc.function.name, args)
            _log(transcript, t0, kind="tool_result", name=tc.function.name, result=result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    _log(transcript, t0, kind="say", text="(stopped after the step limit)")
