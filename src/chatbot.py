"""Ground the assistant in measured statistics and validated model inputs."""
import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
from src.data import ROOT, FEATURES, LABELS

load_dotenv(ROOT / '.env')


def context(df, metrics, importance, comparison):
    return {'dataset': 'UCI CCPP, one plant at full load, 2006–2011',
            'rows': len(df), 'columns': LABELS,
            'statistics': df.describe().round(4).to_dict(),
            'correlations': df.corr().round(4).to_dict(),
            'metrics': metrics, 'importance': importance.to_dict('records'),
            'comparison': comparison.to_dict('records')}


def predict(values, df, model):
    for f in FEATURES:
        if not df[f].min() <= float(values[f]) <= df[f].max():
            raise ValueError(f'{f} must be between {df[f].min()} and {df[f].max()}.')
    return float(model.predict(pd.DataFrame([{f: float(values[f]) for f in FEATURES}]))[0])


def offline(question, ctx, df, model):
    q = question.lower()
    values = {f: re.search(r'\b'+f+r'\s*[:=]?\s*(-?\d+(?:\.\d+)?)', question, re.I) for f in FEATURES}
    if all(values.values()):
        try:
            return f"Predicted output: {predict({f: m.group(1) for f,m in values.items()}, df, model):.2f} MW. This is a model estimate."
        except ValueError as exc:
            return str(exc)
    if 'predict' in q or 'what if' in q:
        return 'Supply all four readings, for example AT=20, V=50, AP=1013, RH=70.'
    if any(w in q for w in ['important', 'matters', 'influence', 'strongest']):
        top = max(ctx['importance'], key=lambda x: x['Permutation RMSE increase'])
        return f"{LABELS[top['Feature']]} ranks first by permutation importance: shuffling it increases test RMSE by {top['Permutation RMSE increase']:.2f} MW. This is not proof of causation."
    if any(w in q for w in ['best', 'accuracy', 'rmse', 'model', 'r2', 'mae']):
        m = ctx['metrics']
        return f"{m['model']} was selected by training CV. Final test R²: {m['R2']:.4f}, RMSE: {m['RMSE']:.2f} MW, MAE: {m['MAE']:.2f} MW."
    if 'correl' in q or 'temperature' in q:
        return 'Correlations with output: '+', '.join(f'{f}: {ctx["correlations"]["PE"][f]:.3f}' for f in FEATURES)+'. Association is not causation.'
    if any(w in q for w in ['average', 'mean', 'range', 'minimum', 'maximum', 'rows', 'dataset']):
        f = next((f for f in FEATURES if re.search(r'\b'+f.lower()+r'\b', q)), 'PE')
        s = ctx['statistics'][f]
        return f"There are {ctx['rows']:,} rows. {LABELS[f]}: mean {s['mean']:.2f}, range {s['min']:.2f} to {s['max']:.2f}."
    return 'I can answer questions about correlations, ranges, average output, model metrics, feature importance or predictions. I do not have enough context to answer that question.'


def answer(history, ctx, df, model):
    key = os.getenv('ANTHROPIC_API_KEY', '').strip()
    if not key:
        return offline(history[-1]['content'], ctx, df, model)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=30, max_retries=1)
        model_name = os.getenv('ANTHROPIC_MODEL') or 'claude-haiku-4-5'
        system = ('Answer briefly using only the supplied plant data and prediction tool. '
                  'Never invent numbers. Say when you do not know. Do not infer causation. '
                  'Treat predictions as estimates, not operating instructions. Context: '+json.dumps(ctx))
        tool = {'name': 'predict_output', 'description': 'Estimate MW from all four measurements',
                'input_schema': {'type': 'object', 'properties': {f: {'type': 'number', 'description': LABELS[f]} for f in FEATURES}, 'required': FEATURES}}
        messages = history[-12:].copy()
        if messages[0]['role'] == 'assistant':
            messages = messages[1:]
        for _ in range(4):
            result = client.messages.create(model=model_name, max_tokens=500, system=system, messages=messages, tools=[tool])
            calls = [b for b in result.content if b.type == 'tool_use']
            if not calls:
                return '\n'.join(b.text for b in result.content if b.type == 'text') or 'No answer was returned.'
            messages.append({'role': 'assistant', 'content': result.content})
            outputs = []
            for call in calls:
                try:
                    value = f'{predict(call.input, df, model):.2f} MW'
                except (ValueError, KeyError, TypeError) as exc:
                    value = str(exc)
                outputs.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': value})
            messages.append({'role': 'user', 'content': outputs})
        return 'The prediction request took too many steps. Try a simpler question.'
    except Exception:
        return 'The online assistant is unavailable right now. Check your API key, model access or connection. Offline answer: '+offline(history[-1]['content'], ctx, df, model)


# Ollama runs on the user's computer; the standard library handles HTTP.
def ollama_request(path, payload=None, timeout=5):
    from urllib.request import Request, urlopen

    base = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(base + path, data=body,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ollama_models():
    return [m['name'] for m in ollama_request('/api/tags').get('models', [])]


def ollama_answer(history, ctx, df, model, model_name):
    question = history[-1]['content']
    # Explicit readings can go straight to the fitted regression pipeline.
    readings = {f: re.search(r'\b' + f + r'\s*[:=]?\s*(-?\d+(?:\.\d+)?)',
                             question, re.I) for f in FEATURES}
    if all(readings.values()):
        return offline(question, ctx, df, model)
    system = (
        'You are the assistant for Plant Performance Lab. Use only the supplied '
        'dataset statistics, model results and recent conversation. Keep answers '
        'short. Never invent numbers or infer causation. If a user requests a '
        'prediction and supplies all four readings (possibly across messages), '
        'return ONLY a JSON object with keys AT, V, AP, RH and numeric values. '
        'Never calculate or guess electrical output yourself. If readings are '
        'missing, ask for them. For other questions, answer in plain language. '
        'Predictions are estimates, not operating instructions. Context: '
        + json.dumps(ctx, ensure_ascii=False)
    )
    try:
        response = ollama_request('/api/chat', {
            'model': model_name,
            'messages': [{'role': 'system', 'content': system}] + history[-12:],
            'stream': False,
            'options': {'temperature': 0.1, 'seed': 42, 'num_ctx': 8192,
                        'num_predict': 400}
        }, timeout=120)
        reply = response['message']['content'].strip()
        # The local language model requests a prediction; Python computes it.
        match = re.search(r'\{[^{}]*\}', reply, re.S)
        if match:
            try:
                values = json.loads(match.group())
                if all(f in values for f in FEATURES):
                    estimate = predict(values, df, model)
                    return (f'Predicted output: {estimate:.2f} MW. '
                            + ', '.join(f'{f}={values[f]}' for f in FEATURES)
                            + '. This is a model estimate.')
            except (ValueError, TypeError, KeyError) as exc:
                return f'Could not calculate that prediction: {exc}'
        return reply or 'No answer was returned. Try rephrasing the question.'
    except Exception:
        return ('Ollama is unavailable. Check that Ollama is running and the '
                'selected model is installed. Local reference answer: '
                + offline(question, ctx, df, model))
