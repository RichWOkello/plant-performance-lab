"""Interactive plant analytics, predictions and a grounded chat assistant."""
import sys
import os
import json
from pathlib import Path
from urllib.parse import urlparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import ROOT, FEATURES, LABELS, load_data
from src.chatbot import (context, answer, predict, offline,
                         ollama_answer, ollama_models)

st.set_page_config(page_title='Plant Performance Lab', layout='wide')
st.markdown('''<style>[data-testid="stMetric"] {background:white;border:1px solid #d7e3ea;border-radius:12px;padding:16px;} .block-container {padding-top:2rem;} h1 {letter-spacing:-1px;} </style>''', unsafe_allow_html=True)
st.caption('ENERGY SYSTEMS / DATA SCIENCE CAPSTONE')
st.title('Plant Performance Lab')
st.write('Hourly power output prediction for a combined cycle power plant.')


def project_files():
    """Keep the analysis and slides close to the dashboard."""
    st.sidebar.divider()
    st.sidebar.subheader('Project files')
    files = [
        ('Download notebook', ROOT / 'notebooks/capstone.ipynb',
         'application/x-ipynb+json'),
        ('Download presentation', ROOT / 'reports/Plant_Performance_Lab_Presentation.pptx',
         'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
    ]
    for label, path, mime in files:
        if path.is_file():
            st.sidebar.download_button(
                label, data=path.read_bytes(), file_name=path.name,
                mime=mime, use_container_width=True,
            )
        else:
            st.sidebar.caption(f'{path.name} is not in the project folder.')
    slides_url = os.getenv('GOOGLE_SLIDES_URL', '').strip()
    parsed = urlparse(slides_url)
    if (parsed.scheme == 'https' and parsed.hostname == 'docs.google.com'
            and parsed.path.startswith('/presentation/')):
        st.sidebar.link_button('Open presentation in Google Slides', slides_url,
                               use_container_width=True)
    else:
        st.sidebar.link_button('Open Google Slides', 'https://slides.google.com',
                               use_container_width=True)
        st.sidebar.caption('Upload the downloaded slides to Google Slides once. '
                           'Add the sharing link to .env to open them here directly.')
    st.sidebar.caption('Open the notebook in VS Code or Jupyter.')


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return joblib.load(ROOT / 'models/best_model.joblib')


@st.cache_data
def get_results():
    p = ROOT / 'models'
    return (json.loads((p/'metrics.json').read_text()), pd.read_csv(p/'comparison.csv'),
            pd.read_csv(p/'importance.csv'), pd.read_csv(p/'predictions.csv'))


page = st.sidebar.radio('Workspace', ['Overview', 'Explore', 'Predict', 'Model Performance', 'Ask the Data'])
st.sidebar.caption('UCI dataset · 2006–2011 · Full-load observations')
project_files()

try:
    df = get_data()
    model = get_model()
    metrics, comparison, importance, pairs = get_results()
except Exception as exc:
    st.warning('Project data or trained model is missing. Run notebooks/capstone.ipynb from top to bottom, then restart the dashboard.')
    st.caption(str(exc))
    st.stop()



def chart(fig):
    fig.update_layout(template='plotly_white', margin=dict(l=20,r=20,t=45,b=20))
    st.plotly_chart(fig, use_container_width=True)


if page == 'Overview':
    for col, (label, value) in zip(st.columns(5), [('Records', f'{len(df):,}'), ('Mean output', f'{df.PE.mean():.2f} MW'), ('Selected model', metrics['model']), ('Test R²', f'{metrics["R2"]:.3f}'), ('Test RMSE', f'{metrics["RMSE"]:.2f} MW')]):
        col.metric(label, value)
    st.subheader('From sensor readings to power estimates')
    st.write('Four measurements describe the conditions under which this plant was running. The model estimates output at full load, which can help explain deviations from expected generation. It is not a plant control system.')
    chart(px.histogram(df, x='PE', nbins=45, labels=LABELS, title='Observed electrical output'))
elif page == 'Explore':
    selected = st.selectbox('Feature', FEATURES, format_func=lambda f: LABELS[f])
    filtered = df.copy()
    for col, f in zip(st.columns(4), FEATURES):
        lo, hi = float(df[f].min()), float(df[f].max())
        a,b = col.slider(LABELS[f], lo, hi, (lo,hi))
        filtered = filtered[filtered[f].between(a,b)]
    st.metric('Matching records', f'{len(filtered):,}')
    if filtered.empty:
        st.info('No matching observations. Widen the filters.')
    else:
        fig = px.scatter(filtered, x=selected, y='PE', opacity=.35, labels=LABELS)
        if filtered[selected].nunique() > 1:
            x = np.array([filtered[selected].min(), filtered[selected].max()])
            fig.add_scatter(x=x, y=np.polyval(np.polyfit(filtered[selected], filtered.PE, 1), x), mode='lines', name='Linear trend')
        chart(fig)
        chart(px.histogram(filtered, x=selected, labels=LABELS, nbins=35))
        chart(px.imshow(filtered.corr(), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', text_auto='.2f', title='Filtered correlations'))
elif page == 'Predict':
    vals = {f: col.slider(LABELS[f], float(df[f].min()), float(df[f].max()), float(df[f].median()), step=.01) for col,f in zip(st.columns(4),FEATURES)}
    output = predict(vals, df, model)
    st.metric('Estimated output', f'{output:.2f} MW', f'{output-df.PE.mean():+.2f} MW vs historical mean')
    lo,hi = df.PE.min(),df.PE.max()
    chart(go.Figure(go.Indicator(mode='gauge+number', value=output, number={'suffix':' MW'}, gauge={'axis':{'range':[min(lo,output),max(hi,output)]},'bar':{'color':'#087F8C'},'threshold':{'value':df.PE.mean(),'line':{'color':'#C47A20','width':4}}})))
    st.caption(f'Observed range: {lo:.2f}–{hi:.2f} MW. Orange marker is the historical mean.')
    if vals['AT']+5 <= df.AT.max():
        changed = predict({**vals,'AT':vals['AT']+5},df,model)
        st.metric('What if temperature rises by 5°C?', f'{changed:.2f} MW', f'{changed-output:+.2f} MW')
    else:
        st.info('A 5°C rise would leave the observed temperature range. No extrapolation is shown.')
    st.caption('Other readings are held fixed in this scenario; their combination may not be typical of real plant operation.')
elif page == 'Model Performance':
    st.write('Model selection uses five-fold training CV. Test results are descriptive, not used to select the winner.')
    st.dataframe(comparison, hide_index=True)
    chart(px.bar(comparison, x='Model', y=['CV RMSE','RMSE'], barmode='group', title='Baseline model comparison (MW)'))
    st.write('Tuned model:', metrics)
    fig = px.scatter(pairs, x='Actual', y='Predicted', opacity=.4)
    lo,hi = pairs.Actual.min(),pairs.Actual.max()
    fig.add_scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Perfect prediction')
    chart(fig)
    fig = px.scatter(pairs,x='Predicted',y='Residual',opacity=.4)
    fig.add_hline(y=0)
    chart(fig)
    chart(px.bar(importance.sort_values('Permutation RMSE increase'),x='Permutation RMSE increase',y='Feature',orientation='h',title='Permutation importance (MW)'))
    chart(px.bar(importance,x='Feature',y='Native importance',title='Native model importance'))
else:
    st.subheader('Ask the Data')
    provider = st.radio('Assistant engine', ['Local Ollama', 'Offline reference', 'Anthropic Claude'], horizontal=True)
    local_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
    if provider == 'Local Ollama':
        local_model = st.text_input('Ollama model', local_model)
        try:
            installed = ollama_models()
            if local_model in installed:
                st.success(f'Connected to local Ollama · {local_model}')
            else:
                st.warning('Model not found. Installed: ' + ', '.join(installed))
        except Exception:
            st.info('Start Ollama on your computer. The offline reference will work if Ollama is unavailable.')
        st.caption('Local AI · No API key or cloud connection required.')
    elif provider == 'Anthropic Claude' and not os.getenv('ANTHROPIC_API_KEY', '').strip():
        st.info('Add ANTHROPIC_API_KEY to .env to use Claude. Until then, the offline reference is used.')
    if 'history' not in st.session_state:
        st.session_state.history = []
    if st.button('Clear chat'):
        st.session_state.history = []
    prompt = None
    questions = ['Which weather variable matters most?', 'What is the best model?', 'How does temperature correlate with output?', 'Predict AT=20, V=50, AP=1013, RH=70']
    for col,q in zip(st.columns(4),questions):
        if col.button(q):
            prompt = q
    typed = st.chat_input('Ask about the plant data…')
    prompt = typed or prompt
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    if prompt:
        st.session_state.history.append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.write(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Checking the data…'):
                ctx = context(df, metrics, importance, comparison)
                if provider == 'Local Ollama':
                    reply = ollama_answer(st.session_state.history, ctx, df, model, local_model)
                elif provider == 'Anthropic Claude':
                    reply = answer(st.session_state.history, ctx, df, model)
                else:
                    reply = offline(prompt, ctx, df, model)
            st.write(reply)
        st.session_state.history.append({'role':'assistant','content':reply})
