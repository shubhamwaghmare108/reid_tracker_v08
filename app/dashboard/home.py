"""Home page: a quick health and navigation overview for the ReID dashboard."""
import streamlit as st
from app.config import Settings
from app.dashboard.shared import apply_theme, display_timestamp, load_runs

apply_theme()
st.markdown(
    '''
    <div class='hero'>
        <div class='eyebrow'>Presence intelligence</div>
        <div class='status'>● DATABASE CONNECTED</div>
        <h1>Person ReID Control Center</h1>
        <p>Monitor recognized visitors, review activity, and manage your enrolled identities.</p>
    </div>
    ''',
    unsafe_allow_html=True
)
# The same data query also makes the completed-run count visible at a glance.
runs = load_runs()
database_ready = bool(Settings().mysql_host and Settings().mysql_user and Settings().mysql_database)
columns = st.columns(3)
for column, label, value, detail in zip(
    columns,
    ['Completed runs', 'Data source', 'Navigation'],
    [str(len(runs)), 'MySQL' if database_ready else 'Not configured', '3 pages'],
    ['Available for analysis', 'Streamlit secrets' if database_ready else 'Add database secrets', 'Home · Analytics · Admin'],
):
    with column:
        st.markdown(
            f'''<div class='metric-card'><div class='metric-name'>{label}</div><div class='metric-value'>{value}</div><div class='metric-detail'>{detail}</div></div>''',
            unsafe_allow_html=True
        )
st.markdown("<div class='section-title'>Get started</div>", unsafe_allow_html=True)
left, right = st.columns(2, gap='large')
with left:
    st.markdown(
        "<div class='info-panel'><h3>Review activity</h3><p>Open Analytics to inspect a camera run, compare recognized people, and follow the IN / OUT timeline.</p></div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        "<div class='info-panel'><h3>Keep the gallery current</h3><p>Use Admin to add reference images or archive old runs when your workspace needs tidying.</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Recent camera runs</div>", unsafe_allow_html=True)
if runs:
    st.dataframe(
        [
            {
                'Run': f"#{run['id']}",
                'Source': run['source'],
                'Started': display_timestamp(run['started_at']),
                'Completed': display_timestamp(run['completed_at']),
            }
            for run in runs[:5]
        ],
        width='stretch',
        hide_index=True,
    )
else:
    st.info('No completed camera runs are available yet. Start the pipeline to populate this overview.')
