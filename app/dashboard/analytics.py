"""Analytics page for reviewing a stored camera run and its presence events."""
import streamlit as st

from app.dashboard.shared import apply_theme, display_timestamp, load_run_details, load_runs, render_page_header


apply_theme()
render_page_header('Run analytics', 'Review known-person presence and detailed movement history.')
if st.button('Refresh data'):
    # Cached results are useful normally, but users can explicitly request fresh data.
    st.cache_data.clear()
    st.rerun()

runs = load_runs()
if not runs:
    st.info('No completed camera runs have been saved yet.')
    st.stop()

# Use readable labels in the widget while retaining the complete database row as its value.
labels = {f"Run #{run['id']} | {run['source']} | {display_timestamp(run['started_at'])}": run for run in runs}
run = labels[st.selectbox('Camera run', list(labels))]
summary, events = load_run_details(run['id'])
st.caption(f"{run['source']} | {display_timestamp(run['started_at'])} -> {display_timestamp(run['completed_at'])}")
if not summary:
    st.info('No known identities were detected in this run.')
    st.stop()

total_seconds = sum(float(person['total_seconds']) for person in summary)
columns = st.columns(3)
for column, label, value in zip(
    columns,
    ['Recognized people', 'Combined presence', 'IN events'],
    [str(len(summary)), f'{total_seconds:.1f}s', str(sum(int(person['entries_count']) for person in summary))],
):
    with column:
        st.markdown(
            f"<div class='metric-card'><div class='metric-name'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='section-title'>Presence summary</div>", unsafe_allow_html=True)
st.dataframe(summary, width='stretch', hide_index=True)
st.markdown("<div class='section-title'>IN / OUT timeline</div>", unsafe_allow_html=True)
if events:
    st.dataframe(
        [dict(event, occurred_at=display_timestamp(event['occurred_at'])) for event in events],
        width='stretch',
        hide_index=True,
    )
else:
    st.info('No completed transitions recorded.')
