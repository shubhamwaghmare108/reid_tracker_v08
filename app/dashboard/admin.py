"""Password-protected page for enrollment uploads and soft-archiving camera runs."""
import os
import re
from uuid import uuid4

import streamlit as st

from app.config import Settings
from app.dashboard.shared import admin_authenticated, apply_theme, delete_run, display_timestamp, load_runs, render_page_header
from app.utils.utils import IMAGE_SUFFIXES


apply_theme()
render_page_header('Administration', 'Enroll known identities and archive completed runs.')
if not admin_authenticated():
    st.stop()

account_status, logout_area = st.columns([5, 1])
with account_status:
    st.caption('Signed in as administrator')
with logout_area:
    if st.button('Log out', key='admin_logout', use_container_width=True):
        st.session_state.pop('admin_authenticated', None)
        st.rerun()

st.markdown("<div class='section-title'>Enroll known person</div>", unsafe_allow_html=True)
name = st.text_input('Identity name', placeholder='e.g. Shubham')
images = st.file_uploader(
    'Reference images', type=[suffix[1:] for suffix in IMAGE_SUFFIXES], accept_multiple_files=True
)
if st.button('Save enrollment images'):
    # Restrict names so they are safe as folder names on supported operating systems.
    if not re.fullmatch(r'[A-Za-z0-9 _-]{1,80}', name.strip()):
        st.error('Use 1-80 letters, numbers, spaces, underscores, or hyphens.')
    elif not images:
        st.error('Select at least one image.')
    else:
        folder = Settings().gallery_dir / name.strip() / 'images'
        folder.mkdir(parents=True, exist_ok=True)
        for image in images:
            suffix = os.path.splitext(image.name)[1].lower()
            (folder / f'{uuid4().hex}{suffix}').write_bytes(image.getvalue())
        person_dir = folder.parent
        # New images invalidate cached means; the next run will rebuild them.
        for embedding_file in ('embedding_mean.npy', 'face_embedding_mean.npy'):
            (person_dir / embedding_file).unlink(missing_ok=True)
        st.success(f'Saved {len(images)} image(s). The body gallery will rebuild on the next run.')

st.markdown("<div class='section-title'>Archive stored run</div>", unsafe_allow_html=True)
runs = load_runs()
if runs:
    labels = {f"Run #{run['id']} | {run['source']} | {display_timestamp(run['started_at'])}": run for run in runs}
    run = labels[st.selectbox('Run to archive', list(labels))]
    # A second confirmation prevents a one-click archive from an accidental selection.
    confirmed = st.checkbox(f"Archive Run #{run['id']} and its records")
    if st.button('Archive selected run', type='primary'):
        if not confirmed:
            st.error('Confirm the archive first.')
        else:
            delete_run(run['id'])
            st.cache_data.clear()
            st.success('Run archived.')
            st.rerun()
