"""Shared Streamlit styling, database reads, and authentication helpers."""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import pymysql
import streamlit as st
from dotenv import dotenv_values
from zoneinfo import ZoneInfo
from app.config import Settings

TIMEZONE_OPTIONS = {
    'India (IST)': 'Asia/Kolkata',
    'UTC': 'UTC',
    'United Kingdom (GMT/BST)': 'Europe/London',
    'US Eastern (ET)': 'America/New_York',
    'Singapore (SGT)': 'Asia/Singapore',
}


def apply_theme():
    """Inject the small CSS theme used consistently across all dashboard pages."""
    st.markdown(
        '''
        <style>
        .stApp { background:#080d13; color: #e8eef9; }
        [data-testid='stAppViewContainer'], [data-testid='stAppViewContainer'] > .main { background: radial-gradient(circle at 8% 0%, #20384e 0%, #101a24 38%, #080d13 100%); }
        #MainMenu, footer, [data-testid='stDecoration'], [data-testid='stAppDeployButton'], .stAppDeployButton { display:none; }
        [data-testid='stToolbar'] { display:flex !important; }
        [data-testid='stToolbarActions'] { display:none !important; }
        [data-testid='stExpandSidebarButton'] { display:flex !important; visibility:visible !important; }
        .block-container { width:100%; max-width:1420px; padding:1.5rem 2.5rem 3.5rem; }
        .hero { padding: 2.25rem 2.4rem; border: 1px solid rgba(155,205,202,.28); border-radius: 20px; background: linear-gradient(125deg,rgba(27,66,77,.94),rgba(15,29,41,.92)); box-shadow: 0 18px 46px rgba(0,0,0,.24); }
        .eyebrow { color:#8bd4ca; font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
        .hero h1 { margin:.35rem 0 .45rem; font-size:2.35rem; letter-spacing:-.04em; color:#fff; }
        .hero p { max-width:680px; margin:0; color:#bfd0d8; font-size:1.02rem; }
        .status { float:right; margin-top:-2.3rem; color:#a8f3c8; background:rgba(40,180,104,.14); padding:.42rem .75rem; border-radius:999px; font-size:.8rem; font-weight:650; }
        .page-header { margin: .4rem 0 1.5rem; }
        .page-header h1 { margin:0; color:#fff; font-size:2rem; letter-spacing:-.035em; }
        .page-header p { margin:.35rem 0 0; color:#aebfca; font-size:.98rem; }
        .metric-card { min-height:120px; padding:1.1rem 1.2rem; border-radius:14px; background:rgba(19,34,45,.84); border:1px solid rgba(167,208,205,.16); }
        .metric-name { color:#aebed4; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
        .metric-value { color:#f7fbff; font-size:1.8rem; font-weight:740; margin-top:.5rem; }
        .metric-detail { color:#8bd4ca; font-size:.84rem; margin-top:.3rem; }
        .section-title { color:#f2f7ff; font-size:1.12rem; font-weight:680; margin:1.8rem 0 .65rem; }
        .section-copy { color:#aebfca; line-height:1.55; }
        .info-panel { min-height:150px; padding:1.2rem 1.3rem; border:1px solid rgba(167,208,205,.14); border-radius:14px; background:rgba(14,26,36,.78); }
        .info-panel h3 { margin:0 0 .45rem; color:#f4f8fb; font-size:1.05rem; }
        .info-panel p { margin:0; color:#aebfca; line-height:1.5; }
        .stButton > button { border-radius:8px; border:1px solid rgba(139,212,202,.46); background:#197d78; color:white; font-weight:650; }
        .stButton > button:hover { background:#23968f; }
        [data-testid='stFileUploader'] button { border-radius:8px !important; border:1px solid rgba(139,212,202,.46) !important; background:#197d78 !important; color:#ffffff !important; font-weight:650 !important; }
        [data-testid='stFileUploader'] button:hover { background:#23968f !important; }
        div[data-testid='stDataFrame'] { border:1px solid rgba(167,208,205,.14); border-radius:10px; overflow:hidden; }
        [data-testid='stSidebar'] { background:#0b151d; }
        [data-testid='stSidebarNav'] a,
        [data-testid='stSidebarNav'] a p { color:#ffffff !important; }
        [data-testid='stSidebarNav'] a:hover { background:rgba(139,212,202,.16); color:#ffffff !important; }
        [data-testid='stWidgetLabel'] p,
        [data-testid='stWidgetLabel'] label,
        .stSelectbox label,
        .stTextInput label { color:#ffffff !important; }
        @media (max-width: 768px) {
            .block-container { padding:1rem 1rem 2.5rem; }
            .hero { padding:1.5rem; }
            .hero h1 { font-size:1.8rem; }
            .status { float:none; display:inline-block; margin:.9rem 0 0; }
            .page-header h1 { font-size:1.7rem; }
            .stHorizontalBlock { gap: .75rem; }
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.selectbox(
            'Display timezone',
            list(TIMEZONE_OPTIONS),
            key='display_timezone_label',
            help='Choose how timestamps are displayed. Database values remain in UTC.',
        )


def render_page_header(title: str, description: str, eyebrow: str = 'Person ReID Control Center') -> None:
    """Render the shared title block used by secondary dashboard pages."""
    st.markdown(
        f"<div class='page-header'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def display_timestamp(value: datetime | None) -> str:
    """Convert a naive UTC database timestamp into the selected display timezone."""
    if value is None:
        return '-'
    utc_value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    timezone_name = TIMEZONE_OPTIONS.get(st.session_state.get('display_timezone_label', 'India (IST)'), 'Asia/Kolkata')
    display_zone = ZoneInfo(timezone_name)
    local_value = utc_value.astimezone(display_zone)
    return local_value.strftime(f'%d %b %Y, %I:%M:%S %p {local_value.tzname()}')


def render_sidebar():
    """Render manual sidebar links for pages that use this helper."""
    with st.sidebar:
        st.markdown('### 👥 Person ReID')
        st.caption('Control Center')
        st.page_link('streamlit_app.py', label='Home', icon='⌂')
        st.page_link('/analytics', label='Analytics', icon='📊')
        st.page_link('/admin', label='Admin', icon='⚙️')
        st.divider()
        st.caption('Presence intelligence')


def connection():
    """Open a short-lived MySQL connection after validating required credentials."""
    settings = Settings()
    if not settings.mysql_host or not settings.mysql_user or not settings.mysql_database:
        raise RuntimeError('MySQL is not configured. Add the REID_MYSQL_* values to Streamlit secrets.')
    return pymysql.connect(host=settings.mysql_host, port=settings.mysql_port, user=settings.mysql_user,
                           password=settings.mysql_password, database=settings.mysql_database,
                           cursorclass=pymysql.cursors.DictCursor, connect_timeout=5, read_timeout=10,
                           autocommit=True, init_command="SET time_zone='+00:00'")


@st.cache_data(ttl=3)
def load_runs() -> list[dict]:
    """Fetch recent non-archived runs; cache briefly to avoid repeated database queries."""
    try:
        db = connection()
    except (RuntimeError, pymysql.MySQLError) as error:
        st.warning(str(error))
        return []
    try:
        with db.cursor() as cursor:
            cursor.execute('SELECT id, source, started_at, completed_at FROM reid_runs WHERE deleted_at IS NULL ORDER BY id DESC LIMIT 100')
            return list(cursor.fetchall())
    finally:
        db.close()


@st.cache_data(ttl=3)
def load_run_details(run_id: int) -> tuple[list[dict], list[dict]]:
    """Fetch the summary rows and chronologically ordered IN/OUT events for one run."""
    try:
        db = connection()
    except (RuntimeError, pymysql.MySQLError) as error:
        st.warning(str(error))
        return [], []
    try:
        with db.cursor() as cursor:
            cursor.execute('SELECT person_name, total_seconds, entries_count, exits_count FROM person_presence WHERE run_id=%s AND deleted_at IS NULL ORDER BY person_name', (run_id,))
            summary = list(cursor.fetchall())
            cursor.execute('SELECT person_name, event_type, occurred_at FROM person_presence_events WHERE run_id=%s AND deleted_at IS NULL ORDER BY occurred_at', (run_id,))
            return summary, list(cursor.fetchall())
    finally:
        db.close()


def delete_run(run_id: int) -> None:
    """Soft-delete a run and its child rows so audit triggers can archive them."""
    db = connection()
    try:
        with db.cursor() as cursor:
            cursor.execute('UPDATE person_presence_events SET deleted_at=UTC_TIMESTAMP() WHERE run_id=%s AND deleted_at IS NULL', (run_id,))
            cursor.execute('UPDATE person_presence SET deleted_at=UTC_TIMESTAMP() WHERE run_id=%s AND deleted_at IS NULL', (run_id,))
            cursor.execute('UPDATE reid_runs SET deleted_at=UTC_TIMESTAMP() WHERE id=%s AND deleted_at IS NULL', (run_id,))
    finally:
        db.close()


def admin_authenticated() -> bool:
    """Show sign-in UI and remember a successful login in this browser session."""
    password = dotenv_values(Path(__file__).resolve().parents[2] / '.env').get('REID_ADMIN_PASSWORD') or os.getenv('REID_ADMIN_PASSWORD', '')
    if not password:
        try:
            password = str(st.secrets.get('REID_ADMIN_PASSWORD', ''))
        except Exception:
            password = ''
    if not password:
        st.warning('Set `REID_ADMIN_PASSWORD` in `.env` to enable administration.')
        return False
    if st.session_state.get('admin_authenticated'):
        return True
    entered = st.text_input('Admin password', type='password')
    if st.button('Sign in'):
        if entered == password:
            st.session_state.admin_authenticated = True
            st.rerun()
        st.error('Incorrect admin password.')
    return False
