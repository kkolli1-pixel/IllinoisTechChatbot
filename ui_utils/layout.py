from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


def load_css(css_path: Path) -> None:
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_title_block(logo_position: str = "center") -> None:
    logo_data_uri = ""
    logo_path = (
        Path(__file__).resolve().parents[1]
        / "Illinois Tech Wordmark Alone"
        / "ILTECH_wht_horiz.png"
    )
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_data_uri = f"data:image/png;base64,{encoded}"

    st.markdown(
        f"""
        <div class="title-wrap logo-{logo_position}">
            <div class="brand-strip">
                {"<img src='" + logo_data_uri + "' alt='Illinois Tech logo' class='iit-logo' />" if logo_data_uri else "<span class='iit-wordmark-text'>ILLINOIS TECH</span>"}
            </div>
            <h1>CHATBOT COMPARISON</h1>
            <p>Ask about tuition, academic calendar, and directory information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_actions(disabled: bool = False) -> tuple[bool, bool]:
    # ✅ Removed spacer div (was causing extra vertical gap)
    st.markdown(
        """
        <div class="footer-actions-text">
            <div class="footer-actions-row">
        """,
        unsafe_allow_html=True,
    )

    # ✅ Fixed spacing + centering
    col_left, col_compare, col_sep, col_feedback, col_right = st.columns(
        [1, 1.2, 0.2, 1.2, 1],  # balanced layout
        gap="medium"
    )

    with col_compare:
        compare_clicked = st.button(
            "COMPARE",
            key="footer_compare",
            disabled=disabled,
            type="tertiary",
            use_container_width=True,  # helps alignment
        )

    with col_sep:
        st.markdown(
            '<div style="text-align:center; font-size:18px; opacity:0.6;">•</div>',
            unsafe_allow_html=True
        )

    with col_feedback:
        feedback_clicked = st.button(
            "GIVE FEEDBACK",
            key="footer_feedback",
            disabled=disabled,
            type="tertiary",
            use_container_width=True,  # helps alignment
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

    return compare_clicked, feedback_clicked