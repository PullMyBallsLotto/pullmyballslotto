# file: pages/Terms.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import app

st.title("Terms of Use — PullMyBallsLotto")
st.caption("Educational only. Not affiliated. No refunds except where required by law.")

# Main content
st.markdown(app.legal_markdown())

# Downloads
md_bytes = app.legal_markdown().encode("utf-8")
html_bytes = (
    "<!doctype html><html><head><meta charset='utf-8'><title>PullMyBallsLotto — Terms</title>"
    "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "max-width:720px;margin:40px auto;line-height:1.55;color:#222} h1,h2{margin:0 0 .5rem} pre{white-space:pre-wrap}</style></head>"
    "<body><h1>Terms of Use</h1><pre>"
    + app.legal_markdown().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    + "</pre></body></html>"
).encode("utf-8")
fname = f"pullmyballslotto_terms_{app.TERMS_VERSION}"

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Terms (.md)", data=md_bytes, file_name=f"{fname}.md", mime="text/markdown")
with col2:
    st.download_button("Download Terms (.html)", data=html_bytes, file_name=f"{fname}.html", mime="text/html")

st.markdown("> Tip: Use your browser’s **Print** (Ctrl/Cmd+P) to save this page as PDF.")
