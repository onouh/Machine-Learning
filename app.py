"""Streamlit presentation layer for hotel-booking analysis."""

from pathlib import Path

import streamlit as st

from hotel_booking.services.analysis import analyze_dataset

st.set_page_config(page_title="Hotel Booking ML", page_icon="🏨", layout="wide")
st.title("City Hotel Booking ML")
st.caption("Analysis UI scaffold — modeling and inference workflows are tracked in TASK.md.")

with st.sidebar:
    data_path = st.text_input("H2 CSV path", value="data/raw/H2.csv")

path = Path(data_path)
if not path.is_file():
    st.info("Place the City Hotel dataset at data/raw/H2.csv, or enter its local path above.")
    st.markdown("See [data/README.md](data/README.md) for data setup guidance.")
else:
    try:
        frame, profile, issues = analyze_dataset(path)
    except (OSError, ValueError, UnicodeError) as exc:
        st.error(f"Could not load dataset: {exc}")
    else:
        first, second, third = st.columns(3)
        first.metric("Rows", f"{profile['rows']:,}")
        second.metric("Columns", profile["columns"])
        third.metric("Exact duplicate rows", f"{profile['exact_duplicate_rows']:,}")

        if issues:
            st.warning("Schema checks found issues:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("H2 schema and IsCanceled labels passed the current checks.")

        if profile["is_canceled_counts"]:
            st.subheader("Cancellation label counts")
            st.bar_chart(profile["is_canceled_counts"])

        st.subheader("Column profile")
        st.dataframe(profile["column_profile"], use_container_width=True, hide_index=True)
        with st.expander("Preview rows"):
            st.dataframe(frame.head(20), use_container_width=True)
