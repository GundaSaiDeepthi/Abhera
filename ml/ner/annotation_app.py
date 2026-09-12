import json
from pathlib import Path
import sys
import streamlit as st

# Path configuration
def get_project_root() -> Path:
    """Dynamically resolves the project root directory."""
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent.parent,
        current_file.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "ml").exists():
            return candidate
    return current_file.parent.parent.parent

project_root = get_project_root()
annotation_dir = project_root / "ml" / "ner" / "annotation"
sample_file = annotation_dir / "annotation_sample.jsonl"
draft_file = annotation_dir / "annotated_ner_draft.jsonl"
queue_file = annotation_dir / "ner_human_review_queue.jsonl"
reviewed_file = annotation_dir / "annotated_ner_reviewed.jsonl"
review_progress_file = annotation_dir / "review_progress.json"
report_file = annotation_dir / "reviewed_ner_report.txt"

sys.path.insert(0, str(annotation_dir))
try:
    from validate_reviewed import validate_reviewed_dataset
except ImportError:
    validate_reviewed_dataset = None

APPROVED_LABELS = {
    "PERP_REL": "Perpetrator Relation (e.g., husband, boss, stranger)",
    "LOCATION": "Harassment Location (e.g., office, home, bus stand)",
    "TIME_FREQ": "Time or Frequency (e.g., every morning, daily)",
    "PLATFORM": "Digital Platform (e.g., social media, email, fake profile)",
    "EVIDENCE": "Physical Harm or Evidence (e.g., belt, slapped, locked inside)",
    "LAW_SEC": "Statutory Law Section (e.g., 354 IPC, POSH Act)",
}


def load_sample_records():
    if not sample_file.exists():
        st.error(f"Sample file not found at: {sample_file}. Run create_annotation_sample.py first.")
        return []
    records = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_draft_map():
    draft_map = {}
    if draft_file.exists():
        with open(draft_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    draft_map[rec["id"]] = rec
    return draft_map


def load_review_queue():
    queue_entries = []
    if queue_file.exists():
        with open(queue_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queue_entries.append(json.loads(line))
    return queue_entries


def load_reviewed_map():
    reviewed_map = {}
    if reviewed_file.exists():
        with open(reviewed_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    reviewed_map[rec["id"]] = rec
    return reviewed_map


def save_reviewed_map(reviewed_map):
    annotation_dir.mkdir(parents=True, exist_ok=True)
    with open(reviewed_file, "w", encoding="utf-8") as f:
        for rec_id, rec in reviewed_map.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    if validate_reviewed_dataset:
        validate_reviewed_dataset()


def main():
    st.set_page_config(
        page_title="Women's Safety Legal AI - Human NER Review Interface",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("⚖️ Women's Safety Legal AI — Human NER Review & Audit Interface")
    st.caption("Human-in-the-loop review workflow with automatic quality-triage support. Zero-fabrication enforcement.")

    sample_records = load_sample_records()
    if not sample_records:
        st.warning("No sample records available. Run python ml/ner/create_annotation_sample.py first.")
        return

    draft_map = load_draft_map()
    reviewed_map = load_reviewed_map()
    queue_entries = load_review_queue()

    total_cnt = len(sample_records)
    reviewed_cnt = len(reviewed_map)
    remaining_cnt = max(0, total_cnt - reviewed_cnt)
    pct_complete = (reviewed_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

    # Mode selection in sidebar
    app_mode = st.sidebar.radio(
        "Select Review Mode:",
        options=["Standard Record-by-Record Review", "Review Flagged Candidates Queue"],
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Human Review Progress")
    st.sidebar.progress(reviewed_cnt / total_cnt if total_cnt > 0 else 0.0)
    st.sidebar.metric("Reviewed Records", f"{reviewed_cnt} / {total_cnt} ({pct_complete:.1f}%)")
    st.sidebar.metric("Remaining Records", f"{remaining_cnt}")
    st.sidebar.metric("Flagged Candidates in Queue", f"{len(queue_entries)}")

    # -------------------------------------------------------------
    # MODE 1: Standard Record-by-Record Review
    # -------------------------------------------------------------
    if app_mode == "Standard Record-by-Record Review":
        # Find first unreviewed record index
        first_unreviewed_idx = 0
        for idx, r in enumerate(sample_records):
            if r["id"] not in reviewed_map:
                first_unreviewed_idx = idx
                break

        if st.sidebar.button("Jump to Next Unreviewed Record"):
            st.session_state["nav_idx"] = first_unreviewed_idx + 1
            st.rerun()

        current_idx_val = st.session_state.get("nav_idx", first_unreviewed_idx + 1)
        selected_idx = st.sidebar.number_input(
            "Navigate Record Index (1 to 400)",
            min_value=1,
            max_value=total_cnt,
            value=current_idx_val,
            step=1,
            key="record_nav_input",
        ) - 1

        st.session_state["nav_idx"] = selected_idx + 1
        curr_record = sample_records[selected_idx]
        rec_id = curr_record["id"]
        text = curr_record["text"]

        if rec_id in reviewed_map:
            status_badge = "✅ REVIEWED (HUMAN APPROVED)"
        elif rec_id in draft_map:
            status_badge = "📝 CANDIDATE DRAFT (PENDING HUMAN REVIEW)"
        else:
            status_badge = "⏳ UNANNOTATED"

        st.markdown(f"### Record ID: `{rec_id}` &nbsp; | &nbsp; Status: **{status_badge}**")
        st.markdown("#### Incident Narrative Text:")
        st.info(f"\"{text}\"")

        if rec_id in reviewed_map:
            active_entities = [dict(e) for e in reviewed_map[rec_id].get("entities", [])]
            source_notice = "Showing human-reviewed annotations."
        elif rec_id in draft_map:
            active_entities = [dict(e) for e in draft_map[rec_id].get("entities", [])]
            source_notice = "Showing pre-populated candidate draft annotations."
        else:
            active_entities = []
            source_notice = "No candidate annotations found."

        st.caption(f"Source: {source_notice}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Add / Edit Entity Span")
            selected_substring = st.text_input("Selected Substring / Text Span:", key=f"input_sub_{rec_id}")
            selected_label = st.selectbox(
                "Select Entity Label:",
                options=list(APPROVED_LABELS.keys()),
                format_func=lambda x: f"{x} - {APPROVED_LABELS[x]}",
                key=f"select_lbl_{rec_id}",
            )

            if st.button("Add Entity Span", type="secondary"):
                clean_sub = selected_substring.strip()
                if not clean_sub:
                    st.error("Please enter a non-empty text span.")
                elif clean_sub not in text:
                    st.error(f"Span '{clean_sub}' not found in narrative.")
                else:
                    start_idx = text.find(clean_sub)
                    end_idx = start_idx + len(clean_sub)

                    overlap = False
                    for existing in active_entities:
                        e_start, e_end = existing["start"], existing["end"]
                        if not (end_idx <= e_start or start_idx >= e_end):
                            overlap = True
                            st.error(f"Span overlaps with existing entity '{existing['text']}' [{e_start}:{e_end}].")
                            break

                    if not overlap:
                        new_ent = {"start": start_idx, "end": end_idx, "label": selected_label, "text": clean_sub}
                        active_entities.append(new_ent)
                        reviewed_map[rec_id] = {"id": rec_id, "text": text, "entities": active_entities}
                        save_reviewed_map(reviewed_map)
                        st.success(f"Added span '{clean_sub}' [{start_idx}:{end_idx}] as '{selected_label}'.")
                        st.rerun()

        with col2:
            st.markdown("#### Entity Spans List")
            if not active_entities:
                st.info("No entity spans assigned to this narrative.")
            else:
                for e_idx, ent in enumerate(active_entities):
                    c_span, c_btn = st.columns([3, 1])
                    c_span.write(f"**{ent['label']}**: `{ent['text']}` `[{ent['start']}:{ent['end']}]`")
                    if c_btn.button("Delete Span", key=f"del_{rec_id}_{e_idx}"):
                        active_entities.pop(e_idx)
                        reviewed_map[rec_id] = {"id": rec_id, "text": text, "entities": active_entities}
                        save_reviewed_map(reviewed_map)
                        st.rerun()

            if st.button("Clear All Entity Spans (Set entities = [])", key=f"clear_{rec_id}"):
                reviewed_map[rec_id] = {"id": rec_id, "text": text, "entities": []}
                save_reviewed_map(reviewed_map)
                st.success("Cleared all entity spans for this record.")
                st.rerun()

        st.markdown("---")
        c_prev, c_save, c_next = st.columns([1, 2, 1])

        if c_prev.button("⬅️ Previous Record"):
            if selected_idx > 0:
                st.session_state["nav_idx"] = selected_idx
                st.rerun()

        if c_save.button("✅ Confirm & Save Reviewed Record", type="primary"):
            reviewed_map[rec_id] = {"id": rec_id, "text": text, "entities": active_entities}
            save_reviewed_map(reviewed_map)
            st.success(f"Record {rec_id} confirmed & saved to annotated_ner_reviewed.jsonl!")
            if selected_idx < total_cnt - 1:
                st.session_state["nav_idx"] = selected_idx + 2
                st.rerun()

        if c_next.button("Next Record ➡️"):
            if selected_idx < total_cnt - 1:
                st.session_state["nav_idx"] = selected_idx + 2
                st.rerun()

    # -------------------------------------------------------------
    # MODE 2: Review Flagged Candidates Queue
    # -------------------------------------------------------------
    elif app_mode == "Review Flagged Candidates Queue":
        if not queue_entries:
            st.warning("Human review queue file (ner_human_review_queue.jsonl) not found or empty. Run audit_ner_candidates.py first.")
            return

        q_idx = st.sidebar.number_input(
            "Navigate Queue Entry (1 to " + str(len(queue_entries)) + ")",
            min_value=1,
            max_value=len(queue_entries),
            value=1,
            step=1,
        ) - 1

        curr_q = queue_entries[q_idx]
        q_id = curr_q["id"]
        q_text = curr_q["text"]
        q_ent = curr_q["entity"]
        q_reason = curr_q["reason"]

        st.subheader(f"Flagged Candidate Queue Entry {q_idx+1} of {len(queue_entries)}")
        st.markdown(f"**Record ID**: `{q_id}`")
        st.markdown(f"**Reason for Review**: ⚠️ `{q_reason}`")
        st.info(f"Narrative Text:\n\"{q_text}\"")

        st.markdown(
            f"**Candidate Span**: `{q_ent['text']}` `[{q_ent['start']}:{q_ent['end']}]` &nbsp; | &nbsp; "
            f"**Proposed Label**: `{q_ent['label']}`"
        )

        col_q1, col_q2 = st.columns([1, 1])

        with col_q1:
            st.markdown("#### Candidate Triage Action")
            triage_action = st.radio(
                "Action for this Flagged Span:",
                options=["KEEP (Approve Candidate Span)", "CHANGE LABEL", "DELETE (Reject Span)"],
            )

            new_label = q_ent["label"]
            if triage_action == "CHANGE LABEL":
                new_label = st.selectbox(
                    "Select New Entity Label:",
                    options=list(APPROVED_LABELS.keys()),
                    index=list(APPROVED_LABELS.keys()).index(q_ent["label"]) if q_ent["label"] in APPROVED_LABELS else 0,
                )

            if st.button("Confirm Candidate Action", type="primary"):
                # Fetch or initialize record in reviewed map
                existing_rec = reviewed_map.get(q_id, draft_map.get(q_id, {"id": q_id, "text": q_text, "entities": []}))
                current_ents = [dict(e) for e in existing_rec.get("entities", [])]

                if triage_action == "DELETE (Reject Span)":
                    current_ents = [e for e in current_ents if not (e["start"] == q_ent["start"] and e["end"] == q_ent["end"] and e["label"] == q_ent["label"])]
                    st.success(f"Span '{q_ent['text']}' deleted.")
                elif triage_action == "CHANGE LABEL":
                    for e in current_ents:
                        if e["start"] == q_ent["start"] and e["end"] == q_ent["end"]:
                            e["label"] = new_label
                    st.success(f"Label updated to '{new_label}'.")
                else:
                    st.success(f"Candidate span '{q_ent['text']}' approved.")

                reviewed_map[q_id] = {
                    "id": q_id,
                    "text": q_text,
                    "entities": current_ents,
                }
                save_reviewed_map(reviewed_map)
                st.rerun()

    # Sidebar Validation Action
    st.sidebar.markdown("---")
    if st.sidebar.button("Run Full Validation Audit"):
        if validate_reviewed_dataset:
            passed, rpt = validate_reviewed_dataset()
            if passed:
                st.sidebar.success("Validation PASSED! All records structurally valid.")
            else:
                st.sidebar.warning("Validation audit executed. See reviewed_ner_report.txt for status.")


if __name__ == "__main__":
    main()
