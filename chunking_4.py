import os
import re
import uuid
import pickle
from urllib.parse import unquote
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================
# FIX: always save chunks.pkl next to this
# script regardless of launch directory
# =========================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =========================================
# CONFIGURATION — update path if needed
# =========================================
MD_DIR = "D:/Deltacubes/python_programming/markdowns"

# Sections with zero scientific value — dropped entirely
NOISE_HEADINGS = {
    "edited by", "reviewed by", "correspondence",
    "specialty section", "citation", "author contributions",
    "acknowledgments", "acknowledgement", "acknowledgements",
    "references", "data availability statement",
    "conflict of interest", "funding",
    "supplementary material", "abbreviations"
}

# =========================================
# LOAD MARKDOWN FILES
# Decodes URL-encoded filenames, skips duplicates
# =========================================
def load_markdown_files():
    documents   = []
    seen_decoded = set()

    for file in sorted(os.listdir(MD_DIR)):
        if not file.endswith(".md"):
            continue

        decoded_name = unquote(file)

        if decoded_name in seen_decoded:
            print(f"  [SKIP DUPLICATE] {file}")
            continue

        seen_decoded.add(decoded_name)
        path = os.path.join(MD_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({"file": decoded_name, "text": text})

    return documents


# =========================================
# TEXT SPLITTER
# chunk_size=512, overlap=80
# separators cover both \n\n paragraphs and
# single-\n lines (common in Docling output)
# =========================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "]
)


# =========================================
# NOISE FILTERS  (text chunks only)
# =========================================
def is_valid_text_chunk(text):
    stripped = text.strip()

    # Minimum 80 chars — keeps mid-sentence split fragments
    # (e.g. "al., 1996). In soybean, qPHD1 was discovered...")
    # that carry real scientific content after the overlap join
    if len(stripped) < 80:
        return False

    # Only image/table placeholders, no real content
    cleaned = re.sub(r'<!--.*?-->', '', stripped).strip()
    if len(cleaned) < 80:
        return False

    # Pure numeric rows leaked from tables
    if re.match(r'^[\d\s\.\,\|\-\+\(\)%±×–—]+$', stripped):
        return False

    # Author affiliation blocks  e.g. "a ICAR-National Bureau..."
    if re.match(
        r'^[a-z]\s+(ICAR|Division|Department|University|Institute|'
        r'School|College|Center|Centre|Laboratory)',
        stripped
    ):
        return False

    # Dense bibliography / reference list blocks
    if stripped.lower().count('doi') > 2 and len(stripped) < 600:
        return False

    # Journal homepage / bare URL lines
    if re.match(r'^(journal homepage|available online|www\.)', stripped.lower()):
        return False

    return True


# =========================================
# PARSE DOCUMENT INTO TYPED BLOCKS
#
# Each block is one of:
#   {"type": "text",  "section": str, "content": str}
#   {"type": "table", "section": str, "title": str, "content": str}
#
# Rules:
#   - Noise headings + their content → dropped
#   - TABLE N | caption lines       → table title
#   - Lines starting with |         → table body rows
#   - Everything else               → text
#   - <!-- image --> placeholders   → stripped
# =========================================
def parse_document(text):
    blocks              = []
    current_section     = "Introduction"
    current_text_lines  = []
    current_table_title = None
    current_table_lines = []
    in_noise_section    = False

    text  = text.replace('\r\n', '\n').replace('\r', '\n')
    text  = re.sub(r'<!--\s*image\s*-->', '', text)
    lines = text.split('\n')

    def flush_text():
        if current_text_lines:
            block_text = '\n'.join(current_text_lines).strip()
            if block_text:
                blocks.append({
                    "type":    "text",
                    "section": current_section,
                    "content": block_text
                })
        current_text_lines.clear()

    def flush_table():
        if not current_table_lines:
            return
        table_body = '\n'.join(current_table_lines).strip()
        real_rows  = [
            l for l in current_table_lines
            if l.startswith('|') and '---' not in l and len(l.strip()) > 3
        ]
        # Skip near-empty tables (lone | from Docling parse failures)
        if len(real_rows) < 2:
            current_table_lines.clear()
            return
        if table_body:
            if current_table_title:
                title = current_table_title
            else:
                # Derive title from column headers when no TABLE N | caption
                header_row = real_rows[0]
                cols       = [c.strip() for c in header_row.strip('|').split('|') if c.strip()]
                cols_str   = ", ".join(cols[:5])
                title      = f"Table ({current_section}): {cols_str}"
            blocks.append({
                "type":    "table",
                "section": current_section,
                "title":   title,
                "content": table_body
            })
        current_table_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Section heading ───────────────────────────────────────
        if re.match(r'^#{1,3}\s+', line):
            flush_text()
            flush_table()
            heading_text  = re.sub(r'^#+\s*', '', line).strip()
            heading_lower = heading_text.lower().lstrip('*').strip()
            in_noise_section = any(
                heading_lower.startswith(n) for n in NOISE_HEADINGS
            )
            if not in_noise_section:
                current_section = heading_text
            i += 1
            continue

        # ── Skip noise sections entirely ──────────────────────────
        if in_noise_section:
            i += 1
            continue

        # ── TABLE N | caption line ────────────────────────────────
        if re.match(r'^TABLE\s+\d+\s*\|', line, re.IGNORECASE):
            flush_text()
            flush_table()
            current_table_title = line.strip()
            current_table_lines = []
            i += 1
            continue

        # ── Markdown table row ────────────────────────────────────
        if line.startswith('|'):
            if not current_table_lines and not current_table_title:
                flush_text()
                current_table_title = None  # will use header-derived title
            current_table_lines.append(line)
            i += 1
            continue

        # ── End of table block ────────────────────────────────────
        if current_table_lines:
            flush_table()
            current_table_title = None

        # ── Regular text line ─────────────────────────────────────
        current_text_lines.append(line)
        i += 1

    flush_text()
    flush_table()
    return blocks


# =========================================
# CHUNK A SINGLE DOCUMENT
#
# Text   → RecursiveCharacterTextSplitter
#           with cross-section overlap bridging
# Tables → one atomic chunk (never split mid-row)
#           large tables split by row, header repeated
# =========================================
def chunk_document(text, source):
    final_chunks = []
    blocks       = parse_document(text)
    prev_tail    = ""  # last 80 chars of previous text block for bridging

    for block in blocks:

        # ── TABLE ─────────────────────────────────────────────────
        if block["type"] == "table":
            title      = block["title"]
            content    = block["content"]
            table_text = f"{title}\n\n{content}"

            if len(table_text) <= 3000:
                final_chunks.append({
                    "id":      str(uuid.uuid4()),
                    "type":    "table",
                    "text":    table_text,
                    "section": block["section"],
                    "title":   title,
                    "source":  source
                })
            else:
                # Split large table by rows, repeat header in every part
                rows         = content.split('\n')
                header_block = '\n'.join(rows[:2])
                data_rows    = rows[2:]
                batch        = []
                part_num     = 1

                for row in data_rows:
                    batch.append(row)
                    candidate = (
                        f"{title} (part {part_num})\n\n"
                        f"{header_block}\n" + '\n'.join(batch)
                    )
                    if len(candidate) > 2800:
                        final_chunks.append({
                            "id":      str(uuid.uuid4()),
                            "type":    "table",
                            "text":    candidate,
                            "section": block["section"],
                            "title":   f"{title} (part {part_num})",
                            "source":  source
                        })
                        batch    = []
                        part_num += 1

                if batch:
                    final_chunks.append({
                        "id":      str(uuid.uuid4()),
                        "type":    "table",
                        "text":    (
                            f"{title} (part {part_num})\n\n"
                            f"{header_block}\n" + '\n'.join(batch)
                        ),
                        "section": block["section"],
                        "title":   f"{title} (part {part_num})",
                        "source":  source
                    })

            prev_tail = ""
            continue

        # ── TEXT ──────────────────────────────────────────────────
        raw = block["content"].strip()
        if not raw:
            continue

        # Cross-section bridging: prepend last 80 chars of previous block
        bridged    = (prev_tail + "\n\n" + raw).strip() if prev_tail else raw
        sub_chunks = splitter.split_text(bridged)

        for chunk in sub_chunks:
            chunk = chunk.strip()
            if is_valid_text_chunk(chunk):
                final_chunks.append({
                    "id":      str(uuid.uuid4()),
                    "type":    "text",
                    "text":    chunk,
                    "section": block["section"],
                    "title":   chunk.split('\n')[0][:100],
                    "source":  source
                })

        prev_tail = raw[-80:].strip() if len(raw) > 80 else raw

    return final_chunks


# =========================================
# MAIN PIPELINE
# =========================================
def generate_all_chunks():
    all_chunks  = []
    docs        = load_markdown_files()
    print(f"Found {len(docs)} unique markdown files\n")

    text_count  = 0
    table_count = 0

    for doc in docs:
        print(f"Processing: {doc['file']}")
        chunks = chunk_document(doc["text"], doc["file"])
        all_chunks.extend(chunks)
        text_count  += sum(1 for c in chunks if c["type"] == "text")
        table_count += sum(1 for c in chunks if c["type"] == "table")

    print(f"\n  Text chunks  : {text_count}")
    print(f"  Table chunks : {table_count}")
    print(f"  Total        : {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = generate_all_chunks()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"\nChunks saved to {out_path}")

    # ── Sanity check ──────────────────────────────────────────────
    print("\n=== Sanity Check ===")
    lengths = [len(c["text"]) for c in chunks]
    print(f"Min length     : {min(lengths)}")
    print(f"Max length     : {max(lengths)}")
    print(f"Mean length    : {sum(lengths) // len(lengths)}")
    print(f"Unique sources : {len(set(c['source'] for c in chunks))}")
    print(f"Double-heading : {len([c for c in chunks if '## ##' in c['text'][:60]])}")
    print(f"Image-only     : {len([c for c in chunks if '<!-- image -->' in c['text'] and len(c['text'].replace('<!-- image -->', '').strip()) < 80])}")

    # qPHD1 verification — confirms mid-paragraph splits are kept
    qphd1 = [c for c in chunks if "qPHD1" in c["text"]]
    print(f"qPHD1 chunks   : {len(qphd1)}  (should be 2)")
    for c in qphd1:
        print(f"  [{c['source'][:50]}] section={c['section']}")

    # Table sample
    table_chunks = [c for c in chunks if c["type"] == "table"]
    print(f"\nTable chunks ({len(table_chunks)} total) — first 5:")
    for c in table_chunks[:5]:
        print(f"  [{c['source'][:35]}]  {c['title'][:70]}")