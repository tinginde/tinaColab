from dataclasses import dataclass
import re
from typing import List, Dict

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]

def read_split_md(md_doc: str) -> List[Document]:
    """Split markdown text into Document objects by headers."""
    docs: List[Document] = []
    lines = md_doc.splitlines()
    current_lines = []
    current_meta = {"header": "", "level": 0}
    header_re = re.compile(r"^(#+)\s*(.*)")

    for line in lines:
        match = header_re.match(line)
        if match:
            if current_lines:
                docs.append(Document(page_content="\n".join(current_lines).strip(), metadata=current_meta))
            level = len(match.group(1))
            header = match.group(2).strip()
            current_meta = {"header": header, "level": level}
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        docs.append(Document(page_content="\n".join(current_lines).strip(), metadata=current_meta))
    return docs
