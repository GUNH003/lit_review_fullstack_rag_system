import tqdm
import json
import fitz  # pymupdf

BASE_PATH = "./rag/"

def page_text_handler_linux(text: str) -> str:
    newline_count = 0
    index = -1
    for i, char in enumerate(text):
        if char == '\n':
            newline_count += 1
            if newline_count == 5:
                index = i
                break
    if index != -1:
        return text[index + 1:]
    return text

def page_text_handler_distributed_computing(text: str) -> str:
    newline_count = 0
    index = -1
    for i, char in enumerate(text):
        if char == '\n':
            newline_count += 1
            if newline_count == 2:
                index = i
                break
    if index != -1:
        return text[index + 1:]
    return text

def page_text_handler_distributed_system(text: str) -> str:
    newline_count = 0
    index = -1
    for i, char in enumerate(text):
        if char == '\n':
            newline_count += 1
            if newline_count == 2:
                index = i
                break
    if index != -1:
        return text[index + 1:]
    return text

def page_text_handler_constructivism(text: str) -> str:
    newline_count = 0
    index = -1
    for i, char in enumerate(text):
        if char == '\n':
            newline_count += 1
            if newline_count == 6:
                index = i
                break
    if index != -1:
        return text[index + 1:]
    return text

def page_text_handler_globalization(text: str) -> str:
    newline_count = 0
    index = -1
    for i, char in enumerate(text):
        if char == '\n':
            newline_count += 1
            if newline_count == 1:
                index = i
                break
    if index != -1:
        return text[index + 1:]
    return text

def process_file(
    file_path: str,
    file_title: str,
    file_author: str,
    document_name: str,
    start: int,
    end: int,
    page_text_handler,
    snippet_size: int = 30,
    snippet_overlap: int = 15
):
    # Load PDF
    doc = fitz.open(file_path)
    # Parse pages 
    pages: list[dict] = []
    page_number = 1
    with tqdm.tqdm(range(start, end), desc=f"Parsing pages for {document_name}") as bar:
        for page_num in bar:
            page = doc[page_num]
            text = page.get_text("text")
            pages.append({
                "title": file_title,
                "author": file_author,
                "page": page_number,
                "content": page_text_handler(text)
            })
            page_number += 1
    print(f"Total number of pages processed for {document_name}: {len(pages)}")
    with open(f"{BASE_PATH}/output/pages_{document_name}.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # Parse lines
    all_lines: list[dict] = []
    with tqdm.tqdm(pages, desc=f"Parsing lines for {document_name}") as bar:
        for p in bar:
            lines = p["content"].splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    all_lines.append({
                        "title": p["title"],
                        "author": p["author"],
                        "page": p["page"],
                        "line": i + 1,
                        "content": line
                    })
    print(f"Total number of lines processed for {document_name}: {len(all_lines)}")
    with open(f"{BASE_PATH}/output/lines_{document_name}.jsonl", "w", encoding="utf-8") as f:
        for line_data in all_lines:
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")
    # Create snippets
    snippets: list[dict] = []
    step = snippet_size - snippet_overlap
    indices = range(0, len(all_lines), step)
    with tqdm.tqdm(indices, desc=f"Creating snippets for {document_name}") as bar:
        for i in bar:
            chunk = all_lines[i:i + snippet_size]
            if chunk:  # Ensure chunk is not empty
                content = "\n".join([line["content"] for line in chunk])
                snippets.append({
                    "title": file_title,
                    "author": file_author,
                    "page": chunk[0]["page"],
                    "line": chunk[0]["line"],
                    "content": content
                })
    print(f"Total number of snippets processed for {document_name}: {len(snippets)}")
    with open(f"{BASE_PATH}/output/snippets_{document_name}.jsonl", "w", encoding="utf-8") as f:
        for snippet_data in snippets:
            f.write(json.dumps(snippet_data, ensure_ascii=False) + "\n")
    return snippets

def main():
    # NOTE: Modify the file names as needed
    documents = [
        {
            "file_path": f"{BASE_PATH}/input/linux.pdf",    # Understanding the Linux Kernel, Third Edition
            "file_title": "Understanding the Linux Kernel, Third Edition",
            "file_author": "Daniel P. Bovet and Marco Cesati",
            "document_name": "linux",
            "start": 18,
            "end": 852,
            "page_text_handler": page_text_handler_linux,
            "snippet_size": 30,
            "snippet_overlap": 20
        },
        {
            "file_path": f"{BASE_PATH}/input/distSys.pdf",   # Distributed Systems: Principles and Paradigms, Second Edition
            "file_title": "Distributed Systems: Principles and Paradigms, Second Edition",
            "file_author": "George Coulouris, Jean Dollimore, Tim Kindberg, Gordon Blair",
            "document_name": "distributed_system",
            "start": 18,
            "end": 983,
            "page_text_handler": page_text_handler_distributed_system,
            "snippet_size": 30,
            "snippet_overlap": 20
        },
        {
            "file_path": f"{BASE_PATH}/input/distComp.pdf", # Distributed Computing Principles, Algorithms, and Systems
            "file_title": "Distributed Computing: Principles, Algorithms, and Systems",
            "file_author": "Ajay D. Kshemkalyani, Mukesh Singhal",
            "document_name": "distributed_computing",
            "start": 20,
            "end": 730,
            "page_text_handler": page_text_handler_distributed_computing,
            "snippet_size": 30,
            "snippet_overlap": 20
        },
        {
            "file_path": f"{BASE_PATH}/input/constructivism.pdf",   # Constructivism in Practice and Theory: Toward a Better Understanding
            "file_title": "Constructivism in Practice and Theory: Toward a Better Understanding",
            "file_author": "James. M. Applefield, Richard Huber & Mahnaz Moallem",
            "document_name": "constructivism",
            "start": 1,
            "end": 34,
            "page_text_handler": page_text_handler_constructivism,
            "snippet_size": 30,
            "snippet_overlap": 20
        },
        {
            "file_path": f"{BASE_PATH}/input/globalization.pdf",    # Globalization: Past, Present, Future
            "file_title": "Globalization: Past, Present, Future",
            "file_author": "Manfred B. Steger, Roland Benedikter, Harald Pechlaner, and Ingrid Kofler",
            "document_name": "globalization",
            "start": 13,
            "end": 345,
            "page_text_handler": page_text_handler_globalization,
            "snippet_size": 30,
            "snippet_overlap": 20
        }
    ]
    total_snippets: list[dict] = []
    for doc_params in documents:
        total_snippets.extend(process_file(**doc_params))
    with open(f"{BASE_PATH}/output/total_snippets.jsonl", "w", encoding="utf-8") as f:
        for snippet_data in total_snippets:
            f.write(json.dumps(snippet_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
