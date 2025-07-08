import os
import pymupdf
from datetime import datetime

class FileParser:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def readDirectory(self):
        all_chunks = []
        for fname in os.listdir(self.dir_path):
            fpath = os.path.join(self.dir_path, fname)
            if os.path.isfile(fpath) and (fname.endswith('.txt') or fname.endswith('.pdf')):
                try:
                    chunks = self.recognize_and_read(fpath)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Failed to parse file: {fname}, reason: {e}")
        return all_chunks

    def recognize_and_read(self, fpath) -> list[dict]:
        if not os.path.exists(fpath):
            raise FileNotFoundError("File does not exist")
        stat = os.stat(fpath)
        meta = {
            "filename": os.path.basename(fpath),
            "filepath": os.path.abspath(fpath),
            "filesize": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "directory": os.path.dirname(os.path.abspath(fpath))
        }
        ext = os.path.splitext(fpath)[1]
        if ext == '.txt':
            return self.read_txt(fpath, meta)
        elif ext == '.pdf':
            return self.read_pdf(fpath, meta)
        else:
            raise ValueError("Unsupported file type, only .txt and .pdf are supported")

    def read_txt(self, fpath, meta) -> list[dict]:
        chunk_list = []
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        paragraphs = content.strip().split('\n\n')
        for p in paragraphs:
            clean_paragraph = p.replace('\n', ' ').strip()
            if clean_paragraph:
                chunk = dict(meta)
                chunk["text"] = clean_paragraph
                chunk_list.append(chunk)
        return chunk_list

    def read_pdf(self, fpath, meta) -> list[dict]:
        doc = pymupdf.open(fpath)
        chunk_list = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                lines = []
                for line in block["lines"]:
                    span_texts = [span["text"] for span in line["spans"]]
                    lines.append("".join(span_texts))
                block_text = " ".join(lines).strip()
                if block_text:
                    chunk = dict(meta)
                    chunk["text"] = block_text
                    chunk_list.append(chunk)
        return chunk_list
