import pymupdf

class PDFTextParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_blocks(self) -> list[str]:
        doc = pymupdf.open(self.pdf_path)
        blocks_list = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 以 dict 模式获取页面中的所有 block

            for block in page.get_text("dict")["blocks"]:
                # 仅处理文字类型的 block（type == 0）
                if block.get("type") != 0:
                    continue
                # 合并该 block 内所有行的 span 文本
                lines = []
                for line in block["lines"]:
                    span_texts = [span["text"] for span in line["spans"]]
                    lines.append("".join(span_texts))

                block_text = " ".join(lines).strip()
                if block_text:
                    blocks_list.append(block_text)

        return blocks_list
