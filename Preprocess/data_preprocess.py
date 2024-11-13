import os
import re
import json
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


class PreProcess:
    def __init__(
        self,
        source_path,
        questions_path,
        insurance_json_name,
        finance_json_name,
        faq_json_name,
    ):
        """定義資料前處理相關參數"""
        self.source_path = source_path
        self.questions_path = questions_path
        self.insurance_json_name = insurance_json_name
        self.finance_json_name = finance_json_name
        self.faq_json_name = faq_json_name
        # 中文數字到阿拉伯數字的映射
        self.chinese_numerals = {
            "○": "0",
            "一": "1",
            "二": "2",
            "三": "3",
            "四": "4",
            "五": "5",
            "六": "6",
            "七": "7",
            "八": "8",
            "九": "9",
        }

        # 定義排除 header 條件：包含五個大寫字母的模式
        self.exclude_pattern = r"(?!.*[A-Z]{5})"
        # 定義篩選 header 條件
        self.header1_pattern = re.compile(
            r"^\n?【" + self.exclude_pattern + r".*?】\n?"
        )
        self.header2_pattern = re.compile(
            r"\n第\s?([一二三四五六七八九十百零]+(\s?[一二三四五六七八九十百零]+)*)\s?條\s|第\s?([一二三四五六七八九十百零]+(\s?[一二三四五六七八九十百零]+)*)\s?條(【.*?】)"
        )
        self.header3_pattern = re.compile(
            r"[一二三四五六七八九十百零]+\s?、\s*(.*?)\s*"
        )
        self.header4_pattern = re.compile(
            r"\(([一二三四五六七八九十百零]+)\)\s*(.*?)\s*"
        )

        # 指定要分割的 header 符號，並標記為屬於 Header 1, 2 ...
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

    # PDF 讀取與數據載入
    def read_pdf(self, pdf_loc):
        """讀取 PDF 並返回指定頁面的提取文本。

        Args:
            pdf_loc (str): PDF 文件的路徑

        Returns:
            str: 指定頁面的提取文本
        """
        if pdf_loc.endswith(".pdf"):
            with pdfplumber.open(pdf_loc) as pdf:
                pdf_text = "".join([page.extract_text() or "" for page in pdf.pages])
            return pdf_text

    def load_data(self, source_path, json_name):
        """從 PDF 載入或提取文本並保存為 JSON 文件。

        Args:
            source_path (str): PDF 文件的路徑
            json_name (str): 要保存的 JSON 文件名稱

        Returns:
            dict: 從 PDF 中提取的文本
        """
        json_file_path = os.path.join(source_path, json_name)

        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                corpus_dict = json.load(json_file)
                return {int(key): value for key, value in corpus_dict.items()}

        # 若 JSON 文件不存在，則從 PDF 中提取並保存文本
        masked_file_ls = os.listdir(source_path)
        corpus_dict = {
            int(file.replace(".pdf", "")): self.read_pdf(
                os.path.join(source_path, file)
            )
            for file in tqdm(masked_file_ls)
        }

        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(corpus_dict, json_file, ensure_ascii=False, indent=4)

        return corpus_dict

    def get_raw_data(self):
        """載入原始資料與問題

        Returns:
            dict: questions 與 corpora 的 dict 資料
        """
        # 載入 insurance、finance 內容
        corpora = {
            "insurance": self.load_data(
                os.path.join(self.source_path, "insurance"), self.insurance_json_name
            ),
            "finance": self.load_data(
                os.path.join(self.source_path, "finance"), self.finance_json_name
            ),
        }
        # 載入 FAQ 內容
        with open(
            os.path.join(self.source_path, "faq", self.faq_json_name),
            "r",
            encoding="utf-8",
        ) as f:
            corpora["faq"] = {
                int(key): str(value) for key, value in json.load(f).items()
            }

        # 載入 questions 內容
        with open(self.questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        return corpora, questions

    def chinese_to_arabic(self, chinese_str):
        """用來轉換中文數字成阿拉伯數字的函數

        Args:
            chinese_str (str): 中文數字

        Returns:
            str: 阿拉伯數字
        """
        # 處理「十」作為「10」的情況
        chinese_str = chinese_str.replace("十", "")
        # 替換每個中文數字，根據映射
        for key, value in self.chinese_numerals.items():
            chinese_str = chinese_str.replace(key, value)
        return chinese_str

    def process_text(self, text):
        """處理字典中的所有文本（找到民國年月的正則表達式，並替換成阿拉伯數字）

        Args:
            text (str): 待處理的文本

        Returns:
            str: 處理後的文本
        """
        text = re.sub(
            r"([一二三四五六七八九○]+)年",
            lambda m: "民國" + self.chinese_to_arabic(m.group(1)) + "年",
            text,
        )
        text = re.sub(
            r"([一二三四五六七八九十]+)月",
            lambda m: self.chinese_to_arabic(m.group(1)) + "月",
            text,
        )
        text = re.sub(
            r"([一二三四五六七八九十]+)日",
            lambda m: self.chinese_to_arabic(m.group(1)) + "日",
            text,
        )
        text = re.sub(
            r"([一二三四]+)季",
            lambda m: self.chinese_to_arabic(m.group(1)) + "季",
            text,
        )
        return text

    def process_dict(self, corpora):
        """將 dict 中的資料進行日期轉換處理

        Args:
            corpora (dict): _description_

        Returns:
            dict: _description_
        """
        for key, text in corpora.items():
            corpora[key] = self.process_text(text)
        return corpora

    def build_documents(self, corpora):
        """建立並產出 Documents

        Args:
            corpora (dict): 還沒被 chunk 過的資料

        Returns:
            dict: 經過 MarkdownHeaderTextSplitter 與 RecursiveCharacterTextSplitter 處理過的資料
        """
        # 初始化 MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

        # 初始化 RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            add_start_index=True,
        )

        # 建立一個字典來儲存不同類別的切分資料
        doc_dict = {}
        all_categories = ["finance", "faq"]

        documents = []
        for key, _ in corpora["insurance"].items():
            modified_text = re.sub(
                self.header1_pattern,
                lambda match: "\n# " + match.group(0).replace("\n", "") + "\n",
                corpora["insurance"][key],
            )
            modified_text = re.sub(
                self.header2_pattern,
                lambda match: "\n## " + match.group(0).replace("\n", "") + "\n",
                modified_text,
            )
            modified_text = re.sub(
                self.header3_pattern,
                lambda match: "\n### " + match.group(0).replace("\n", "") + "\n",
                modified_text,
            )
            modified_text = re.sub(
                self.header4_pattern,
                lambda match: "\n#### " + match.group(0).replace("\n", "") + "\n",
                modified_text,
            )
            chunks = markdown_splitter.split_text(modified_text)
            for chunk in chunks:
                chunk.metadata["pid"] = key
                chunk.metadata["category"] = "insurance"
                documents.append(chunk)
        # 將 documents 儲存到字典中
        doc_dict["insurance"] = documents

        # 處理 finance, faq 類別
        # 進行處理並儲存結果
        for category in all_categories:
            documents = []
            for key, content in corpora[category].items():
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append(
                        Document(page_content=chunk, metadata={"pid": key})
                    )
            # 將 documents 儲存到字典中
            doc_dict[category] = documents

        # 將 insurance documents 中的 metadata 的 header 加到 page_content 中
        for idx in range(len(doc_dict["insurance"])):
            full_content = []
            for key, content in doc_dict["insurance"][idx].metadata.items():
                if key != "pid" and key != "category":
                    full_content.append(content)
            full_content.append(doc_dict["insurance"][idx].page_content)
            doc_dict["insurance"][idx].page_content = " ".join(full_content)

        return doc_dict
