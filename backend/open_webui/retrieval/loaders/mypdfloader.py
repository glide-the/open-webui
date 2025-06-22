"""
form chatchat 
"""
from typing import List, Tuple

import cv2
import numpy as np
import tqdm
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from PIL import Image
 
PDF_OCR_THRESHOLD: Tuple[float, float] = (0.6, 0.6)

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try: 

        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
    return ocr


class RapidOCRPDFLoader(UnstructuredFileLoader):
    
    def _get_elements(self) -> List:
  
        def pdf2text(filepath):
            import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            if self.unstructured_kwargs.get("extract_images", False):
                ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(
                total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            for i, page in enumerate(doc):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                if self.unstructured_kwargs.get("extract_images", False):
                    

                    img_list = page.get_image_info(xrefs=True)
                    for img in img_list:
                        if xref := img.get("xref"):
                            bbox = img["bbox"]
                            # 检查图片尺寸是否超过设定的阈值
                            if (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[
                                0
                            ] or (bbox[3] - bbox[1]) / (
                                page.rect.height
                            ) < PDF_OCR_THRESHOLD[1]:
                                continue
                            pix = fitz.Pixmap(doc, xref)
                    
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)

                            result, _ = ocr(img_array)
                            if result:
                                ocr_result = [line[1] for line in result]
                                resp += "\n".join(ocr_result)
                    pass

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(
        file_path="/mnt/ceph/develop/jiawei/open-webui/backend/data/uploads/ca507f21-4506-46ae-ab0b-a01babd4e820_NBT10223-2019煤炭建设工程资料归档及档案管理规范（同李园）(2).pdf",
        extract_images=True
    )
    docs = loader.load()
    print(docs)
