"""Split articles into retrieval chunks.

Articles are split on `##` section headings first, so each chunk covers one
topic, then any long section is split by size. Each chunk's text is prefixed
with "<article title> > <section>" so a short chunk still carries its context
when embedded and when shown to the LLM.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_section_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "section")])


def chunk_documents(documents: list[Document], chunk_size: int = 1500, chunk_overlap: int = 150) -> list[Document]:
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        sections = _section_splitter.split_text(document.page_content)
        pieces = [s for s in size_splitter.split_documents(sections) if s.page_content.strip()]
        for i, section in enumerate(pieces):
            heading = section.metadata.get("section", "")
            prefix = f"{document.metadata['title']} > {heading}" if heading else document.metadata["title"]
            metadata = {**document.metadata, "section": heading, "chunk_id": f"{document.metadata['file']}#{i}"}
            chunks.append(Document(page_content=f"{prefix}\n\n{section.page_content.strip()}", metadata=metadata))
    return chunks
