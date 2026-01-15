"""Document loaders for various file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import markdown
import pypdf
from bs4 import BeautifulSoup
from docx import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """Extract text content from file.

        Args:
            file_path: Path to the file to load.

        Returns:
            Extracted text content.
        """
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if this loader supports the given file type.

        Args:
            file_path: Path to check.

        Returns:
            True if this loader can handle the file.
        """
        pass


class PDFLoader(BaseLoader):
    """Loader for PDF files using pypdf."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> str:
        reader = pypdf.PdfReader(str(file_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)


class DOCXLoader(BaseLoader):
    """Loader for Microsoft Word documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def load(self, file_path: Path) -> str:
        doc = Document(str(file_path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def load(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".md"

    def load(self, file_path: Path) -> str:
        md_content = file_path.read_text(encoding="utf-8")
        # Convert markdown to HTML, then extract plain text
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n\n")


class HTMLLoader(BaseLoader):
    """Loader for HTML files."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".html", ".htm"]

    def load(self, file_path: Path) -> str:
        html_content = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "lxml")
        # Remove script, style, and navigation elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        return soup.get_text(separator="\n\n", strip=True)


class DocumentLoaderRegistry:
    """Registry for document loaders with automatic format detection."""

    def __init__(self):
        self._loaders = [
            PDFLoader(),
            DOCXLoader(),
            TextLoader(),
            MarkdownLoader(),
            HTMLLoader(),
        ]

    def get_loader(self, file_path: Path) -> Optional[BaseLoader]:
        """Get appropriate loader for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Loader instance or None if no loader supports the format.
        """
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        return None

    def load(self, file_path: Path) -> str:
        """Load a document using the appropriate loader.

        Args:
            file_path: Path to the document.

        Returns:
            Extracted text content.

        Raises:
            ValueError: If no loader supports the file format.
        """
        loader = self.get_loader(file_path)
        if loader is None:
            raise ValueError(f"No loader available for file type: {file_path.suffix}")
        return loader.load(file_path)

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file format is supported.

        Args:
            file_path: Path to check.

        Returns:
            True if the format is supported.
        """
        return self.get_loader(file_path) is not None
