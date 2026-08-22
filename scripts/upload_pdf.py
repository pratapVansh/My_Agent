"""
Upload PDF documents to your personal memory.
Supports both single file and multiple files.
"""
import asyncio
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyPDF2 import PdfReader  # noqa: E402
from app.memory.memory_manager import memory_manager  # noqa: E402
from app.services.chunking_service import chunking_service  # noqa: E402


async def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    print(f"Reading PDF: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(page_text)
                print(f"   [OK] Extracted page {i}/{len(reader.pages)}")

        full_text = "\n\n".join(text_parts)
        print(f"   [OK] Total text length: {len(full_text)} characters")
        return full_text

    except Exception as e:
        print(f"   [ERROR] Error reading PDF: {e}")
        return ""


async def upload_single_pdf(pdf_path: str, user_id: str, document_type: str = "general"):
    """
    Upload a single PDF to memory.

    Args:
        pdf_path: Path to the PDF file
        user_id: Your user ID
        document_type: Type of document (resume, projects, notes, etc.)
    """
    # Extract text
    text = await extract_text_from_pdf(pdf_path)

    if not text:
        print("[ERROR] No text extracted from PDF")
        return

    # Initialize memory
    await memory_manager.initialize()

    # Store with metadata
    metadata = {
        "source_file": Path(pdf_path).name,
        "document_type": document_type,
        "file_path": pdf_path
    }

    print(f"\nStoring in long-term memory...")

    # Store as resume (uses semantic chunking + embeddings)
    doc_id = await memory_manager.store_resume(
        user_id=user_id,
        resume_text=text,
        metadata=metadata
    )

    print(f"[SUCCESS] Uploaded successfully!")
    print(f"   Document ID: {doc_id}")
    print(f"   User ID: {user_id}")
    print(f"   Type: {document_type}")

    return doc_id


async def upload_multiple_pdfs(pdf_folder: str, user_id: str):
    """
    Upload all PDFs from a folder.

    Args:
        pdf_folder: Path to folder containing PDFs
        user_id: Your user ID
    """
    folder = Path(pdf_folder)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in {pdf_folder}")
        return

    print(f"Found {len(pdf_files)} PDF files\n")

    results = []
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print('='*60)

        # Detect document type from filename
        filename_lower = pdf_file.name.lower()
        if "resume" in filename_lower or "cv" in filename_lower:
            doc_type = "resume"
        elif "project" in filename_lower:
            doc_type = "projects"
        elif "skill" in filename_lower:
            doc_type = "skills"
        elif "research" in filename_lower or "notes" in filename_lower:
            doc_type = "research"
        else:
            doc_type = "general"

        doc_id = await upload_single_pdf(str(pdf_file), user_id, doc_type)
        results.append((pdf_file.name, doc_id))

    print(f"\n\n{'='*60}")
    print("[SUCCESS] ALL UPLOADS COMPLETE")
    print('='*60)
    print(f"Total files uploaded: {len(results)}")
    for filename, doc_id in results:
        print(f"  [OK] {filename} -> {doc_id}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("PDF UPLOAD TOOL")
    print("=" * 60)
    print()

    parser = argparse.ArgumentParser(description="Upload PDF documents into long-term memory")
    parser.add_argument("path", help="Path to PDF file or directory containing PDFs")
    parser.add_argument("--user-id", required=True, help="Stable user_id used by chat queries")
    args = parser.parse_args()

    if not args.path:
        print("Usage:")
        print()
        print("  Upload single PDF:")
        print(f"    python scripts/upload_pdf.py path/to/your/file.pdf --user-id your_user_id")
        print()
        print("  Upload all PDFs from folder:")
        print(f"    python scripts/upload_pdf.py path/to/your/folder/ --user-id your_user_id")
        print()
        print("Examples:")
        print(f"    python scripts/upload_pdf.py path/to/resume.pdf --user-id user_123")
        print(f"    python scripts/upload_pdf.py path/to/pdf_folder/ --user-id user_123")
        print()
        sys.exit(1)

    path = args.path
    user_id = args.user_id.strip().lower()

    print("UPLOAD user_id:", user_id)

    path_obj = Path(path)

    if not path_obj.exists():
        print(f"[ERROR] Path not found: {path}")
        sys.exit(1)

    # Upload single file or folder
    if path_obj.is_file():
        await upload_single_pdf(str(path_obj), user_id)
    elif path_obj.is_dir():
        await upload_multiple_pdfs(str(path_obj), user_id)
    else:
        print(f"[ERROR] Invalid path: {path}")

    print("\n" + "=" * 60)
    print("Done! Your data is now in the system's memory.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
