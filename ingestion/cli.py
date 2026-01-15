"""Command-line interface for document ingestion."""

import argparse
from pathlib import Path

from ingestion.pipeline import IngestionPipeline


def main():
    """Main entry point for ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system vector store"
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to file or directory to ingest",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Recursively process directories (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not process subdirectories",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show collection statistics after ingestion",
    )

    args = parser.parse_args()
    path = Path(args.path)
    recursive = not args.no_recursive

    pipeline = IngestionPipeline()

    print(f"Starting ingestion from: {path}")
    print("-" * 50)

    if path.is_file():
        try:
            chunks = pipeline.ingest_file(path)
            print(f"\nSuccessfully ingested {chunks} chunks from {path}")
        except Exception as e:
            print(f"\nError: {e}")
            return 1
    elif path.is_dir():
        results = pipeline.ingest_directory(path, recursive=recursive)
        print("\n" + "=" * 50)
        print("Ingestion Summary:")
        print(f"  Files processed: {results['processed_files']}")
        print(f"  Total chunks: {results['total_chunks']}")
        if results["errors"]:
            print(f"  Errors: {len(results['errors'])}")
            for error in results["errors"]:
                print(f"    - {error['file']}: {error['error']}")
    else:
        print(f"Error: Path not found: {path}")
        return 1

    if args.stats:
        print("\n" + "-" * 50)
        stats = pipeline.get_collection_stats()
        print("Collection Statistics:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total documents: {stats['document_count']}")

    return 0


if __name__ == "__main__":
    exit(main())
