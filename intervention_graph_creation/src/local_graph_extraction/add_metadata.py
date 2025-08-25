import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..data_interfaces.models import Publication
from ..data_interfaces.ard_json_loader import (
    load_publications_from_hf_ard,
    load_publications_from_local_ard,
)
from ..data_interfaces.arxiv_api_loader import load_publications_from_folder
from ..data_interfaces.utils import (
    extract_arxiv_id_from_url,
    parse_arxiv_id_from_filename,
)

logger = logging.getLogger(__name__)


class MetadataAdder:
    def __init__(
        self,
        ard_dataset_path: Optional[str] = None,
        arxiv_pdfs_path: Optional[str] = None,
    ):
        self.ard_dataset_path = ard_dataset_path
        self.arxiv_pdfs_path = Path(arxiv_pdfs_path) if arxiv_pdfs_path else None
        self._publications: List[Publication] = []
        self._publication_index: Dict[str, Publication] = {}
        self._loaded = False

    def _load_publications(self) -> None:
        """Load all publications and create index."""
        if self._loaded:
            return

        logger.info("Loading publications...")
        try:
            if self.ard_dataset_path and Path(self.ard_dataset_path).exists():
                ard_pubs = load_publications_from_local_ard(self.ard_dataset_path)
            else:
                ard_pubs = load_publications_from_hf_ard(dedupe=True)
            self._publications.extend(ard_pubs)
            logger.info(f"Loaded {len(ard_pubs)} ARD publications")
        except Exception as e:
            logger.warning(f"Failed to load ARD publications: {e}")

        try:
            if self.arxiv_pdfs_path and self.arxiv_pdfs_path.exists():
                arxiv_pubs = load_publications_from_folder(str(self.arxiv_pdfs_path))
                self._publications.extend(arxiv_pubs)
                logger.info(f"Loaded {len(arxiv_pubs)} ArXiv publications")
        except Exception as e:
            logger.warning(f"Failed to load ArXiv publications: {e}")

        # Create index
        for pub in self._publications:
            # Index by URL
            if pub.url:
                self._publication_index[pub.url] = pub

                # Extract and index ArXiv ID if it's an ArXiv URL
                arxiv_id = extract_arxiv_id_from_url(pub.url)
                if arxiv_id:
                    self._publication_index[arxiv_id] = pub
                    # Also index without version
                    base_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                    if base_arxiv_id != arxiv_id:
                        self._publication_index[base_arxiv_id] = pub

            # Index by PDF filename if available
            if pub.pdf_file_path:
                pdf_path = Path(pub.pdf_file_path)
                arxiv_id = parse_arxiv_id_from_filename(pdf_path.name)
                if arxiv_id:
                    self._publication_index[arxiv_id] = pub
                    base_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                    if base_arxiv_id != arxiv_id:
                        self._publication_index[base_arxiv_id] = pub

        self._loaded = True
        logger.info(f"Indexed {len(self._publication_index)} publication identifiers")

    def get_publication(self, identifier: str) -> Optional[Publication]:
        """
        Get publication by identifier (ArXiv ID, URL, filename, etc.).
        """
        self._load_publications()

        # Direct lookup
        if identifier in self._publication_index:
            return self._publication_index[identifier]

        # Try extracting ArXiv ID from filename
        if identifier.endswith(".pdf"):
            arxiv_id = parse_arxiv_id_from_filename(identifier)
            if arxiv_id and arxiv_id in self._publication_index:
                return self._publication_index[arxiv_id]

        # Try extracting ArXiv ID from URL
        arxiv_id = extract_arxiv_id_from_url(identifier)
        if arxiv_id and arxiv_id in self._publication_index:
            return self._publication_index[arxiv_id]

        return None

    def create_source_metadata(
        self, publication: Publication, source_identifier: str
    ) -> Dict[str, Any]:
        """
        Create source metadata dictionary from publication.
        """

        # Determine source type from URL
        source_type = "unknown"
        if publication.url:
            if "arxiv.org" in publication.url:
                source_type = "arxiv"
            elif "alignmentforum.org" in publication.url:
                source_type = "alignmentforum"
            elif "lesswrong.com" in publication.url:
                source_type = "lesswrong"
            elif "youtube.com" in publication.url or "youtu.be" in publication.url:
                source_type = "youtube"
            elif any(
                domain in publication.url
                for domain in ["blog", "medium.com", "substack.com"]
            ):
                source_type = "blog"

        # Use ArXiv ID as paper_id if available
        paper_id = source_identifier
        if publication.url:
            arxiv_id = extract_arxiv_id_from_url(publication.url)
            if arxiv_id:
                paper_id = arxiv_id

        metadata = {
            "paper_id": paper_id,
            "title": publication.title,
            "authors": publication.authors,
            "date_published": publication.date_published,
            "url": publication.url,
            "source_type": source_type,
        }

        # Add abstract if available
        if publication.abstract:
            metadata["abstract"] = publication.abstract

        return metadata

    def add_metadata_to_schema(
        self, schema_data: Dict[str, Any], source_identifier: str
    ) -> Dict[str, Any]:
        # Try to find the publication
        publication = self.get_publication(source_identifier)

        if not publication:
            logger.warning(
                f"Could not find publication for identifier: {source_identifier}"
            )
            # Still add basic metadata with just the identifier
            source_metadata = {
                "paper_id": source_identifier,
                "title": None,
                "authors": None,
                "date_published": None,
                "url": None,
                "source_type": "unknown",
            }
        else:
            source_metadata = self.create_source_metadata(
                publication, source_identifier
            )

        # Add metadata to all nodes
        if "nodes" in schema_data:
            for node in schema_data["nodes"]:
                node["source_metadata"] = source_metadata.copy()

        # Add metadata to all edges
        if "logical_chains" in schema_data:
            for chain in schema_data["logical_chains"]:
                if "edges" in chain:
                    for edge in chain["edges"]:
                        edge["source_metadata"] = source_metadata.copy()

        return schema_data

    def process_extraction_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        source_identifier: Optional[str] = None,
    ) -> str:
        """
        Process a single extraction file and add metadata.
        """
        input_path = Path(input_path)

        # Default output path
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_with_metadata.json"
        else:
            output_path = Path(output_path)

        # Default source identifier
        if not source_identifier:
            source_identifier = input_path.stem
            # Try to extract ArXiv ID from filename
            arxiv_id = parse_arxiv_id_from_filename(input_path.name)
            if arxiv_id:
                source_identifier = arxiv_id

        # Load extraction data
        with input_path.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)

        # Add metadata
        enriched_data = self.add_metadata_to_schema(schema_data, source_identifier)

        # Save enriched data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Added metadata to {input_path} -> {output_path}")
        return str(output_path)

    def process_directory(
        self, input_dir: str, output_dir: Optional[str] = None, pattern: str = "*.json"
    ) -> List[str]:
        """
        Process all extraction files in a directory.
        """
        input_dir = Path(input_dir)
        if not output_dir:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)

        output_paths = []

        # Find all matching files
        json_files = list(input_dir.glob(pattern))
        if not json_files:
            logger.warning(
                f"No files matching pattern '{pattern}' found in {input_dir}"
            )
            return output_paths

        logger.info(f"Processing {len(json_files)} files...")

        for json_file in json_files:
            # Skip files that already have metadata
            if "_with_metadata" in json_file.name:
                logger.info(f"{json_file} already has metadata")
                continue

            try:
                output_path = output_dir / f"{json_file.stem}_with_metadata.json"
                result_path = self.process_extraction_file(
                    str(json_file), str(output_path)
                )
                output_paths.append(result_path)
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                continue

        logger.info(f"Successfully processed {len(output_paths)} files")
        return output_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add metadata to extraction outputs")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file or directory containing extraction JSONs",
    )
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument(
        "--source-id", "-s", help="Source identifier (ArXiv ID, filename, etc.)"
    )
    parser.add_argument("--ard-path", help="Path to local ARD dataset")
    parser.add_argument("--arxiv-path", help="Path to ArXiv PDFs directory")

    args = parser.parse_args()

    adder = MetadataAdder(
        ard_dataset_path=args.ard_path, arxiv_pdfs_path=args.arxiv_path
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        output_path = adder.process_extraction_file(
            str(input_path), args.output, args.source_id
        )
        print(f"Processed: {output_path}")
    elif input_path.is_dir():
        # Process directory
        output_paths = adder.process_directory(str(input_path), args.output)
        print(f"Processed {len(output_paths)} files")

        # Show first 5
        for path in output_paths[:5]:
            print(f"  {path}")
        if len(output_paths) > 5:
            print(f"  ... and {len(output_paths) - 5} more")
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
