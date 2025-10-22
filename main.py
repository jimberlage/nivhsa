import argparse
import asyncio
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Self

import aiofiles
from anthropic import AsyncAnthropic
from anthropic.types import Message
from markitdown import MarkItDown
from pdf2image import convert_from_path
from pydantic import BaseModel


def is_empty_str(s: Optional[str]) -> bool:
    """
    Check if a string is None or empty after stripping whitespace.

    Args:
        s: The string to check, can be None.

    Returns:
        True if the string is None or empty/whitespace-only, False otherwise.
    """

    if s is None or s.strip() == "":
        return True

    return False


class Result(BaseModel):
    document_type: Optional[str] = None
    payor: Optional[str] = None
    text: Optional[str] = None

    def __repr__(self):
        return f"Result(document_type={self.document_type!r}, payor={self.payor!r}, text={self.text!r})"

    def merge(self, other: Self):
        """
        Merge fields from another Result instance, only updating empty fields.

        Args:
            other: Another Result instance to merge fields from.
        """

        if is_empty_str(self.document_type) and not is_empty_str(other.document_type):
            self.document_type = other.document_type

        if is_empty_str(self.payor) and not is_empty_str(other.payor):
            self.payor = other.payor

        if is_empty_str(self.text) and not is_empty_str(other.text):
            self.text = other.text

    @classmethod
    def parse_from_message(cls, message: Message, text: str) -> Self:
        """
        Parse a Result from an Anthropic message and text content.

        Args:
            message: The Anthropic message containing tool use results.
            text: The extracted text content from the document.

        Returns:
            A Result instance with parsed document_type, payor, and text fields.
        """

        if message.content is None or len(message.content) == 0:
            return cls()

        for content_block in message.content:
            if hasattr(content_block, "type") and content_block.type == "tool_use":
                tool_input = content_block.input

                if isinstance(tool_input, dict):
                    return cls(
                        document_type=tool_input.get("document_type"),
                        payor=tool_input.get("payor"),
                        text=text,
                    )

        return cls()


@dataclass
class Args:
    claims_path: str

    @classmethod
    def parse(cls) -> Self:
        """
        Parse command-line arguments and validate the claims path.

        Returns:
            An Args instance with validated claims_path.

        Raises:
            RuntimeError: If the claims path doesn't exist or is not a directory.
        """

        parser = argparse.ArgumentParser(
            description="Process claims files using Anthropic API"
        )
        parser.add_argument(
            "--claims-path",
            required=True,
            type=str,
            help="Path to the directory containing claims files",
        )

        os_args = parser.parse_args()

        # Validate that the claims path exists and is a directory
        if not os.path.exists(os_args.claims_path):
            raise RuntimeError(
                f"Error: Claims path '{os_args.claims_path}' does not exist."
            )

        if not os.path.isdir(os_args.claims_path):
            raise RuntimeError(
                f"Error: Claims path '{os_args.claims_path}' is not a directory."
            )

        return cls(claims_path=os_args.claims_path)


async def parse_text_file(markdown_client: MarkItDown, filepath: str) -> str:
    """
    Parse text content from a file using MarkItDown.

    Args:
        markdown_client: The MarkItDown client instance.
        filepath: Path to the file to parse.

    Returns:
        The extracted text content from the file.
    """

    markdown_result = markdown_client.convert(filepath)

    return markdown_result.text_content


async def parse_image_file(anthropic_client: AsyncAnthropic, filepath: str) -> str:
    """
    Parse text content from a PDF file by converting to images and using Claude vision.

    Args:
        anthropic_client: The Anthropic client for API calls.
        filepath: Path to the PDF file to parse.

    Returns:
        The extracted text content from all pages of the PDF.
    """

    # Convert PDF to images
    images = convert_from_path(filepath)

    # Convert all pages to base64 encoded JPEGs
    content = []

    for i, image in enumerate(images):
        # Convert PIL Image to JPG format and encode as base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        # Encode as base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                },
            }
        )

    # Add text instruction after all images
    content.append(
        {
            "type": "text",
            "text": f"Please analyze this document which has {len(images)} page(s).  Extract all text from this image and return it as a single string.",
        }
    )

    message = await anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="claude-sonnet-4-5-20250929",
    )

    if message.content is None or len(message.content) == 0:
        return ""

    for content_block in message.content:
        if hasattr(content_block, "type") and content_block.type == "text":
            return content_block.text or ""

    return ""


async def categorize_str(anthropic_client: AsyncAnthropic, text: str) -> Result:
    """
    Categorize a text string using Claude to extract document type and payor information.

    Args:
        anthropic_client: The Anthropic client for API calls.
        text: The text content to analyze.

    Returns:
        A Result containing the document type, payor, and original text.
    """

    if not text or text.strip() == "":
        return Result()

    message = await anthropic_client.messages.create(
        max_tokens=1024,
        system="You are an analyst, examining faxes related to durable medical equipment.  Your task is to carefully examine documents and extract all textual content while identifying document characteristics.  Be thorough in your text extraction, capturing even small or partially obscured text. Pay close attention to document quality indicators such as handwriting, photocopying artifacts, and overall document condition.  Documents may be compliance reports, delivery tickets, orders, physician notes, prescriptions, or sleep study reports.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this document.  We are interested in the type of the document, as well as other features like the payor (insurer).",
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
        model="claude-sonnet-4-5-20250929",
        tools=[
            {
                "name": "analysis",
                "description": "Analyze some features of the document.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "enum": [
                                "compliance_report",
                                "delivery_ticket",
                                "order",
                                "physician_note",
                                "prescription",
                                "sleep_study_report",
                            ],
                            "description": "The type of document being analyzed",
                        },
                        "payor": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": "The payor or insurance company mentioned in the document, if any",
                        },
                    },
                    "required": ["payor"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "analysis"},
    )

    return Result.parse_from_message(message, text)


async def categorize_image_file(anthropic_client: AsyncAnthropic, filepath: str):
    """
    Categorize a file by treating it as an image/PDF and extracting text via OCR.

    Args:
        anthropic_client: The Anthropic client for API calls.
        filepath: Path to the image or PDF file to analyze.

    Returns:
        A Result containing the document type, payor, and extracted text.
    """

    text = await parse_image_file(anthropic_client, filepath)

    return await categorize_str(anthropic_client, text)


async def categorize_text_file(
    anthropic_client: AsyncAnthropic, markdown_client: MarkItDown, filepath: str
):
    """
    Categorize a file by treating it as a text document and parsing with MarkItDown.

    Args:
        anthropic_client: The Anthropic client for API calls.
        markdown_client: The MarkItDown client for text extraction.
        filepath: Path to the text file to analyze.

    Returns:
        A Result containing the document type, payor, and extracted text.
    """

    text = await parse_text_file(markdown_client, filepath)

    return await categorize_str(anthropic_client, text)


async def categorize_file(
    anthropic_client: AsyncAnthropic, markdown_client: MarkItDown, filepath: str
) -> Result:
    """
    Categorize a file using both image OCR and text parsing approaches in parallel.

    Args:
        anthropic_client: The Anthropic client for API calls.
        markdown_client: The MarkItDown client for text extraction.
        filepath: Path to the file to analyze.

    Returns:
        A merged Result containing the best available document type, payor, and text
        from both parsing approaches.
    """

    image_result, text_result = await asyncio.gather(
        categorize_image_file(anthropic_client, filepath),
        categorize_text_file(anthropic_client, markdown_client, filepath),
    )

    text_result.merge(image_result)

    return text_result


async def write_file_metadata(filepath: str, result: Result):
    """
    Write the analysis result to a metadata JSON file asynchronously.

    Args:
        filepath: The original file path (metadata will be saved as filepath.metadata.json).
        result: The Result object containing analysis data to write.
    """

    metadata_filepath = f"{filepath}.metadata.json"

    # Convert the Pydantic model to a dictionary for JSON serialization
    result_dict = result.model_dump()

    async with aiofiles.open(metadata_filepath, "w", encoding="utf-8") as f:
        await f.write(json.dumps(result_dict, indent=2, ensure_ascii=False))


async def process_file(anthropic_client: AsyncAnthropic, filepath: str):
    """
    Process a single file by categorizing it and writing metadata, skipping metadata files.

    Args:
        anthropic_client: The Anthropic client for API calls.
        filepath: Path to the file to process.
    """

    # Skip files like "*.metadata.json"
    if filepath.endswith(".metadata.json"):
        return

    # Not async-safe [1], so using one client per-file to avoid mixing file state.
    # [1]: https://github.com/microsoft/markitdown/issues/13
    markdown_client = MarkItDown(enable_plugins=False)

    result = await categorize_file(anthropic_client, markdown_client, filepath)

    await write_file_metadata(filepath, result)


async def process_files(anthropic_client: AsyncAnthropic, args: Args):
    """
    Process all files in the claims directory sequentially.

    Args:
        anthropic_client: The Anthropic client for API calls.
        args: Parsed command-line arguments containing the claims directory path.
    """

    filepaths = os.listdir(args.claims_path)

    for filepath in filepaths:
        full_filepath = os.path.join(args.claims_path, filepath)

        await process_file(anthropic_client, full_filepath)


async def main():
    """
    Main entry point for the claims processing application.

    Parses command-line arguments, initializes the Anthropic client,
    and processes all files in the specified claims directory.

    Returns:
        Exit code (0 for success).
    """

    args = Args.parse()

    # Initialize Anthropic client
    anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    await process_files(anthropic_client, args)

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
