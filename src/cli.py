import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel

from src.config import TaggerConfig
from src.scanner import DirectoryScanner
from src.raw_extractor import RawExtractor
from src.vision_tagger import VisionTagger
from src.xmp_manager import XMPManager


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-tagger",
        description="Local AI Vision Auto-Tagger for Lightroom RAW & Image files with XMP sidecars.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the directory containing RAW or image files",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model ID or substring in LM Studio (e.g. 'google/gemma-3-12b' or 'gemma-3'). Default: auto-detect vision model",
    )
    parser.add_argument(
        "-l", "--list-models",
        action="store_true",
        help="List available models in LM Studio and exit",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scan directory recursively",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio API base URL (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze images and print tags without writing or modifying XMP files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tags in XMP files instead of merging",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag images even if they already have tags in their XMP sidecars",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1536,
        help="Max pixel dimension (long edge) to send to vision model (default: 1536)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds per image (default: 60)",
    )
    return parser


def list_models_action(api_url: str) -> int:
    """Queries LM Studio and prints all available models with vision detection."""
    tagger = VisionTagger(api_url=api_url)
    try:
        models = tagger.get_available_models()
    except Exception as e:
        console.print(f"[bold red]Failed to connect to LM Studio at {api_url}:[/bold red] {e}")
        return 1

    table = Table(title=f"Models Available in LM Studio ({api_url})", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="bold cyan")
    table.add_column("Type", justify="center")

    for i, m in enumerate(models, 1):
        is_vision = tagger.is_likely_vision_model(m)
        type_str = "[bold green]👁️ Vision / Multimodal[/bold green]" if is_vision else "[dim]Text Only[/dim]"
        table.add_row(str(i), m, type_str)

    console.print(table)
    console.print("\n💡 [bold]To use a specific model:[/bold] pass [cyan]-m <model_id>[/cyan] or a substring (e.g. [cyan]-m gemma-3[/cyan])\n")
    return 0


def run_pipeline(config: TaggerConfig) -> int:
    if config.folder is None:
        console.print("[bold red]Error:[/bold red] Folder argument is required.")
        console.print("Run [cyan]./tagger --help[/cyan] for usage or [cyan]./tagger -l[/cyan] to list available models.")
        return 1

    console.print(
        Panel.fit(
            "[bold cyan]Local Vision XMP Auto-Tagger for Lightroom[/bold cyan]\n"
            f"[dim]Folder:[/dim] [green]{config.folder.resolve()}[/green]\n"
            f"[dim]LM Studio URL:[/dim] [yellow]{config.api_url}[/yellow]\n"
            f"[dim]Dry Run:[/dim] {'[bold yellow]YES (No files modified)[/bold yellow]' if config.dry_run else '[green]NO (Writing XMP)[/green]'}\n"
            f"[dim]Mode:[/dim] {'[red]Overwrite existing tags[/red]' if config.overwrite_tags else '[blue]Merge with existing tags[/blue]'}",
            title="📸 Image Tagger",
            border_style="cyan"
        )
    )

    # 1. Connect & check vision model
    tagger = VisionTagger(api_url=config.api_url, model=config.model, timeout=config.timeout)
    try:
        resolved_model = tagger.resolve_model()
        is_vision = tagger.is_likely_vision_model(resolved_model)
        type_badge = "[bold green](Vision Model)[/bold green]" if is_vision else "[bold yellow](⚠️ May be text-only)[/bold yellow]"
        console.print(f"🤖 [bold green]Connected to LM Studio![/bold green] Using model: [bold yellow]{resolved_model}[/bold yellow] {type_badge}\n")
    except Exception as e:
        console.print(f"[bold red]Error resolving model in LM Studio:[/bold red] {e}")
        console.print("\n[yellow]Tips:[/yellow]")
        console.print("  1. Ensure LM Studio is open and the local server is running on port 1234.")
        console.print("  2. Run [bold cyan]./tagger -l[/bold cyan] to see available models.")
        return 1

    # 2. Scan files
    scanner = DirectoryScanner(
        folder=config.folder,
        recursive=config.recursive,
        skip_tagged=config.skip_tagged,
    )

    try:
        items = scanner.discover_files()
    except Exception as e:
        console.print(f"[bold red]Scan error:[/bold red] {e}")
        return 1

    if not items:
        console.print("[yellow]No eligible images found to process.[/yellow]")
        if config.skip_tagged:
            console.print("[dim](If images are already tagged, use --force to re-tag them)[/dim]")
        return 0

    console.print(f"🔍 Found [bold cyan]{len(items)}[/bold cyan] eligible image(s) to process.\n")

    # 3. Process images
    extractor = RawExtractor(max_dimension=config.max_dimension, jpeg_quality=config.jpeg_quality)
    success_count = 0
    fail_count = 0
    consecutive_api_errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(items))

        for item in items:
            progress.update(task, description=f"[cyan]Tagging [bold]{item.image_path.name}[/bold]...")

            try:
                # Extract and encode
                b64_img, dims = extractor.prepare_base64_jpeg(item.image_path)

                # Query VLM
                result = tagger.analyze_image(b64_img)

                # Reset consecutive error counter on success
                consecutive_api_errors = 0

                # Display in console
                console.print(f"\n📷 [bold]{item.image_path.name}[/bold] [dim]({dims[0]}x{dims[1]})[/dim]")
                if result.description:
                    console.print(f"   [dim italic]{result.description}[/dim italic]")
                if result.places:
                    console.print(f"   🏛️  [bold magenta]Places:[/bold magenta] {', '.join(result.places)}")
                console.print(f"   🏷️  [bold green]Tags:[/bold green] {', '.join(result.tags)}")
                if result.hierarchical_tags:
                    console.print(f"   🌲 [bold blue]Hierarchy:[/bold blue] {', '.join(result.hierarchical_tags[:4])}{'...' if len(result.hierarchical_tags) > 4 else ''}")

                # Write XMP if not dry run
                if not config.dry_run:
                    final_flat, final_hier = XMPManager.write_tags(
                        xmp_path=item.xmp_path,
                        new_flat_tags=result.tags,
                        new_hierarchical_tags=result.hierarchical_tags,
                        overwrite=config.overwrite_tags,
                    )
                    console.print(f"   💾 [dim]Saved to {item.xmp_path.name} ({len(final_flat)} total tags)[/dim]")

                success_count += 1
            except Exception as e:
                console.print(f"\n   ❌ [red]Failed [bold]{item.image_path.name}[/bold]:[/red] {e}")
                fail_count += 1

                # If the vision model API is rejecting requests with 400 or compatibility issues, abort early
                if "LM Studio" in str(e) or "400" in str(e):
                    consecutive_api_errors += 1
                    if consecutive_api_errors >= 3:
                        console.print("\n[bold red]⚠️ Aborting batch:[/bold red] 3 consecutive API errors from LM Studio.")
                        console.print("[yellow]The selected model appears to reject vision/image requests. Please select a vision model (e.g. -m gemma-3) or check LM Studio.[/yellow]")
                        break

            progress.advance(task)

    # Summary
    console.print("\n" + "=" * 60)
    summary_table = Table(title="Batch Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")

    summary_table.add_row("[green]Successfully Tagged[/green]", str(success_count))
    if fail_count > 0:
        summary_table.add_row("[red]Failed[/red]", str(fail_count))
    if config.dry_run:
        summary_table.add_row("[yellow]Dry Run[/yellow]", "No files were written")

    console.print(summary_table)

    if not config.dry_run and success_count > 0:
        console.print(
            "\n💡 [bold green]In Lightroom Classic:[/bold green] Select your folder or photos and click\n"
            "   [bold cyan]Metadata -> Read Metadata from File[/bold cyan] (Cmd+Option+S on Mac) to load new tags into your catalog!"
        )

    return 0 if fail_count == 0 else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        sys.exit(list_models_action(args.url))

    if args.folder is None:
        parser.print_help()
        sys.exit(1)

    config = TaggerConfig(
        folder=args.folder,
        recursive=args.recursive,
        api_url=args.url,
        model=args.model,
        dry_run=args.dry_run,
        overwrite_tags=args.overwrite,
        skip_tagged=not args.force,
        max_dimension=args.max_size,
        timeout=args.timeout,
    )

    sys.exit(run_pipeline(config))


if __name__ == "__main__":
    main()
