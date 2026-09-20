#!/usr/bin/env python3
"""Stone Age 2 ANZ -> PNG extractor.

The decoder targets the ANZ graphics container used by the supplied Stone Age 2
resource. It extracts every sprite/frame as a PNG and can additionally create
one atlas per ANZ file.

Dependency:
    pip install pillow

Usage:
    python anz_to_png.py <input-folder>
    python anz_to_png.py <input-folder> -o <output-folder>
    python anz_to_png.py <file.anz>
    python anz_to_png.py <input-folder> --no-frames --atlas

The converter scans recursively by default when the input is a directory.

The format contains some metadata fields whose semantic names are not needed
for graphics extraction. The actual image path is fully decoded here:
ANZ header -> object/frame descriptors -> shared reference stream -> 32x32
zlib-compressed indexed tiles -> per-file 256-entry RGBA palette.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

MAGIC = b"ANZ "
EXPECTED_VERSION = 0x101
TILE_SIZE = 32
PALETTE_ENTRIES = 256
PALETTE_BYTES = PALETTE_ENTRIES * 4
NULL_TILE = 0xFFFFFFFF
OBJECT_HEADER_WORDS = 7
OBJECT_HEADER_SIZE = OBJECT_HEADER_WORDS * 4 + 4  # 28 bytes + 4-byte pad
RECORD_WORDS = 13
RECORD_SIZE = RECORD_WORDS * 4


class ANZError(Exception):
    """Raised for malformed or unsupported ANZ files."""


@dataclass
class FrameRecord:
    index: int
    cols: int
    tile_refs: int
    bbox: tuple[float, float, float, float]
    subindex: int
    refs: list[int]
    pixel_width: int = 0
    pixel_height: int = 0


@dataclass
class ANZObject:
    object_id: int
    records: list[FrameRecord]
    max_tiles: int
    palette: bytes


@dataclass
class ANZFile:
    path: Path
    version: int
    object_count: int
    reference_count: int
    tile_size: int
    tile_width: int
    tile_height: int
    bpp: int
    tile_count: int
    tile_index_offset: int
    tile_data_offset: int
    reference_offset: int
    tail_size: int
    objects: list[ANZObject]
    refs: list[int]
    tiles: list[bytes]
    palette: bytes


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def checked_slice(data: bytes, start: int, size: int, label: str) -> bytes:
    end = start + size
    if start < 0 or end > len(data):
        raise ANZError(
            f"{label} außerhalb der Datei: 0x{start:X}..0x{end:X} "
            f"bei Dateigröße 0x{len(data):X}"
        )
    return data[start:end]


def parse_objects(data: bytes, object_count: int) -> tuple[list[ANZObject], int]:
    """Parse the object/frame descriptors and return (objects, end_offset).

    The final object descriptor is followed by an 8-byte boundary overlap before
    the global reference stream. This is intentionally handled after parsing;
    the stream is validated against the header counts and tile index boundary.
    """
    objects: list[ANZObject] = []
    offset = 0x30

    for object_number in range(object_count):
        checked_slice(data, offset, OBJECT_HEADER_SIZE, "Objekt-Header")
        h = struct.unpack_from("<7I", data, offset)
        object_id, _zero, record_count, has_palette, max_tiles, _zero2, _zero3 = h
        offset += OBJECT_HEADER_SIZE

        if record_count > 100000:
            raise ANZError(f"Unplausible record_count in object {object_id}: {record_count}")
        if max_tiles > 100000:
            raise ANZError(f"Unplausible max_tiles in object {object_id}: {max_tiles}")

        records_raw = []
        for _ in range(record_count):
            checked_slice(data, offset, RECORD_SIZE, "Frame-Record")
            vals = struct.unpack_from("<13I", data, offset)
            records_raw.append(vals)
            offset += RECORD_SIZE

        palette = b""
        if has_palette:
            palette = checked_slice(data, offset, PALETTE_BYTES, "Palette")
            offset += PALETTE_BYTES

        trailer_size = max_tiles * 28 + 4
        trailer = checked_slice(data, offset, trailer_size, "Objekt-Trailer")
        offset += trailer_size

        records: list[FrameRecord] = []
        for frame_index, vals in enumerate(records_raw):
            raw_cols = vals[0]
            tile_refs = vals[1]
            # Some environment resources contain descriptor-only records with
            # a signed value in word 0 and no tile references. They describe
            # placement/animation metadata and are not raster frames.
            descriptor_only = tile_refs == 0
            cols = 0 if descriptor_only else raw_cols
            if not descriptor_only and cols == 0:
                raise ANZError(f"Object {object_id}, frame {frame_index}: cols=0")
            if not descriptor_only and tile_refs % cols != 0:
                raise ANZError(
                    f"Object {object_id}, frame {frame_index}: tile_refs={tile_refs}, cols={cols}"
                )
            # vals[3:7] are the four float bit-patterns.
            bbox = tuple(
                struct.unpack("<f", struct.pack("<I", v))[0] for v in vals[3:7]
            )  # type: ignore[assignment]
            records.append(
                FrameRecord(
                    index=frame_index,
                    cols=cols,
                    tile_refs=tile_refs,
                    bbox=bbox,  # type: ignore[arg-type]
                    subindex=vals[8],
                    refs=[],
                    pixel_width=(u32(trailer, frame_index * 28 + 12) if frame_index < max_tiles else 0),
                    pixel_height=(u32(trailer, frame_index * 28 + 16) if frame_index < max_tiles else 0),
                )
            )

        objects.append(
            ANZObject(
                object_id=object_id,
                records=records,
                max_tiles=max_tiles,
                palette=palette,
            )
        )

    return objects, offset


def parse_tiles(
    data: bytes,
    tile_index_offset: int,
    tile_count: int,
    tile_width: int,
    tile_height: int,
    bpp: int,
) -> tuple[list[bytes], int, int]:
    """Decode the indexed zlib tiles.

    Every index entry is ``(relative_stream_offset, compressed_length)``. The
    offset is relative to the start of the index table and already points at the
    zlib stream; there is no additional per-tile prefix.
    """
    index_size = tile_count * 8
    checked_slice(data, tile_index_offset, index_size, "Tile-Index")
    entries = [
        struct.unpack_from("<2I", data, tile_index_offset + i * 8)
        for i in range(tile_count)
    ]

    if entries[0][0] != index_size:
        raise ANZError(
            f"Tile-Index stimmt nicht: erster Offset {entries[0][0]} != Indexgröße {index_size}"
        )

    tile_data_offset = tile_index_offset + index_size
    tiles: list[bytes] = []
    for i in range(tile_count):
        rel, compressed_len = entries[i]
        if rel < index_size:
            raise ANZError(f"Tile {i}: Offset zeigt in den Indexbereich: {rel}")

        stream_start = tile_index_offset + rel
        if stream_start > len(data):
            raise ANZError(f"Tile {i}: Streamstart außerhalb der Datei")

        if i < tile_count - 1:
            stream_end = stream_start + compressed_len
            checked_slice(data, stream_start, compressed_len, f"Tile {i} zlib")
            stream = data[stream_start:stream_end]
            raw = zlib.decompress(stream)
        else:
            # The final ANZ tile may be followed by a small opaque tail.
            dec = zlib.decompressobj()
            raw = dec.decompress(data[stream_start:])
            raw += dec.flush()
            tail_size = len(dec.unused_data)
            # For the current format, the reference/index structure is already
            # validated above; the tail is not needed to render the sprites.

        expected_size = tile_width * tile_height * (bpp // 8)
        if len(raw) != expected_size:
            raise ANZError(
                f"Tile {i}: zlib ergab {len(raw)} Bytes statt {expected_size}"
            )
        tiles.append(raw)

    # The caller wants the end of the actual zlib data, not the opaque tail.
    # Re-run the final stream boundary calculation only for reporting.
    final_rel = entries[-1][0]
    final_stream_start = tile_index_offset + final_rel
    dec = zlib.decompressobj()
    dec.decompress(data[final_stream_start:])
    dec.flush()
    tail_size = len(dec.unused_data)
    return tiles, tile_data_offset, tail_size


def decode_anz_data(data: bytes, path: Path) -> ANZFile:
    if len(data) < 0x30:
        raise ANZError("Datei ist zu klein für einen ANZ-Header")
    if data[:4] != MAGIC:
        raise ANZError(f"Kein ANZ-Header: {data[:4]!r}")

    version = u32(data, 0x04)
    object_count = u32(data, 0x08)
    reference_count = u32(data, 0x0C)
    tile_size = u32(data, 0x14)
    bpp = u32(data, 0x1C)
    tile_count = u32(data, 0x20)
    tile_w = u32(data, 0x24)
    tile_h = u32(data, 0x28)

    if version != EXPECTED_VERSION:
        print(
            f"Warnung: ANZ-Version 0x{version:X}, erwartet 0x{EXPECTED_VERSION:X}",
            file=sys.stderr,
        )
    if tile_size != tile_w or tile_w <= 0 or tile_h <= 0:
        raise ANZError(
            f"Nicht unterstützte Tilegröße: header tile_size={tile_size}, "
            f"tile_w={tile_w}, tile_h={tile_h}"
        )
    if bpp not in (8, 32):
        raise ANZError(f"Nicht unterstützte Farbtiefe: {bpp} bpp")
    if object_count == 0 or tile_count == 0:
        raise ANZError("ANZ enthält keine Objekte oder Tiles")

    objects, object_end = parse_objects(data, object_count)

    # The final four bytes parsed as part of the descriptor boundary are actually
    # the first tile reference. Starting eight bytes early inserted a spurious
    # NULL tile and shifted every subsequent frame by one tile.
    reference_offset = object_end - 4
    if reference_offset < 0:
        raise ANZError("Ungültiger Reference-Offset")
    refs_bytes = reference_count * 4
    reference_end = reference_offset + refs_bytes
    if reference_end > len(data):
        raise ANZError("Reference-Stream reicht über die Dateigrenze hinaus")

    refs = list(struct.unpack_from(f"<{reference_count}I", data, reference_offset))
    invalid_refs = [i for i, r in enumerate(refs) if r != NULL_TILE and r >= tile_count]
    if invalid_refs:
        raise ANZError(
            f"{len(invalid_refs)} ungültige Tile-Referenzen, erste bei Ref {invalid_refs[0]}"
        )

    expected_refs = sum(r.tile_refs for o in objects for r in o.records)
    if expected_refs != reference_count:
        raise ANZError(
            f"Reference-Count mismatch: Frames benötigen {expected_refs}, Header meldet {reference_count}"
        )

    ref_cursor = 0
    for obj in objects:
        for frame in obj.records:
            if frame.tile_refs == 0:
                continue
            frame.refs = refs[ref_cursor : ref_cursor + frame.tile_refs]
            ref_cursor += frame.tile_refs
    if ref_cursor != reference_count:
        raise ANZError("Reference-Stream wurde nicht vollständig verbraucht")

    tile_index_offset = reference_end
    tile_index_size = tile_count * 8
    checked_slice(data, tile_index_offset, tile_index_size, "Tile-Index")
    if tile_index_offset + tile_index_size > len(data):
        raise ANZError("Tile-Index außerhalb der Datei")

    # The first index pair is additionally a very strong structural check.
    first_a, first_b = struct.unpack_from("<2I", data, tile_index_offset)
    if first_a != tile_index_size:
        # A helpful fallback: some files may contain the special final-boundary
        # overlap. Do not silently accept an unrelated layout.
        raise ANZError(
            f"Unerwarteter Tile-Index: first=(0x{first_a:X},0x{first_b:X}), "
            f"Indexgröße=0x{tile_index_size:X}"
        )

    # All objects in the sample share the same 256-entry palette.
    unique_palettes = {o.palette for o in objects if o.palette}
    if bpp == 8 and len(unique_palettes) != 1:
        print(
            f"Warnung: {len(unique_palettes)} verschiedene Objekt-Paletten gefunden",
            file=sys.stderr,
        )
    palette = next((o.palette for o in objects if o.palette), b"")

    tiles, tile_data_offset, tail_size = parse_tiles(
        data, tile_index_offset, tile_count, tile_w, tile_h, bpp
    )

    return ANZFile(
        path=path,
        version=version,
        object_count=object_count,
        reference_count=reference_count,
        tile_size=tile_size,
        tile_width=tile_w,
        tile_height=tile_h,
        bpp=bpp,
        tile_count=tile_count,
        tile_index_offset=tile_index_offset,
        tile_data_offset=tile_data_offset,
        reference_offset=reference_offset,
        tail_size=tail_size,
        objects=objects,
        refs=refs,
        tiles=tiles,
        palette=palette,
    )


def decode_anz(path: Path) -> ANZFile:
    return decode_anz_data(path.read_bytes(), path)


def decode_ana(path: Path) -> list[tuple[int, bytes]]:
    """Return the embedded ``(resource_id, ANZ bytes)`` entries of an ANA archive."""
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"ANA ":
        raise ANZError(f"Kein ANA-Header: {data[:4]!r}")
    version, count = struct.unpack_from("<2I", data, 4)
    if version != 0x100:
        print(f"Warnung: ANA-Version 0x{version:X}, erwartet 0x100", file=sys.stderr)
    table_end = 12 + count * 12
    checked_slice(data, 12, count * 12, "ANA-Inhaltstabelle")
    entries = [struct.unpack_from("<3I", data, 12 + i * 12) for i in range(count)]
    result: list[tuple[int, bytes]] = []
    previous_offset = table_end
    for i, (resource_id, offset, reserved) in enumerate(entries):
        end = entries[i + 1][1] if i + 1 < count else len(data)
        if reserved != 0:
            raise ANZError(f"ANA-Eintrag {i}: reserviertes Feld ist {reserved}")
        if offset < table_end or offset < previous_offset or end <= offset or end > len(data):
            raise ANZError(f"ANA-Eintrag {i}: ungültiger Bereich 0x{offset:X}..0x{end:X}")
        payload = data[offset:end]
        if payload[:4] != MAGIC:
            raise ANZError(f"ANA-Eintrag {i} ({resource_id}): kein eingebetteter ANZ-Header")
        result.append((resource_id, payload))
        previous_offset = offset
    return result


def palette_rgba(palette: bytes) -> list[tuple[int, int, int, int]]:
    # The file stores little-endian A8R8G8B8 DWORDs, hence the byte order in
    # memory/on disk is B, G, R, A.
    colors = [
        (palette[i + 2], palette[i + 1], palette[i], palette[i + 3])
        for i in range(0, len(palette), 4)
    ]
    if len(colors) != 256:
        raise ANZError("Palette hat nicht 256 Einträge")
    # Index 0 is the transparent/null pixel in the supplied ANZ format. Its RGB
    # bytes are irrelevant, but preserving them in the palette metadata is useful.
    colors[0] = (colors[0][0], colors[0][1], colors[0][2], 0)
    return colors


def build_tile_images(
    tiles: list[bytes], palette: bytes, tile_width: int, tile_height: int, bpp: int
) -> list[Image.Image]:
    result: list[Image.Image] = []
    if bpp == 32:
        for raw in tiles:
            result.append(
                Image.frombytes("RGBA", (tile_width, tile_height), raw, "raw", "BGRA")
            )
        return result

    colors = palette_rgba(palette)
    for raw in tiles:
        rgba = bytearray(len(raw) * 4)
        j = 0
        for idx in raw:
            c = colors[idx]
            rgba[j : j + 4] = bytes(c)
            j += 4
        result.append(Image.frombytes("RGBA", (tile_width, tile_height), bytes(rgba)))
    return result


def render_frame(
    frame: FrameRecord,
    tile_images: list[Image.Image],
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    rows = frame.tile_refs // frame.cols
    out = Image.new(
        "RGBA", (frame.cols * tile_width, rows * tile_height), (0, 0, 0, 0)
    )
    blank = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
    for k, ref in enumerate(frame.refs):
        tile = blank if ref == NULL_TILE else tile_images[ref]
        x = (k % frame.cols) * tile_width
        y = (k // frame.cols) * tile_height
        out.alpha_composite(tile, (x, y))
    # The tile grid is padded to complete tiles. The object trailer stores the
    # real frame dimensions; keeping the padded area caused adjacent sprite
    # fragments and oversized atlas cells in the original decoder.
    if 0 < frame.pixel_width <= out.width and 0 < frame.pixel_height <= out.height:
        out = out.crop((0, 0, frame.pixel_width, frame.pixel_height))
    return out


def write_frames(anz: ANZFile, out_root: Path) -> list[tuple[int, int, Path, Image.Image]]:
    frames_dir = out_root / "objects"
    frames_dir.mkdir(parents=True, exist_ok=True)
    palette_cache: dict[bytes, list[Image.Image]] = {}

    rendered: list[tuple[int, int, Path, Image.Image]] = []
    for obj in anz.objects:
        # Palettes belong to objects. Using only the first object's palette made
        # many otherwise correct sprites appear blue.
        tile_images = palette_cache.get(obj.palette)
        if tile_images is None:
            tile_images = build_tile_images(
                anz.tiles, obj.palette, anz.tile_width, anz.tile_height, anz.bpp
            )
            palette_cache[obj.palette] = tile_images
        obj_dir = frames_dir / str(obj.object_id)
        obj_dir.mkdir(parents=True, exist_ok=True)
        for frame in obj.records:
            if frame.tile_refs == 0:
                continue
            image = render_frame(
                frame, tile_images, anz.tile_width, anz.tile_height
            )
            path = obj_dir / f"frame_{frame.index:03d}.png"
            image.save(path)
            rendered.append((obj.object_id, frame.index, path, image))
    return rendered


def write_atlas(rendered: list[tuple[int, int, Path, Image.Image]], path: Path) -> None:
    """Create a readable atlas without changing the sprite pixels.

    Each cell is 128x128. Frames are centered and nearest-neighbour scaled down
    only when a frame exceeds the cell; no interpolation is used.
    """
    if not rendered:
        return
    cell_w = 128
    cell_h = 128
    columns = 12
    rows = math.ceil(len(rendered) / columns)
    atlas = Image.new("RGBA", (columns * cell_w, rows * cell_h), (70, 70, 70, 255))
    draw = ImageDraw.Draw(atlas)

    for i, (object_id, frame_index, _frame_path, image) in enumerate(rendered):
        if image.width > cell_w or image.height > cell_h:
            scale = min(cell_w / image.width, cell_h / image.height)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.NEAREST,
            )
        x0 = (i % columns) * cell_w
        y0 = (i // columns) * cell_h
        x = x0 + (cell_w - image.width) // 2
        y = y0 + (cell_h - image.height) // 2
        atlas.alpha_composite(image, (x, y))
        draw.text((x0 + 2, y0 + 2), f"{object_id}:{frame_index}", fill=(255, 255, 255, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(path)


def write_manifest(anz: ANZFile, out_root: Path) -> None:
    manifest = {
        "source": str(anz.path),
        "version": anz.version,
        "object_count": anz.object_count,
        "reference_count": anz.reference_count,
        "tile_size": anz.tile_size,
        "tile_width": anz.tile_width,
        "tile_height": anz.tile_height,
        "bpp": anz.bpp,
        "tile_count": anz.tile_count,
        "reference_offset": anz.reference_offset,
        "tile_index_offset": anz.tile_index_offset,
        "tile_data_offset": anz.tile_data_offset,
        "opaque_tail_bytes": anz.tail_size,
        "objects": [
            {
                "id": obj.object_id,
                "record_count": len(obj.records),
                "max_tiles": obj.max_tiles,
                "frames": [
                    {
                        "index": f.index,
                        "cols": f.cols,
                        "rows": (f.tile_refs // f.cols if f.cols else 0),
                        "descriptor_only": f.tile_refs == 0,
                        "tile_refs": f.tile_refs,
                        "pixel_width": f.pixel_width,
                        "pixel_height": f.pixel_height,
                        "bbox": list(f.bbox),
                        "subindex": f.subindex,
                    }
                    for f in obj.records
                ],
            }
            for obj in anz.objects
        ],
        "palette_unique_entries": (
            len(set(anz.palette[i : i + 4] for i in range(0, len(anz.palette), 4)))
            if anz.palette else 0
        ),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def iter_graphic_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".anz", ".ana"}:
            raise ANZError(f"Eingabedatei ist keine .anz- oder .ana-Datei: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise ANZError(f"Eingabepfad existiert nicht: {input_path}")

    yield from sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".anz", ".ana"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stone Age 2 .anz/.ana -> PNG Konverter")
    parser.add_argument(
        "input",
        type=Path,
        help=".anz/.ana-Datei oder Ordner mit solchen Dateien (rekursiv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("anz_png"),
        help="Ausgabeordner (Standard: ./anz_png)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Zusätzlich einen Atlas pro .anz-Datei erzeugen",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Keine einzelnen Frames schreiben (nur zusammen mit --atlas sinnvoll)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="manifest.json nicht schreiben",
    )
    args = parser.parse_args()

    files = list(iter_graphic_files(args.input))
    if not files:
        print("Keine .anz- oder .ana-Dateien gefunden.", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    success = 0
    failures = 0

    for path in files:
        try:
            if args.input.is_dir():
                source_out = args.output / path.relative_to(args.input).parent / path.stem
            else:
                source_out = args.output / path.stem

            if path.suffix.lower() == ".ana":
                embedded = decode_ana(path)
                print(f"[ANA] {path}: {len(embedded)} eingebettete ANZ-Dateien")
                jobs = [
                    (payload, Path(f"{path.name}/{resource_id}.anz"), source_out / str(resource_id))
                    for resource_id, payload in embedded
                ]
            else:
                print(f"[ANZ] {path}")
                jobs = [(path.read_bytes(), path, source_out)]

            for payload, virtual_path, file_out in jobs:
                anz = decode_anz_data(payload, virtual_path)
                print(
                    f"      [{anz.path.stem}] {anz.object_count} Objekte, "
                    f"{sum(len(o.records) for o in anz.objects)} Frames, "
                    f"{anz.tile_count} Tiles, {anz.reference_count} Referenzen"
                )
                file_out.mkdir(parents=True, exist_ok=True)

                rendered: list[tuple[int, int, Path, Image.Image]] = []
                if not args.no_frames:
                    rendered = write_frames(anz, file_out)
                    print(f"      PNG-Frames: {len(rendered)}")
                elif args.atlas:
                    palette_cache: dict[bytes, list[Image.Image]] = {}
                    for obj in anz.objects:
                        tile_images = palette_cache.get(obj.palette)
                        if tile_images is None:
                            tile_images = build_tile_images(
                                anz.tiles, obj.palette, anz.tile_width, anz.tile_height, anz.bpp
                            )
                            palette_cache[obj.palette] = tile_images
                        for frame in obj.records:
                            if frame.tile_refs == 0:
                                continue
                            rendered.append(
                                (
                                    obj.object_id,
                                    frame.index,
                                    Path(),
                                    render_frame(
                                        frame, tile_images, anz.tile_width, anz.tile_height
                                    ),
                                )
                            )

                if args.atlas:
                    atlas_path = file_out / "atlas.png"
                    write_atlas(rendered, atlas_path)
                    print(f"      Atlas: {atlas_path}")

                if not args.no_manifest:
                    write_manifest(anz, file_out)

                success += 1
        except Exception as exc:
            failures += 1
            print(f"      FEHLER: {exc}", file=sys.stderr)

    print(f"Fertig: {success} erfolgreich, {failures} fehlgeschlagen.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
