import re
from pathlib import Path
from typing import List, Set, Tuple, Optional
import xml.etree.ElementTree as ET

# Standard namespaces for Adobe Lightroom XMP
NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
}

# Register namespaces so ET doesn't write 'ns0:', 'ns1:', etc.
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

RDF_URI = NAMESPACES["rdf"]
DC_URI = NAMESPACES["dc"]
LR_URI = NAMESPACES["lr"]
X_URI = NAMESPACES["x"]

XPACKET_HEADER = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
XPACKET_FOOTER = '\n<?xpacket end="w"?>\n'


class XMPManager:
    """Reads and writes Adobe Lightroom-compatible XMP sidecar files in pure Python."""

    @staticmethod
    def get_sidecar_path(image_path: Path) -> Path:
        """Lightroom convention: <image_name>.xmp (or <image_name>.<ext>.xmp in some camera exports).
        Lightroom Classic standard for RAW is <image_name>.xmp.
        """
        return image_path.with_suffix(".xmp")

    @classmethod
    def read_tags(cls, xmp_path: Path) -> Tuple[List[str], List[str]]:
        """
        Reads existing flat tags and hierarchical tags from an XMP sidecar.
        Returns:
            (flat_tags, hierarchical_tags)
        """
        if not xmp_path.exists():
            return [], []

        content = xmp_path.read_text(encoding="utf-8", errors="replace")
        if "subject" not in content and "hierarchicalSubject" not in content:
            return [], []

        xml_content = cls._strip_xpacket(content)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return [], []

        flat_tags: List[str] = []
        hierarchical_tags: List[str] = []

        # Find dc:subject/rdf:Bag/rdf:li
        for subject in root.iter(f"{{{DC_URI}}}subject"):
            for li in subject.iter(f"{{{RDF_URI}}}li"):
                if li.text and li.text.strip():
                    flat_tags.append(li.text.strip())

        # Find lr:hierarchicalSubject/rdf:Bag/rdf:li
        for h_subj in root.iter(f"{{{LR_URI}}}hierarchicalSubject"):
            for li in h_subj.iter(f"{{{RDF_URI}}}li"):
                if li.text and li.text.strip():
                    hierarchical_tags.append(li.text.strip())

        return flat_tags, hierarchical_tags

    @classmethod
    def write_tags(
        cls,
        xmp_path: Path,
        new_flat_tags: List[str],
        new_hierarchical_tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Updates or creates an XMP sidecar file with tags.
        Preserves all existing Lightroom develop adjustments, ratings, labels, etc.
        Returns:
            (final_flat_tags, final_hierarchical_tags)
        """
        new_hierarchical_tags = new_hierarchical_tags or []

        if xmp_path.exists():
            final_flat, final_hier = cls._update_existing_xmp(
                xmp_path, new_flat_tags, new_hierarchical_tags, overwrite
            )
        else:
            final_flat, final_hier = cls._create_new_xmp(
                xmp_path, new_flat_tags, new_hierarchical_tags
            )

        return final_flat, final_hier

    @classmethod
    def _create_new_xmp(
        cls,
        xmp_path: Path,
        flat_tags: List[str],
        hierarchical_tags: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Creates a brand new Adobe-compliant XMP sidecar file."""
        root = ET.Element(
            f"{{{X_URI}}}xmpmeta",
            {f"{{{X_URI}}}xmptk": "Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        "},
        )
        rdf = ET.SubElement(root, f"{{{RDF_URI}}}RDF")
        desc = ET.SubElement(
            rdf,
            f"{{{RDF_URI}}}Description",
            {f"{{{RDF_URI}}}about": ""},
        )

        deduped_flat = cls._dedup_tags(flat_tags)
        deduped_hier = cls._dedup_tags(hierarchical_tags)

        if deduped_flat:
            dc_subj = ET.SubElement(desc, f"{{{DC_URI}}}subject")
            bag = ET.SubElement(dc_subj, f"{{{RDF_URI}}}Bag")
            for tag in deduped_flat:
                li = ET.SubElement(bag, f"{{{RDF_URI}}}li")
                li.text = tag

        if deduped_hier:
            lr_subj = ET.SubElement(desc, f"{{{LR_URI}}}hierarchicalSubject")
            bag = ET.SubElement(lr_subj, f"{{{RDF_URI}}}Bag")
            for tag in deduped_hier:
                li = ET.SubElement(bag, f"{{{RDF_URI}}}li")
                li.text = tag

        cls._save_xml(xmp_path, root)
        return deduped_flat, deduped_hier

    @classmethod
    def _update_existing_xmp(
        cls,
        xmp_path: Path,
        new_flat: List[str],
        new_hier: List[str],
        overwrite: bool,
    ) -> Tuple[List[str], List[str]]:
        """Safely updates an existing XMP sidecar without altering other metadata."""
        content = xmp_path.read_text(encoding="utf-8", errors="replace")
        if "subject" not in content and "hierarchicalSubject" not in content:
            return [], []

        xml_content = cls._strip_xpacket(content)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            # If corrupt, re-create fresh
            return cls._create_new_xmp(xmp_path, new_flat, new_hier)

        desc = cls._find_or_create_description(root)

        # Handle dc:subject
        current_flat: List[str] = []
        dc_subj = desc.find(f"{{{DC_URI}}}subject")
        if dc_subj is not None:
            bag = dc_subj.find(f"{{{RDF_URI}}}Bag")
            if bag is not None:
                for li in bag.findall(f"{{{RDF_URI}}}li"):
                    if li.text:
                        current_flat.append(li.text.strip())
                desc.remove(dc_subj)

        final_flat = new_flat if overwrite else cls._merge_tags(current_flat, new_flat)
        if final_flat:
            dc_subj = ET.SubElement(desc, f"{{{DC_URI}}}subject")
            bag = ET.SubElement(dc_subj, f"{{{RDF_URI}}}Bag")
            for tag in final_flat:
                li = ET.SubElement(bag, f"{{{RDF_URI}}}li")
                li.text = tag

        # Handle lr:hierarchicalSubject
        current_hier: List[str] = []
        lr_subj = desc.find(f"{{{LR_URI}}}hierarchicalSubject")
        if lr_subj is not None:
            bag = lr_subj.find(f"{{{RDF_URI}}}Bag")
            if bag is not None:
                for li in bag.findall(f"{{{RDF_URI}}}li"):
                    if li.text:
                        current_hier.append(li.text.strip())
                desc.remove(lr_subj)

        final_hier = new_hier if overwrite else cls._merge_tags(current_hier, new_hier)
        if final_hier:
            lr_subj = ET.SubElement(desc, f"{{{LR_URI}}}hierarchicalSubject")
            bag = ET.SubElement(lr_subj, f"{{{RDF_URI}}}Bag")
            for tag in final_hier:
                li = ET.SubElement(bag, f"{{{RDF_URI}}}li")
                li.text = tag

        cls._save_xml(xmp_path, root)
        return final_flat, final_hier

    @classmethod
    def _find_or_create_description(cls, root: ET.Element) -> ET.Element:
        """Finds or creates rdf:Description in the root element."""
        rdf = root.find(f"{{{RDF_URI}}}RDF")
        if rdf is None:
            # Check if root is already rdf:RDF
            if root.tag == f"{{{RDF_URI}}}RDF":
                rdf = root
            else:
                rdf = ET.SubElement(root, f"{{{RDF_URI}}}RDF")

        desc = rdf.find(f"{{{RDF_URI}}}Description")
        if desc is None:
            desc = ET.SubElement(rdf, f"{{{RDF_URI}}}Description", {f"{{{RDF_URI}}}about": ""})

        return desc

    @staticmethod
    def _strip_xpacket(content: str) -> str:
        """Strips the <?xpacket ... ?> wrappers before parsing XML."""
        content = re.sub(r"<\?xpacket begin=.*?\?>", "", content, flags=re.DOTALL)
        content = re.sub(r"<\?xpacket end=.*?\?>", "", content, flags=re.DOTALL)
        return content.strip()

    @classmethod
    def _save_xml(cls, xmp_path: Path, root: ET.Element) -> None:
        """Writes the XML with xpacket wrapping to a temporary file, then renames atomically."""
        ET.indent(root, space=" ", level=0)
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
        output_content = XPACKET_HEADER + xml_bytes.decode("utf-8") + XPACKET_FOOTER

        temp_path = xmp_path.with_suffix(".xmp.tmp")
        temp_path.write_text(output_content, encoding="utf-8")
        temp_path.replace(xmp_path)

    @staticmethod
    def _dedup_tags(tags: List[str]) -> List[str]:
        """Deduplicate tags preserving order and case-insensitive uniqueness."""
        seen = set()
        deduped = []
        for tag in tags:
            cleaned = tag.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(cleaned)
        return deduped

    @classmethod
    def _merge_tags(cls, existing: List[str], new_tags: List[str]) -> List[str]:
        """Merges new tags into existing ones without duplicate entries."""
        seen = {t.lower() for t in existing if t.strip()}
        merged = list(existing)
        for tag in new_tags:
            cleaned = tag.strip()
            if not cleaned:
                continue
            if cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                merged.append(cleaned)
        return merged
