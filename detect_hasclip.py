from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional, Tuple

import pikepdf
from pikepdf import Name, Pdf, Object, Dictionary, Array, Stream


TEXT_SHOW_OPS = {"Tj", "TJ", "'", '"'}   # text painting ops
CLIP_OPS = {"W", "W*"}                   # clip path operators

if hasattr(pikepdf, "parsecontent"):
    PIKEPDF_PARSE_CONTENT = pikepdf.parsecontent
elif hasattr(pikepdf, "parse_content_stream"):
    PIKEPDF_PARSE_CONTENT = pikepdf.parse_content_stream
else:
    PIKEPDF_PARSE_CONTENT = None


class GraphicsState:
    """
    Minimal graphics state to answer 'is there a clipping path active?'.
    Real PDFs have more, but this is enough to mimic HasClip.
    """
    __slots__ = ("clipped",)

    def __init__(self, clipped: bool = False):
        self.clipped = clipped

    def clone(self) -> "GraphicsState":
        return GraphicsState(self.clipped)


def _ensure_dictionary(obj: Any) -> Optional[Dictionary]:
    if isinstance(obj, Dictionary):
        return obj
    try:
        return Dictionary(obj)
    except Exception:
        return None


def page_operations(page: Dictionary):
    """
    Yield (operands, operator) tuples from a page's content streams.
    pikepdf normalizes page.Contents into a ContentStream with .operations().
    """
    # pikepdf.Page is just a thin wrapper around the dict; using dict is fine here.
    for operands, operator in page.contents.operations():
        yield operands, operator


def get_xobject(resources: Dictionary, name: Name) -> Optional[Stream]:
    """
    Resolve a Form XObject by name from the given resources dictionary.
    """
    if not resources:
        return None
    resources_dict = _ensure_dictionary(resources)
    if resources_dict is None:
        return None
    xobjs = resources_dict.get("/XObject", None)
    if not isinstance(xobjs, (Dictionary, Object)):
        return None
    if isinstance(xobjs, Object):
        try:
            xobjs = Dictionary(xobjs)
        except Exception:
            return None
    stream = xobjs.get(name, None)
    if stream is None:
        return None
    try:
        # Indirect objects need resolving
        return stream.get_object() if hasattr(stream, "get_object") else stream
    except Exception:
        return None


def xobject_resources(xobj: Stream, inherited: Optional[Dictionary]) -> Optional[Dictionary]:
    """
    Determine which /Resources apply inside the XObject.
    Form XObjects may have their own /Resources; otherwise inherit.
    """
    own = xobj.get("/Resources", None)
    own_dict = _ensure_dictionary(own)
    if own_dict is not None:
        return own_dict
    return _ensure_dictionary(inherited)


def _iter_content_operations(content_stream: Any):
    if PIKEPDF_PARSE_CONTENT is None:
        return
    if content_stream is None:
        return
    if isinstance(content_stream, (Array, list, tuple)):
        for item in content_stream:
            yield from _iter_content_operations(item)
        return
    try:
        instructions = PIKEPDF_PARSE_CONTENT(content_stream)
    except Exception:
        if isinstance(content_stream, Object):
            try:
                resolved = content_stream.get_object()
            except Exception:
                resolved = None
            if resolved is not None and resolved is not content_stream:
                yield from _iter_content_operations(resolved)
        return
    for instruction in instructions:
        yield instruction.operands, instruction.operator


def extract_text_from_operands(op: str, operands: Any) -> str:
    """
    Best-effort extraction of literal text from Tj/TJ/'/" operands.
    This is optional (for reporting). We do NOT decode with font encodings here.
    """
    if op == "Tj":
        # one string
        s = operands[0]
        return str(s)
    elif op == "TJ":
        # array of strings and spacings; join strings
        parts = []
        for item in operands[0]:
            # strings come as pikepdf.String or Python str
            if isinstance(item, (str, pikepdf.String)):
                parts.append(str(item))
        return "".join(parts)
    elif op == "'":
        # move to next line and show text (one string)
        return str(operands[0])
    elif op == '"':
        # set word & char spacing, move to next line and show text (third operand is string)
        return str(operands[2])
    return ""


def walk_content(
    pdf: Pdf,
    resources: Optional[Dictionary],
    content_stream: Stream | pikepdf.ContentStream,
    results: List[Dict[str, Any]],
    page_num: int,
    xobject_stack: List[str],
    inherited_clip: bool = False,
):
    """
    Walk a content stream, tracking clipping via q/Q and W/W*.
    Recurse into Form XObjects encountered via 'Do' operator.
    """
    state = GraphicsState(clipped=inherited_clip)
    stack: List[GraphicsState] = []

    for operands, operator in _iter_content_operations(content_stream):
        try:
            op_name = operator.name  # type: ignore[attr-defined]
        except Exception:
            op_name = str(operator)

        if op_name == "q":
            stack.append(state.clone())
        elif op_name == "Q":
            if stack:
                state = stack.pop()
            else:
                # Defensive: stray Q should not crash us
                state = GraphicsState(clipped=False)

        elif op_name in CLIP_OPS:
            # A clipping path is established for subsequent painting in THIS state.
            state.clipped = True

        elif op_name in TEXT_SHOW_OPS:
            # We are painting text now; record whether clip is active.
            text_snippet = extract_text_from_operands(op_name, operands)
            results.append(
                {
                    "page": page_num,
                    "operator": op_name,
                    "has_clip": state.clipped,
                    "text_preview": text_snippet[:60],  # preview only
                    "xobject_context": list(xobject_stack),  # where we were nested
                }
            )

        elif op_name == "Do":
            # Invoke XObject (could be an image or a Form XObject that itself contains text).
            xname = operands[0]
            if not isinstance(xname, Name):
                continue
            xobj = get_xobject(resources, xname) if resources is not None else None
            if not isinstance(xobj, Stream):
                continue

            subtype = xobj.get("/Subtype", None)
            if subtype == Name("/Form"):
                # Recurse into the form's content stream
                child_resources = xobject_resources(xobj, resources)
                xobject_stack.append(str(xname))
                try:
                    walk_content(
                        pdf=pdf,
                        resources=child_resources,
                        content_stream=xobj,
                        results=results,
                        page_num=page_num,
                        xobject_stack=xobject_stack,
                        inherited_clip=state.clipped,  # clipping survives into nested content
                    )
                finally:
                    xobject_stack.pop()

            # If it's an image XObject, we ignore (it doesn't paint text directly).

        # NOTE: We don't need to track the current "path" geometry here—only whether
        # a clip was *activated* (W/W*). The geometry would matter if you wanted to
        # prove the text actually intersects the clip region.


def analyze_pdf_hasclip(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Analyze an entire PDF and return a list of text-paint events with a has_clip flag.
    """
    if PIKEPDF_PARSE_CONTENT is None:
        raise RuntimeError("pikepdf does not expose a content parsing function")

    pdf = Pdf.open(pdf_path)
    results: List[Dict[str, Any]] = []

    for i, page in enumerate(pdf.pages, start=1):  # 1-based page numbering for readability
        resources = _ensure_dictionary(page.get("/Resources", None))
        content_stream = page.obj.get("/Contents")
        walk_content(
            pdf=pdf,
            resources=resources,
            content_stream=content_stream,
            results=results,
            page_num=i,
            xobject_stack=[],
            inherited_clip=False,
        )

    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_hasclip.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    rows = analyze_pdf_hasclip(pdf_path)

    # Simple human-readable report
    total = len(rows)
    clipped = sum(1 for r in rows if r["has_clip"])
    unclipped = total - clipped

    print(f"\nAnalyzed text-paint events: {total}")
    print(f"  with    clip: {clipped}")
    print(f"  without clip: {unclipped}\n")

    # Show a few examples
    print("Sample rows:")
    for r in rows[:20]:
        ctx = " -> ".join(r["xobject_context"]) if r["xobject_context"] else "(page stream)"
        preview = r["text_preview"].encode("utf-8", "replace").decode("utf-8")
        print(
            f"Page {r['page']:>3}  op={r['operator']:<2}  HasClip={str(r['has_clip']):<5}  "
            f"CTX={ctx:<20}  TEXT='{preview}'"
        )


if __name__ == "__main__":
    main()
