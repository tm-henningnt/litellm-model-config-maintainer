"""Apply this instance's litellm source patches to an installed litellm.

Two defects in litellm break Aliases this instance offers. Each has a
one-place fix in litellm's own transform layer, which no callback or
config setting reaches. `docs/gotchas.md` records both, with the
reasoning. `litellm_maintainer/litellm_patches.py` holds the markers
`doctor` reads to prove the fix is present.

The image build runs this file. A failure fails the build, so a proxy
image never ships without both fixes. That is the container's answer to
the trap the host install has: an upgrade there removes both edits and
nothing reports the loss.

Run with `--verify` to check the markers without editing. Run with no
argument to patch, then verify.

Every edit states the exact text it expects and how many times. A count
that does not match raises, because litellm moved the code and a silent
partial patch is worse than a failed build.
"""

from __future__ import annotations

import pathlib
import sys

# --- chatgpt_stream ---------------------------------------------------
# A provider whose transport only streams sets `stream` inside
# `transform_responses_api_request`. litellm then reads that flag back as
# the caller's intent, so a caller that asked for one JSON object gets an
# SSE body. Keep the caller's intent in its own variable.

STREAM_CAPTURE_OLD = '''        # Check if streaming is requested
        stream = response_api_optional_request_params.get("stream", False)

        api_base: Final = responses_api_provider_config.get_complete_url('''

STREAM_CAPTURE_NEW = '''        # Check if streaming is requested
        stream = response_api_optional_request_params.get("stream", False)
        # A provider whose transport only streams (chatgpt) sets stream in
        # transform_responses_api_request. That is a transport requirement,
        # not a request for a streamed response. Keep the caller's intent
        # separate, or a caller that asked for one JSON object gets SSE.
        client_requested_stream = bool(stream)

        api_base: Final = responses_api_provider_config.get_complete_url('''

STREAM_USE_OLD = "        is_stream_request: Final = bool(stream)\n"
STREAM_USE_NEW = "        is_stream_request: Final = client_requested_stream\n"

# --- usage_only_chunk -------------------------------------------------
# A provider may end a stream with a chunk whose `choices` list is empty,
# carrying only `usage`. litellm indexes `chunk.choices[0]` before it
# checks the merge path, so the chunk raises IndexError and kills the SSE
# generator. Merge first, translate second, and skip a chunk with no
# choices.

MERGE_ORDER_OLD = '''                )
                is_final_chunk = chunk.choices[0].finish_reason is not None
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                    applied_edits=(self.applied_edits if is_final_chunk and not will_merge_into_held else None),
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if will_merge_into_held:
                    merged_chunk = self._merge_usage_into_held_stop_reason_chunk(chunk)
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()
'''

MERGE_ORDER_NEW = '''                )
                # Merge a usage chunk into the held stop_reason chunk before any
                # translation. The translated chunk is discarded on this path, and
                # a provider may end the stream with a usage-only chunk carrying
                # no choices: translating that indexes choices[0], and the
                # IndexError kills the SSE generator before message_delta and
                # message_stop are ever emitted.
                if will_merge_into_held:
                    merged_chunk = self._merge_usage_into_held_stop_reason_chunk(chunk)
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # A usage-only chunk outside the merge path carries no content.
                if not chunk.choices:
                    continue

                is_final_chunk = chunk.choices[0].finish_reason is not None
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                    applied_edits=(self.applied_edits if is_final_chunk else None),
                )
'''

NEW_BLOCK_GUARD_OLD = '''        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
'''

NEW_BLOCK_GUARD_NEW = '''        # If chunk indicates a tool call
        # A usage-only chunk has no choices and starts no content block.
        if not chunk.choices or chunk.choices[0].finish_reason is not None:
'''

HTTP_HANDLER = "llms/custom_httpx/llm_http_handler.py"
STREAM_ITERATOR = "llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py"

# (relative path, old text, new text, expected occurrences)
EDITS: tuple[tuple[str, str, str, int], ...] = (
    (HTTP_HANDLER, STREAM_CAPTURE_OLD, STREAM_CAPTURE_NEW, 2),
    (HTTP_HANDLER, STREAM_USE_OLD, STREAM_USE_NEW, 2),
    (STREAM_ITERATOR, MERGE_ORDER_OLD, MERGE_ORDER_NEW, 2),
    (STREAM_ITERATOR, NEW_BLOCK_GUARD_OLD, NEW_BLOCK_GUARD_NEW, 1),
)

# The marker `doctor` reads, per file. Keep these equal to
# `REQUIRED_PATCHES` in `litellm_maintainer/litellm_patches.py`.
MARKERS: tuple[tuple[str, str], ...] = (
    (HTTP_HANDLER, "client_requested_stream"),
    (STREAM_ITERATOR, "if not chunk.choices"),
)


def litellm_root() -> pathlib.Path:
    """The installed litellm package directory."""
    import litellm

    return pathlib.Path(litellm.__file__).parent


def verify(root: pathlib.Path) -> list[str]:
    """Return one message per marker that is absent."""
    missing = []
    for relative_path, marker in MARKERS:
        text = (root / relative_path).read_text()
        if marker not in text:
            missing.append(f"{relative_path} does not carry {marker!r}")
    return missing


def apply(root: pathlib.Path) -> None:
    """Apply every edit. Raise when the expected text is not found."""
    for relative_path, old, new, expected in EDITS:
        path = root / relative_path
        text = path.read_text()
        if new in text and old not in text:
            print(f"already patched: {relative_path}")
            continue
        found = text.count(old)
        if found != expected:
            raise SystemExit(
                f"{relative_path}: expected {expected} occurrences of the "
                f"anchor, found {found}. litellm moved this code. Read "
                f"docs/gotchas.md and correct this file before you build."
            )
        path.write_text(text.replace(old, new))
        print(f"patched {relative_path} ({expected} site(s))")


def main() -> int:
    root = litellm_root()
    print(f"litellm at {root}")
    if "--verify" not in sys.argv:
        apply(root)
    missing = verify(root)
    if missing:
        for message in missing:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print("both patches present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
