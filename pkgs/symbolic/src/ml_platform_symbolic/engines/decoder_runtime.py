"""Narrow opt-in hooks in pinned native decoders; absent callbacks leave logits untouched."""
import argparse
import hashlib
from pathlib import Path


def patch_source(source, engine):
    if engine == "e2e":
        class_name = "TransformerModel"
        anchors = [
            "            scores = self.proj(tensor)  # (bs, n_words)\n",
            "            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)\n",
        ]
        additions = [
            '            if getattr(self, "symbolic_operator_mask", None) is not None:\n'
            '                scores = self.symbolic_operator_mask(generated[:cur_len].transpose(0, 1), scores, max_len-cur_len)\n'
        ] * 2
    elif engine == "flash":
        class_name = "FlashANSRModel"
        anchors = [
            "                    if grammar is not None:\n                        # Hard grammar constraint: applied AFTER",
            "                    work = active_logits\n",
        ]
        additions = [
            '                    if getattr(self, "symbolic_operator_mask", None) is not None:\n'
            '                        next_token_logits = self.symbolic_operator_mask(None, next_token_logits)\n',
            '                    if getattr(self, "symbolic_operator_mask", None) is not None:\n'
            '                        work = self.symbolic_operator_mask(None, work)\n',
        ]
    else:
        raise ValueError("Unknown native decoder patch target")
    marker = f"{class_name}.symbolic_operator_mask_supported = True"
    if engine == "e2e" and marker in source and "symbolic_prefix_mask_version" not in source:
        old = ('            if getattr(self, "symbolic_operator_mask", None) is not None:\n'
               '                scores = self.symbolic_operator_mask(None, scores)\n')
        if source.count(old) != 2:
            raise ValueError("Incomplete legacy E2E decoder hook")
        source = source.replace(old, "").replace(
            "\n# Opt-in operator restriction; raw Flash/beam scores retained.\n" + marker + "\n", "")
    if engine == "e2e":
        marker += "\nTransformerModel.symbolic_prefix_mask_version = 2"
    replacements = [anchors[0] + additions[0], anchors[1] + additions[1]]
    if engine == "flash":
        replacements[0] = additions[0] + anchors[0]
    if marker in source:
        if source.count(marker) != 1 or any(source.count(s) != 1 for s in replacements):
            raise ValueError("Incomplete native operator-mask hook patch")
        return source
    if any(source.count(a) != 1 for a in anchors):
        raise ValueError("Native decoder differs from pinned reviewed hook locations")
    for before, after in zip(anchors, replacements, strict=True):
        source = source.replace(before, after)
    return source + "\n# Opt-in operator restriction; raw Flash/beam scores retained.\n" + marker + "\n"


def patch(path, engine):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if engine == "e2e":
        supported = {
            "dc1e373ad1052a916012e1d403d8bcd3091399ae3b72f10c3ad319903a71827c",
            "40e2b37713196a3ecf01f351cb01dda5f70728fae6d006e55ffd2811628d8821",
            "550b28db1764c69006a152615caf7b0a13e086ec163d897260314c9e4269ba24",
        }
        checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if checksum not in supported:
            raise ValueError("E2E decoder source SHA256 is not the reviewed pinned revision; rebuild the pinned profile: " + checksum)
    result = patch_source(source, engine)
    if result != source:
        path.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--engine", required=True, choices=["e2e", "flash"])
    args = parser.parse_args()
    patch(args.source, args.engine)
