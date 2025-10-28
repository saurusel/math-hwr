import os, random
from typing import Dict, List, Optional
from PIL import Image

CLASS2TOKEN = {
    "0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9",
    "x":"x","y":"y","z":"z",
    "plus":"+",
    "minus":"-",
    "times":"*",
    "divide":"÷",
    "equals":"=",
    "dot":"·"  # centered multiplication dot
}

SUPPORTED_EXT = {".png",".jpg",".jpeg",".bmp",".webp"}

class SymbolBank:
    def __init__(self, root: str, validate_on_load: bool = True):
        """
        Load symbol bank with optional quality validation.

        Args:
            root: Path to symbols directory
            validate_on_load: If True, pre-filter bad quality symbols
        """
        self.root = root
        self.by_token: Dict[str, List[str]] = {}
        self.validate_on_load = validate_on_load

        # Import validator if needed
        if validate_on_load:
            try:
                from .symbol_validator import validate_symbol
                self._validator = validate_symbol
            except:
                self._validator = None
                validate_on_load = False

        for cls, tok in CLASS2TOKEN.items():
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            paths = []

            for fn in os.listdir(d):
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXT:
                    full_path = os.path.join(d, fn)

                    # Validate symbol quality if enabled
                    if validate_on_load and self._validator:
                        try:
                            img = Image.open(full_path).convert("L")
                            is_valid, _ = self._validator(img, tok,
                                                          max_aspect_ratio=3.0,
                                                          max_ink_coverage=0.35,  # STRICT for digits
                                                          min_ink_pixels=10)
                                                          # Operators/vars use adaptive (higher) limits
                            if not is_valid:
                                continue  # Skip this bad symbol
                        except:
                            continue  # Skip if can't load

                    paths.append(full_path)

            if paths:
                self.by_token.setdefault(tok, []).extend(paths)

    def has_token(self, tok: str) -> bool:
        return tok in self.by_token and len(self.by_token[tok])>0

    def sample(self, tok: str) -> Optional[Image.Image]:
        lst = self.by_token.get(tok)
        if not lst: return None
        path = random.choice(lst)
        img = Image.open(path).convert("L")
        return img
