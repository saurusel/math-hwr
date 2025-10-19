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
    def __init__(self, root: str):
        self.root = root
        self.by_token: Dict[str, List[str]] = {}
        for cls, tok in CLASS2TOKEN.items():
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            paths = []
            for fn in os.listdir(d):
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXT:
                    paths.append(os.path.join(d, fn))
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
