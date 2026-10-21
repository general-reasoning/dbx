"""Fonts that draw every character they can find: a system font, and for what it lacks, the first of FALLBACKS that has it.

A pygame font draws a character it has no glyph for as a box -- Helvetica Neue
has no ``⊙``, ``∧``, ``∨``, ``→`` or ``⟨⟩``, so a transcript about logic or
physics came out in boxes. `Font` has a pygame font's interface -- `render`,
`size`, `get_linesize` and the rest -- and draws each run of characters its
font lacks in a fallback that has them, on the same baseline. Text all its
font has is drawn by that font alone, as before.

pygame says nothing of a missing glyph (``metrics`` answers for it as for any),
so a font is taken to lack a character when it draws it exactly as it draws
one no font has: its missing-glyph box.
"""

import pygame

#: Where a character the font lacks is looked for, in order: broad mathematical and symbol coverage first.
FALLBACKS = ("stixtwomath", "stixgeneral", "applesymbols", "menlo", "dejavusans", "notosanssymbols", "notosansmath")
#: A character no font has a glyph for: what a font draws for it is its missing-glyph box.
NO_GLYPH = "\U000F0000"


class Font:
    """A pygame font, *names* as `pygame.font.SysFont` takes them, that draws what it lacks in a fallback (`FALLBACKS`)."""

    def __init__(self, names, size, bold=False, italic=False):
        self.font = pygame.font.SysFont(names, size, bold=bold, italic=italic)
        self._size_, self._bold_, self._italic_ = size, bold, italic
        self._loaded_ = None  # the fallbacks, made when first needed: most text needs none
        self._has_ = {}  # (font index, character) -> whether it has a glyph for it; 0 is self.font
        self._boxes_ = {}  # font index -> its missing-glyph box, as bytes

    def __getattr__(self, name):
        return getattr(self.font, name)  # get_linesize, get_height, get_ascent, ...

    def size(self, text):
        runs = self._runs_(text)
        if len(runs) == 1:
            return runs[0][0].size(text)
        sizes = [f.size(piece) for f, piece in runs]
        return sum(w for w, _ in sizes), max(h for _, h in sizes)

    def render(self, text, antialias, color, background=None):
        runs = self._runs_(text)
        if len(runs) == 1:
            return runs[0][0].render(text, antialias, color, background)
        imgs = [(f, f.render(piece, antialias, color, background)) for f, piece in runs]
        ascent = max(f.get_ascent() for f, _ in imgs)
        h = max(ascent - f.get_ascent() + img.get_height() for f, img in imgs)
        out = pygame.Surface((sum(img.get_width() for _, img in imgs), h), pygame.SRCALPHA)
        x = 0
        for f, img in imgs:
            out.blit(img, (x, ascent - f.get_ascent()))  # on one baseline
            x += img.get_width()
        return out

    # 4. Helpers ----------------------------------------------------------------

    def _runs_(self, text):
        """*text* as runs, ``(pygame font, piece)``, each piece drawn by the first font that has its characters."""
        if text.isascii():
            return [(self.font, text)]
        runs = []
        for ch in text:
            f = self._font_for_(ch)
            if runs and runs[-1][0] is f:
                runs[-1] = (f, runs[-1][1] + ch)
            else:
                runs.append((f, ch))
        return runs or [(self.font, text)]

    def _font_for_(self, ch):
        if ch.isspace() or ch.isascii() or self._has_glyph_(0, ch):
            return self.font
        fallbacks = self._fallbacks_()
        for i, f in enumerate(fallbacks, start=1):
            if self._has_glyph_(i, ch):
                return f
        return self.font  # none has it: its box, as before

    def _fallbacks_(self):
        if self._loaded_ is None:
            self._loaded_ = []
            for name in FALLBACKS:
                path = pygame.font.match_font(name)
                if path:
                    f = pygame.font.Font(path, self._size_)
                    f.set_bold(self._bold_)
                    f.set_italic(self._italic_)
                    self._loaded_.append(f)
        return self._loaded_

    def _has_glyph_(self, i, ch):
        key = (i, ch)
        if key not in self._has_:
            f = self.font if i == 0 else self._loaded_[i - 1]
            if i not in self._boxes_:
                self._boxes_[i] = _drawn_(f, NO_GLYPH)
            self._has_[key] = _drawn_(f, ch) != self._boxes_[i]
        return self._has_[key]


def _drawn_(font, ch):
    return pygame.image.tobytes(font.render(ch, True, (255, 255, 255)), "RGBA")
