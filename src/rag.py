"""Recuperación híbrida sobre el corpus: embeddings multilingües + BM25 fusionados con RRF."""
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _leer(path):
    txt = path.read_text(encoding="utf-8")
    meta = {}
    if txt.startswith("---"):
        _, cab, cuerpo = txt.split("---", 2)
        for linea in cab.strip().splitlines():
            k, v = linea.split(":", 1)
            meta[k.strip()] = v.strip()
    else:
        cuerpo = txt
    return meta, cuerpo.strip()


def cargar_chunks():
    """Cada párrafo de cada documento es un chunk que conserva título, fuente y url."""
    chunks = []
    for p in sorted(CORPUS.glob("*.md")):
        meta, cuerpo = _leer(p)
        for i, parrafo in enumerate(x.strip() for x in cuerpo.split("\n\n") if x.strip()):
            chunks.append({"id": f"{p.stem}#{i + 1}", "doc": p.stem, "titulo": meta.get("titulo", p.stem),
                           "fuente": meta.get("fuente", ""), "url": meta.get("url", ""), "texto": parrafo})
    return chunks


STOP = set("""a al algo ante antes como con contra cual cuando de del desde donde e el ella ellas ellos en entre era es esa ese eso esta este esto estos fue ha hace hay la las le les lo los mas me mi mis muy no nos o para pero por porque que qué se sea segun ser si sin sobre son su sus tambien tengo tiene un una uno unos y ya yo cómo cuál cuándo dónde está están hoy""".split())


def tokenizar(t):
    # Raíz aproximada: los primeros 6 caracteres, así límite y límites coinciden
    return [w[:6] for w in re.findall(r"\w+", t.lower()) if w not in STOP]


class Buscador:
    def __init__(self, usar_embeddings=True):
        self.chunks = cargar_chunks()
        textos = [f"{c['titulo']}. {c['texto']}" for c in self.chunks]
        self.bm25 = BM25Okapi([tokenizar(t) for t in textos])
        self.modelo = None
        self.modo = "bm25"
        if usar_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.modelo = SentenceTransformer(MODELO_EMB)
                # El mismo modelo y la misma normalización se usan para documentos y consultas
                self.E = self.modelo.encode(textos, normalize_embeddings=True)
                self.modo = "hibrido"
            except Exception as e:  # sin internet o sin la librería se sigue solo con BM25
                print("Aviso: búsqueda solo con BM25 porque no cargó el modelo de embeddings:", e)

    def _rank_bm25(self, q):
        return list(np.argsort(-self.bm25.get_scores(tokenizar(q))))

    def _rank_vec(self, q):
        v = self.modelo.encode([q], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.E @ v)))

    def buscar(self, consulta, k=3):
        rankings = [self._rank_bm25(consulta)]
        if self.modelo is not None:
            rankings.append(self._rank_vec(consulta))
        puntaje = {}
        for r in rankings:
            for pos, idx in enumerate(r):
                puntaje[idx] = puntaje.get(idx, 0.0) + 1.0 / (60 + pos + 1)
        mejores = sorted(puntaje, key=puntaje.get, reverse=True)[:k]
        return [dict(self.chunks[i], score=round(puntaje[i], 4)) for i in mejores]


if __name__ == "__main__":
    b = Buscador()
    print(b.modo, len(b.chunks))
    for r in b.buscar("¿qué hago si el aire está naranja y tengo asma?"):
        print(r["id"], r["score"], r["texto"][:80])
