"""Aura song analysis: genre, mood, tempo, key and energy for every song.

Runs on GitHub Actions after each YAYINLA.bat upload (see analiz.yml).
  1. Listens to each song with an AudioSet music model (MIT AST, BSD-3).
  2. Measures tempo, energy, brightness, beat strength and key with librosa.
  3. Turns that into one genre and up to three moods.
  4. Writes analiz.json (cache, so each song is analysed once) and
     updates catalog.json, which the app reads.

A genre chosen by the owner (sub folder in "muzikler") always wins.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

VERSION = 4  # bump to re-analyse every song
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 22050
AST_SR = 16000
WINDOW_S = 10.24
WINDOWS = 5

ROOT = Path(os.environ.get("AURA_ROOT", "."))
CATALOG = ROOT / "catalog.json"
ANALYSIS = ROOT / "analiz.json"

# --------------------------------------------------------------------------
# AudioSet label -> Aura genre. Weight < 1 for weaker hints.
GENRES: dict[str, list[tuple[str, float]]] = {
    "Pop": [("Pop music", 1.0), ("Independent music", 0.5), ("Song", 0.25)],
    "Rock": [("Rock music", 1.0), ("Rock and roll", 1.0), ("Progressive rock", 1.0), ("Psychedelic rock", 1.0),
             ("Grunge", 1.0), ("Punk rock", 1.0)],
    "Metal": [("Heavy metal", 1.0)],
    "Hip-Hop": [("Hip hop music", 1.0), ("Rapping", 0.7), ("Beatboxing", 0.4)],
    "R&B": [("Rhythm and blues", 1.0), ("Soul music", 1.0), ("Funk", 0.9), ("Gospel music", 0.7)],
    "Electronic": [("Electronic music", 0.9), ("Electronica", 1.0), ("Synthesizer", 0.35), ("Drum machine", 0.35)],
    "Dance": [("Electronic dance music", 1.0), ("House music", 1.0), ("Techno", 1.0), ("Trance music", 1.0),
              ("Dubstep", 1.0), ("Drum and bass", 1.0), ("Disco", 1.0), ("Dance music", 0.8)],
    "Jazz": [("Jazz", 1.0), ("Swing music", 1.0)],
    "Blues": [("Blues", 1.0)],
    "Classical": [("Classical music", 1.0), ("Opera", 1.0), ("Orchestra", 0.6), ("Chamber music", 1.0)],
    "Ambient": [("Ambient music", 1.0), ("New-age music", 1.0), ("Background music", 0.45), ("Drone", 0.5)],
    "Soundtrack": [("Soundtrack music", 1.0), ("Theme music", 0.8), ("Video game music", 0.9)],
    "Folk": [("Folk music", 1.0), ("Country", 1.0), ("Bluegrass", 1.0), ("Traditional music", 0.7)],
    "Latin": [("Music of Latin America", 1.0), ("Salsa music", 1.0), ("Flamenco", 1.0), ("Cumbia", 1.0)],
    "Reggae": [("Reggae", 1.0), ("Ska", 0.9)],
    "World": [("Middle Eastern music", 1.0), ("Music of Asia", 0.9), ("Music of Africa", 0.9), ("Afrobeat", 1.0),
              ("Carnatic music", 1.0), ("Music of Bollywood", 1.0)],
}

VOCALS = [("Singing", 1.0), ("Male singing", 1.0), ("Female singing", 1.0), ("Child singing", 1.0),
          ("Rapping", 1.0), ("Choir", 0.8), ("Vocal music", 0.8), ("A capella", 1.0), ("Speech", 0.5)]
NATURE = [("Stream", 1.0), ("Rain", 1.0), ("Raindrop", 1.0), ("Waves, surf", 1.0), ("Ocean", 1.0),
          ("Bird", 1.0), ("Bird vocalization, bird call, bird song", 1.0), ("Chirp, tweet", 1.0),
          ("Wind", 0.8), ("Rustling leaves", 1.0), ("Water", 0.7), ("Thunderstorm", 1.0), ("Insect", 0.7),
          ("Cricket", 0.9), ("Frog", 0.8), ("Owl", 1.0)]

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# --------------------------------------------------------------------------
class Listener:
    """The AudioSet model. If it cannot load, analysis still works on audio features."""

    def __init__(self) -> None:
        self.ok = False
        try:
            import torch
            from transformers import ASTFeatureExtractor, ASTForAudioClassification

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            self.torch = torch
            self.fe = ASTFeatureExtractor.from_pretrained(MODEL)
            self.model = ASTForAudioClassification.from_pretrained(MODEL).eval()
            labels = self.model.config.id2label
            self.index = {str(v).strip().lower(): int(k) for k, v in labels.items()}
            self.ok = True
            print(f"Model ready: {MODEL} ({len(self.index)} labels)")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: music model not available, using audio features only: {e}")

    def probs(self, y16: np.ndarray) -> dict[str, float]:
        """Average label probabilities over several 10 s windows across the song."""
        if not self.ok:
            return {}
        n = int(WINDOW_S * AST_SR)
        total = len(y16)
        if total <= n:
            chunks = [y16]
        else:
            start, end = int(total * 0.08), int(total * 0.92) - n
            if end <= start:
                start, end = 0, total - n
            chunks = [y16[int(s): int(s) + n] for s in np.linspace(start, end, WINDOWS)]
        out = np.zeros(len(self.index))
        with self.torch.no_grad():
            for c in chunks:  # one at a time keeps memory low
                inputs = self.fe(c, sampling_rate=AST_SR, return_tensors="pt")
                logits = self.model(**inputs).logits[0]
                out += self.torch.sigmoid(logits).numpy()
        out /= len(chunks)
        return {name: float(out[i]) for name, i in self.index.items()}


def score(p: dict[str, float], spec: list[tuple[str, float]]) -> float:
    return max((p.get(name.lower(), 0.0) * w for name, w in spec), default=0.0)


# --------------------------------------------------------------------------
def features(y: np.ndarray) -> dict[str, float | str]:
    import librosa

    dur = len(y) / SR
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    loud = rms[rms > 1e-4]
    db = 20 * math.log10(float(np.mean(loud)) + 1e-9) if len(loud) else -60.0
    dyn = float(np.percentile(20 * np.log10(loud + 1e-9), 95) - np.percentile(20 * np.log10(loud + 1e-9), 10)) if len(loud) else 0.0

    harm, perc = librosa.effects.hpss(y)
    perc_ratio = float(np.sum(perc ** 2) / (np.sum(y ** 2) + 1e-9))

    hop = 256
    onset = librosa.onset.onset_strength(y=perc, sr=SR, hop_length=hop)
    onset_rate = float(len(librosa.onset.onset_detect(onset_envelope=onset, sr=SR, hop_length=hop)) / max(dur, 1))

    # Beat strength: how strongly the onset curve repeats at a tempo lag,
    # scaled by how much percussion there is (pads and drones repeat too).
    ac = librosa.autocorrelate(onset - onset.mean(), max_size=int(4 * SR / hop))
    lo, hi = int(0.25 * SR / hop), int(1.6 * SR / hop)
    pulse = clip(float(np.max(ac[lo:hi]) / (ac[0] + 1e-9))) if ac[0] > 0 else 0.0
    beatiness = clip(pulse * clip(perc_ratio / 0.2))

    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=SR, hop_length=hop,
                                                      start_bpm=110, aggregate=np.median))[0])
    while tempo and tempo < 70:
        tempo *= 2
    while tempo > 180:
        tempo /= 2

    # Refine tempo and find where the beats fall, so the app's visuals hit
    # the real beats: fit a straight line through the tracked beat times.
    offset = 0.0
    try:
        # Kick drum band: beats land on the kick, not on the off-beat hi-hats.
        low = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop, fmax=250, n_mels=32)
        _, beats = librosa.beat.beat_track(onset_envelope=low, sr=SR, hop_length=hop, bpm=tempo, units="time")
        beats = np.asarray(beats, dtype=float)
        if len(beats) >= 16:
            period = 60.0 / tempo
            idx = np.round((beats - beats[0]) / period)
            slope, intercept = np.polyfit(idx, beats, 1)
            if 0.85 * period < slope < 1.15 * period:
                tempo = 60.0 / float(slope)
                offset = float((intercept - 0.015) % slope)  # onset curve runs ~15 ms late
    except Exception:  # noqa: BLE001
        pass

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR).mean(axis=1)
    best = (-2.0, "C", "major")
    major_best, minor_best = -2.0, -2.0
    for i in range(12):
        rot = np.roll(chroma, -i)
        cm = float(np.corrcoef(rot, MAJOR)[0, 1])
        cn = float(np.corrcoef(rot, MINOR)[0, 1])
        major_best, minor_best = max(major_best, cm), max(minor_best, cn)
        if cm > best[0]:
            best = (cm, NOTES[i], "major")
        if cn > best[0]:
            best = (cn, NOTES[i], "minor")
    majorness = clip(0.5 + (major_best - minor_best) * 2.5)

    # Perceived energy: loudness, rhythmic activity and brightness together.
    loud_n = clip((db + 32) / 22)
    energy = clip(0.30 * loud_n + 0.40 * clip(perc_ratio / 0.3) + 0.15 * clip(onset_rate / 5) + 0.15 * clip(centroid / 3500))

    return {
        "seconds": dur, "db": db, "dynamics": dyn, "perc": perc_ratio, "onsetRate": onset_rate,
        "pulse": pulse, "beatiness": beatiness, "tempo": tempo, "beatOffset": offset, "centroid": centroid, "flatness": flat, "energy": energy,
        "major": majorness, "key": f"{best[1]} {best[2]}", "keyConf": best[0],
    }


# --------------------------------------------------------------------------
MOOD_ORDER = ["Energize", "Workout", "Party", "Happy", "Chill", "Focus", "Sleep", "Nature",
              "Late night", "Romantic", "Melancholy"]


def decide(f: dict, p: dict[str, float]) -> dict:
    e, bpm, pulse = f["energy"], f["tempo"], f["beatiness"]
    bright = clip(f["centroid"] / 3500)
    major = f["major"]
    has_model = bool(p)

    g = {name: score(p, spec) for name, spec in GENRES.items()} if has_model else {}
    vocal = score(p, VOCALS) if has_model else 0.0
    nature = score(p, NATURE) if has_model else 0.0
    happy = p.get("happy music", 0.0)
    sad = p.get("sad music", 0.0)
    tender = p.get("tender music", 0.0)
    exciting = p.get("exciting music", 0.0)
    angry = p.get("angry music", 0.0)
    lullaby = p.get("lullaby", 0.0)

    # ---- genre
    ranked = sorted(g.items(), key=lambda kv: kv[1], reverse=True)
    genre, conf = (ranked[0] if ranked else ("", 0.0))
    # Lo-fi: relaxed hip-hop / jazz / electronic beats, slow and warm.
    if genre in ("Hip-Hop", "Jazz", "Electronic") and 60 <= bpm <= 96 and e < 0.55 and bright < 0.75 and vocal < 0.3:
        genre = "Lo-fi"
    if not genre or conf < 0.06:
        # No clear answer from the model: judge by sound.
        if e < 0.42 and pulse < 0.45:
            genre = "Ambient"
        elif pulse > 0.55 and bpm >= 115:
            genre = "Dance"
        elif pulse > 0.45:
            genre = "Electronic"
        else:
            genre = "Ambient" if e < 0.5 else "Pop"
        conf = max(conf, 0.0)

    # ---- moods (0..1 each)
    has_beat = pulse >= 0.3
    beat_bpm = bpm if has_beat else 64.0  # beat-less music counts as slow
    n = lambda x, s=0.35: clip(x / s)  # model probabilities are small; stretch them
    calm = 1 - e
    dance = max(g.get("Dance", 0), g.get("Pop", 0) * 0.8, g.get("Latin", 0), g.get("Hip-Hop", 0) * 0.8)
    mid = lambda x, c, w: clip(1 - abs(x - c) / w)  # 1 at c, 0 at c +- w
    m = {
        "Energize": 0.45 * e + 0.25 * n(exciting) + 0.30 * clip((beat_bpm - 100) / 45),
        "Workout": (0.40 * e + 0.30 * pulse + 0.30 * clip((beat_bpm - 112) / 40))
        if (beat_bpm >= 112 and e >= 0.6 and pulse >= 0.5) else 0.0,
        "Party": (0.45 * n(dance) + 0.30 * e + 0.25 * pulse) if (98 <= beat_bpm <= 140 and has_beat) else 0.0,
        "Happy": 0.50 * n(happy, 0.25) + 0.30 * major + 0.20 * bright,
        "Chill": 0.45 * calm + 0.20 * n(tender) + 0.20 * clip((118 - beat_bpm) / 40) + 0.15 * (1 - n(angry)),
        "Focus": 0.30 * (1 - n(vocal, 0.4)) + 0.35 * mid(e, 0.38, 0.4) + 0.15 * (1 - n(exciting))
        + 0.20 * clip((125 - beat_bpm) / 45),
        "Sleep": (0.45 * calm + 0.25 * (1 - pulse) + 0.15 * (1 - bright) + 0.15 * n(lullaby, 0.2)) if e < 0.42 else 0.0,
        "Nature": n(nature, 0.25),
        "Late night": 0.35 * (1 - major) + 0.30 * (1 - bright) + 0.20 * max(n(g.get("Jazz", 0)), n(g.get("Electronic", 0)))
        + 0.15 * mid(e, 0.45, 0.45),
        "Romantic": 0.50 * n(tender) + 0.30 * n(vocal, 0.4) + 0.20 * clip((112 - beat_bpm) / 40),
        "Melancholy": 0.55 * n(sad, 0.25) + 0.30 * (1 - major) + 0.15 * calm,
    }
    if not has_model:
        # Without the model, keep only moods we can judge from sound alone.
        for k in ("Nature", "Romantic", "Happy", "Melancholy"):
            m[k] *= 0.6
    ranked_m = sorted(m.items(), key=lambda kv: kv[1], reverse=True)
    moods = [k for k, v in ranked_m if v >= 0.6][:3] or [ranked_m[0][0]]
    moods.sort(key=MOOD_ORDER.index)

    # Tempo for the visuals. Beat-less music gets a slow, breathing pulse instead.
    tempo = round(float(bpm), 2) if has_beat else 64.0
    offset = round(float(f["beatOffset"]), 3) if has_beat else 0.0

    return {
        "genre": genre,
        "genreConfidence": round(float(conf), 3),
        "genreTop": [[k, round(float(v), 3)] for k, v in ranked[:3]],
        "moods": moods,
        "moodScores": {k: round(float(v), 3) for k, v in ranked_m[:5]},
        "bpm": int(round(tempo)),
        "tempo": tempo,
        "beatOffset": offset,
        "hasBeat": has_beat,
        "energy": round(float(e), 3),
        "key": f["key"],
        "vocals": round(float(vocal), 3),
        "nature": round(float(nature), 3),
        "model": has_model,
    }


def analyse(path: Path, listener: Listener) -> dict:
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(y) < SR:
        raise ValueError("audio too short")
    f = features(y)
    p = listener.probs(librosa.resample(y, orig_sr=SR, target_sr=AST_SR)) if listener.ok else {}
    res = decide(f, p)
    res["features"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}
    return res


# --------------------------------------------------------------------------
def apply(track: dict, res: dict) -> None:
    """Writes the results into a catalog entry. The owner's own choices win."""
    if track.get("genreSource", "default") != "folder":
        track["genre"] = res["genre"]
        track["genreSource"] = "auto"
    if track.get("bpmSource", "default") != "tag":
        track["bpm"] = res["bpm"]
        track["tempo"] = res["tempo"]
        track["beatOffset"] = res["beatOffset"]
    track["moods"] = res["moods"]
    track["energy"] = res["energy"]
    track["key"] = res["key"]


def main() -> int:
    if not CATALOG.exists():
        print("catalog.json not found, nothing to do.")
        return 0
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    tracks = catalog.get("tracks", [])
    cache = {}
    if ANALYSIS.exists():
        try:
            cache = json.loads(ANALYSIS.read_text(encoding="utf-8-sig")).get("tracks", {})
        except Exception:  # noqa: BLE001
            cache = {}

    todo = []
    for t in tracks:
        path = ROOT / t["file"]
        if not path.exists():
            continue
        size = path.stat().st_size
        c = cache.get(t["id"])
        if not (c and c.get("size") == size and c.get("v") == VERSION):
            todo.append((t, path, size))
    print(f"{len(tracks)} songs, {len(todo)} to analyse.")

    listener = Listener() if todo else None
    done = 0
    for t, path, size in todo:
        t0 = time.time()
        try:
            res = analyse(path, listener)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {t.get('title')}: {e}")
            continue
        res.update({"v": VERSION, "size": size, "title": t.get("title"), "analysed": time.strftime("%Y-%m-%d %H:%M")})
        cache[t["id"]] = res
        done += 1
        print(f"  {t.get('title')}: {res['genre']} | {', '.join(res['moods'])} | {res['bpm']} BPM | "
              f"{res['key']} | energy {res['energy']} ({time.time() - t0:.1f}s)")

    ids = {t["id"] for t in tracks}
    cache = {k: v for k, v in cache.items() if k in ids}
    for t in tracks:
        if t["id"] in cache:
            apply(t, cache[t["id"]])

    # Only rewrite files whose content really changed, so nothing is
    # committed (and no new Pages build starts) when there is nothing new.
    new_analysis = {"version": VERSION, "model": MODEL, "tracks": cache}
    old_analysis = json.loads(ANALYSIS.read_text(encoding="utf-8-sig")) if ANALYSIS.exists() else None
    if new_analysis != old_analysis:
        ANALYSIS.write_text(json.dumps(new_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    if catalog != json.loads(CATALOG.read_text(encoding="utf-8-sig")):
        CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as s:
            s.write("## Aura song analysis\n\n| Song | Genre | Moods | BPM | Key | Energy |\n|---|---|---|---|---|---|\n")
            for t in tracks:
                r = cache.get(t["id"])
                if r:
                    g = t.get("genre") + (" (your folder)" if t.get("genreSource") == "folder" else "")
                    s.write(f"| {t.get('title')} | {g} | {', '.join(r['moods'])} | {t.get('bpm')} | {r['key']} | {r['energy']} |\n")
    print(f"Done. {done} analysed now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
