import sqlite3, numpy as np, time
from threading import RLock

class FaceMatcher:
    def __init__(self, db_path="db/attend.sqlite"):
        self.db_path = db_path
        self._cache = None
        self._names = None
        self._last_load = 0
        self._lock = RLock()

    def _load(self):
        with self._lock:
            if time.time() - self._last_load < 5 and self._cache is not None:
                return
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS persons(
                id INTEGER PRIMARY KEY, name TEXT UNIQUE)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS face_vectors(
                person_id INTEGER, vec BLOB)""")
            cur.execute("""SELECT persons.name, face_vectors.vec
                           FROM face_vectors JOIN persons
                           ON persons.id=face_vectors.person_id""")
            rows = cur.fetchall()
            conn.close()
            names, vecs = [], []
            for name, blob in rows:
                v = np.frombuffer(blob, dtype=np.float32)
                names.append(name); vecs.append(v)
            if vecs:
                self._cache = np.stack(vecs)
                self._names = names
            else:
                self._cache = np.zeros((0,512), np.float32)
                self._names = []
            self._last_load = time.time()

    def identify(self, vec, cos_min=0.65):
        self._load()
        if self._cache.shape[0] == 0:
            return "Unknown", 0.0
        sims = self._cache @ vec
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= cos_min:
            return self._names[idx], float(sims[idx])
        return "Unknown", float(sims[idx])
