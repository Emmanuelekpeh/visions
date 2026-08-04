import sqlite3
import faiss
import numpy as np
import json
import time
import threading

class WorldMemory:
    def __init__(self, db_path="universe.db", embedding_dim=512):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # Thread safety lock for all database operations
        self._db_lock = threading.RLock()
        
        # Initialize SQLite
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Initialize FAISS (CPU)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_map = {}  # Maps FAISS index to concept ID
        self.next_faiss_id = 0
        self.faiss_path = db_path.replace('.db', '.index')
        self.id_map_path = db_path.replace('.db', '_id_map.json')
        
        self._init_db()
        self._load_faiss_from_db()
    
    @property
    def db_lock(self):
        """Get the database lock for external modules that need direct connection access."""
        return self._db_lock

    def save_faiss_index(self):
        """Save FAISS index and ID map to disk."""
        with self._db_lock:
            faiss.write_index(self.index, self.faiss_path)
            with open(self.id_map_path, 'w') as f:
                json.dump({str(k): v for k, v in self.id_map.items()}, f)

    def get_living_ids(self) -> set:
        """IDs that still exist in the concepts table."""
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM concepts")
            return {row[0] for row in cursor.fetchall()}

    def concept_exists(self, concept_id: int) -> bool:
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM concepts WHERE id = ? LIMIT 1", (concept_id,))
            return cursor.fetchone() is not None

    def rebuild_faiss_from_db(self):
        """
        Rebuild FAISS + id_map from living concepts only.
        Required after extinctions: IndexFlatL2 cannot delete, so pruned IDs
        otherwise remain as ghosts and poison nearest-neighbor / evolution.
        """
        with self._db_lock:
            print("Rebuilding FAISS index from living concepts...")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.id_map = {}
            self.next_faiss_id = 0

            cursor = self.conn.cursor()
            cursor.execute("SELECT id, embedding FROM concepts ORDER BY id")
            count = 0
            for concept_id, emb_json in cursor.fetchall():
                if not emb_json:
                    continue
                try:
                    emb = np.array(json.loads(emb_json), dtype=np.float32).reshape(1, -1)
                except Exception:
                    continue
                if emb.shape[1] != self.embedding_dim:
                    continue
                self.index.add(emb)
                self.id_map[self.next_faiss_id] = concept_id
                self.next_faiss_id += 1
                count += 1

            self.save_faiss_index()
            print(f"  -> FAISS rebuilt with {count} living vectors.")
            return count

    def purge_orphan_metadata(self, living_ids: set = None):
        """Remove graph/epistemic metadata rows for concepts that no longer exist."""
        with self._db_lock:
            if living_ids is None:
                living_ids = self.get_living_ids()
            cursor = self.conn.cursor()
            removed = 0
            for table in ("graph_metadata", "epistemic_metadata"):
                cursor.execute(f"SELECT concept_id FROM {table}")
                orphans = [row[0] for row in cursor.fetchall() if row[0] not in living_ids]
                for cid in orphans:
                    cursor.execute(f"DELETE FROM {table} WHERE concept_id = ?", (cid,))
                    removed += 1
            self.conn.commit()
            if removed:
                print(f"  -> Purged {removed} orphan metadata rows.")
            return removed

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding JSON,
                image_path TEXT,
                parent_ids JSON,
                generation INTEGER,
                coherence_score REAL,
                novelty_score REAL,
                times_referenced INTEGER DEFAULT 0,
                date_added REAL,
                tags JSON DEFAULT '[]',
                generator_version INTEGER DEFAULT 0
            )
        ''')
        
        # Add image_path column if it doesn't exist (migration for existing databases)
        try:
            self.cursor.execute("SELECT image_path FROM concepts LIMIT 1")
        except sqlite3.OperationalError:
            print("Migrating database: Adding image_path column...")
            self.cursor.execute("ALTER TABLE concepts ADD COLUMN image_path TEXT")
            self.conn.commit()
            print("  -> Migration complete")
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER,
                event_type TEXT,
                timestamp REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fossils (
                id INTEGER PRIMARY KEY,
                embedding JSON,
                image_path TEXT,
                parent_ids JSON,
                generation INTEGER,
                coherence_score REAL,
                novelty_score REAL,
                times_referenced INTEGER DEFAULT 0,
                date_added REAL,
                date_extinct REAL,
                extinction_reason TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ingested_files (
                filename TEXT PRIMARY KEY,
                concept_id INTEGER
            )
        ''')
        
        # Ecology Engine (Phase 6: Species, Culture, Speciation)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ecology_species (
                id INTEGER PRIMARY KEY,
                data JSON
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transformations (
                id INTEGER PRIMARY KEY,
                delta_latent JSON,
                magnitude REAL,
                energy_cost REAL,
                tier TEXT,
                confidence REAL,
                parent_id INTEGER,
                children_ids JSON,
                success_count INTEGER,
                failure_history JSON,
                total_offspring_fitness REAL,
                novelty_created REAL,
                domains JSON,
                creation_time REAL,
                last_used REAL
            )
        ''')
        
        # Epistemic metadata table (NEW - stores epistemic memory state)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS epistemic_metadata (
                concept_id INTEGER PRIMARY KEY,
                tier TEXT,
                confidence REAL,
                reality_distance REAL,
                prediction_errors JSON,
                validation_count INTEGER,
                creation_time REAL,
                last_validation REAL,
                promotion_eligible INTEGER,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        ''')
        
        # Graph metadata table (stores learned graph state)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_metadata (
                concept_id INTEGER PRIMARY KEY,
                tier TEXT,
                confidence REAL,
                fitness REAL,
                validation_count INTEGER,
                prediction_errors JSON,
                selection_count INTEGER,
                offspring_count INTEGER,
                successful_offspring INTEGER,
                last_visited REAL,
                visit_frequency REAL,
                region_explored INTEGER,
                reality_distance REAL,
                combines_with JSON,
                contradicts JSON,
                specializes_into JSON,
                transformed_into JSON,
                caused_by JSON,
                failed_invalid JSON,
                failed_unstable JSON,
                failed_sterile JSON,
                failed_catastrophic JSON,
                ecology JSON,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        ''')
        
        self.conn.commit()

    def _load_faiss_from_db(self):
        import os
        if os.path.exists(self.faiss_path) and os.path.exists(self.id_map_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
                with open(self.id_map_path, 'r') as f:
                    loaded_map = json.load(f)
                    self.id_map = {int(k): v for k, v in loaded_map.items()}
                self.next_faiss_id = max(self.id_map.keys()) + 1 if self.id_map else 0
                print(f"Loaded FAISS index from disk with {self.index.ntotal} vectors.")

                # Detect ghost IDs (extinct concepts still in FAISS) and rebuild
                living = self.get_living_ids()
                mapped = set(self.id_map.values())
                ghosts = mapped - living
                if ghosts or self.index.ntotal != len(living) or len(mapped) != len(living):
                    print(
                        f"[FAISS] Stale index detected "
                        f"(vectors={self.index.ntotal}, living={len(living)}, ghosts={len(ghosts)}). "
                        f"Rebuilding..."
                    )
                    self.rebuild_faiss_from_db()
                return
            except Exception as e:
                print(f"Failed to load FAISS index from disk: {e}. Rebuilding from DB...")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.id_map = {}
                self.next_faiss_id = 0

        self.rebuild_faiss_from_db()

    def add_concept(self, embedding: np.ndarray, image_path: str, parent_ids: list, 
                    generation: int, coherence: float, novelty: float, tags: list = None,
                    generator_version: int = 0):
        """
        Add a concept to the database.
        
        Args:
            embedding: Concept embedding (always stored)
            image_path: Path to the saved image
            parent_ids: Parent concept IDs
            generation: Generation number
            coherence: Coherence score
            novelty: Novelty score
            tags: Semantic tags
            generator_version: 0 = stored latent, 1+ = generator version, -1 = real image (always stored)
        """
        with self._db_lock:
            if tags is None:
                tags = []
                
            # Insert to SQLite
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO concepts (embedding, image_path, parent_ids, generation, coherence_score, novelty_score, date_added, tags, generator_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                json.dumps(embedding.tolist()),
                image_path,
                json.dumps(parent_ids),
                generation,
                coherence,
                novelty,
                time.time(),
                json.dumps(tags),
                generator_version
            ))
            concept_id = cursor.lastrowid
            self.conn.commit()
            
            # Insert to FAISS
            self.index.add(embedding.reshape(1, -1))
            self.id_map[self.next_faiss_id] = concept_id
            self.next_faiss_id += 1
            
            return concept_id

    def get_concept(self, concept_id: int):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
            return cursor.fetchone()

    def get_nearest_concepts(self, embedding: np.ndarray, k=5):
        with self._db_lock:
            if self.index.ntotal == 0:
                return []

            # Oversample then filter — ghosts may still linger until next rebuild
            living = None
            search_k = min(self.index.ntotal, max(k * 8, k))
            distances, indices = self.index.search(embedding.reshape(1, -1), search_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx not in self.id_map:
                    continue
                concept_id = self.id_map[idx]
                if living is None:
                    living = self.get_living_ids()
                if concept_id not in living:
                    continue
                results.append((concept_id, float(distances[0][i])))
                if len(results) >= k:
                    break

            return results

    def get_all_concepts(self):
        # Use a new cursor to avoid recursive use issues when called from a thread
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, image_path, parent_ids, generation, embedding FROM concepts")
            return cursor.fetchall()
        
    def get_max_generation(self):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MAX(generation) FROM concepts")
            res = cursor.fetchone()
            return res[0] if res and res[0] is not None else 0

    def increment_reference(self, concept_id: int):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE concepts SET times_referenced = times_referenced + 1 WHERE id = ?", (concept_id,))
            self.conn.commit()

    def is_file_ingested(self, filename: str) -> bool:
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM ingested_files WHERE filename = ?", (filename,))
            return cursor.fetchone() is not None

    def mark_file_ingested(self, filename: str, concept_id: int):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO ingested_files (filename, concept_id) VALUES (?, ?)", (filename, concept_id))
            self.conn.commit()
        
    def prune_concept(self, concept_id: int, reason: str = "Low Fitness"):
        """Move concept to fossils table and remove from active memory"""
        with self._db_lock:
            cursor = self.conn.cursor()
            
            # Get concept data
            cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
            row = cursor.fetchone()
            if not row:
                return False
                
            # Insert into fossils (ignore if already fossilized)
            cursor.execute('''
                INSERT OR REPLACE INTO fossils (id, embedding, image_path, parent_ids, generation, coherence_score, novelty_score, times_referenced, date_added, date_extinct, extinction_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                time.time(), reason
            ))
            
            # Delete from concepts
            cursor.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))

            # Keep metadata layers in sync — orphan rows cause restart ghost lookups
            cursor.execute("DELETE FROM graph_metadata WHERE concept_id = ?", (concept_id,))
            cursor.execute("DELETE FROM epistemic_metadata WHERE concept_id = ?", (concept_id,))
            
            # Log event
            cursor.execute("INSERT INTO history (concept_id, event_type, timestamp) VALUES (?, ?, ?)", 
                          (concept_id, f"extinction: {reason}", time.time()))
                          
            self.conn.commit()
            
            # IndexFlatL2 cannot delete in-place; drop from id_map so nearest-neighbor
            # filters skip this ID. Periodic rebuild_faiss_from_db() clears the vector.
            dead_slots = [faiss_idx for faiss_idx, cid in self.id_map.items() if cid == concept_id]
            for faiss_idx in dead_slots:
                del self.id_map[faiss_idx]
            
            return True

    def get_ingested_files(self, limit: int = None):
        """Get list of ingested file concept IDs."""
        with self._db_lock:
            cursor = self.conn.cursor()
            if limit:
                cursor.execute("SELECT concept_id, filename FROM ingested_files LIMIT ?", (limit,))
            else:
                cursor.execute("SELECT concept_id, filename FROM ingested_files")
            return cursor.fetchall()
    
    def get_image_path_for_concept(self, concept_id: int):
        """Get the image path for a concept ID."""
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT image_path FROM concepts WHERE id = ?", (concept_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def count_ingested_files(self) -> int:
        """Count total ingested files."""
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ingested_files")
            return cursor.fetchone()[0]
    
    def clean_orphan_parent_ids(self):
        """Remove references to extinct parent concepts from parent_ids field."""
        with self._db_lock:
            cursor = self.conn.cursor()
            living_ids = self.get_living_ids()
            fixed_parents = 0
            
            for cid, pjson in cursor.execute("SELECT id, parent_ids FROM concepts").fetchall():
                if not pjson:
                    continue
                try:
                    parent_list = json.loads(pjson)
                    if not parent_list:
                        continue
                    cleaned = [p for p in parent_list if p in living_ids]
                    if len(cleaned) != len(parent_list):
                        cursor.execute(
                            "UPDATE concepts SET parent_ids = ? WHERE id = ?",
                            (json.dumps(cleaned), cid)
                        )
                        fixed_parents += 1
                except Exception:
                    continue
            
            if fixed_parents:
                self.conn.commit()
                print(f"  Cleaned dead parent_ids on {fixed_parents} concepts.")
            return fixed_parents
    
    def get_latest_concept_id(self):
        """Get the ID of the most recently added concept."""
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM concepts ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_real_image_anchor_ids(self):
        """Get concept IDs for real images (ingested or generator_version = -1)."""
        with self._db_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT id FROM concepts
                    WHERE image_path IS NOT NULL
                      AND (generator_version = -1 OR id IN (SELECT concept_id FROM ingested_files))
                    """
                )
                return [int(r[0]) for r in cursor.fetchall()]
            except Exception:
                return []
    
    def get_concepts_with_image(self):
        """Get all concept IDs that have stored images."""
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM concepts WHERE image_path IS NOT NULL")
            return [int(r[0]) for r in cursor.fetchall()]
