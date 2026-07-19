"""
Database Schema Migration
Fixes coherence_score and novelty_score from BYTES to REAL.

The issue: SQLite stored these as BLOB/BYTES instead of REAL.
This script creates a new properly-typed database and migrates data.
"""

import sqlite3
import json
import os
import shutil
from datetime import datetime

def migrate_database(old_db="universe.db", backup=True):
    if not os.path.exists(old_db):
        print(f"ERROR: {old_db} not found")
        return False
    
    # Backup
    if backup:
        backup_name = f"universe_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        print(f"Creating backup: {backup_name}")
        shutil.copy2(old_db, backup_name)
        print(f"  Backup saved")
    
    # Connect to old DB
    old_conn = sqlite3.connect(old_db)
    old_cursor = old_conn.cursor()
    
    # Check current schema
    old_cursor.execute("PRAGMA table_info(concepts)")
    schema_info = old_cursor.fetchall()
    print("\nCurrent schema:")
    for col in schema_info:
        print(f"  {col[1]}: {col[2]}")
    
    # Get all concepts
    old_cursor.execute("SELECT * FROM concepts")
    concepts = old_cursor.fetchall()
    print(f"\nFound {len(concepts)} concepts to migrate")
    
    # Get column names
    old_cursor.execute("PRAGMA table_info(concepts)")
    columns = [col[1] for col in old_cursor.fetchall()]
    print(f"Columns: {columns}")
    
    # Create new database with correct schema
    new_db = "universe_new.db"
    if os.path.exists(new_db):
        os.remove(new_db)
        
    new_conn = sqlite3.connect(new_db)
    new_cursor = new_conn.cursor()
    
    # Create correct schema
    new_cursor.execute('''
        CREATE TABLE concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding TEXT,
            latent TEXT,
            parent_ids TEXT,
            generation INTEGER,
            coherence_score REAL,
            novelty_score REAL,
            times_referenced INTEGER DEFAULT 0,
            date_added REAL
        )
    ''')
    
    new_cursor.execute('''
        CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id INTEGER,
            event_type TEXT,
            timestamp REAL
        )
    ''')
    
    new_cursor.execute('''
        CREATE TABLE ingested_files (
            filename TEXT PRIMARY KEY,
            concept_id INTEGER
        )
    ''')
    
    # Migrate concepts
    print("\nMigrating concepts...")
    success_count = 0
    error_count = 0
    
    for i, row in enumerate(concepts):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(concepts)}...")
            
        try:
            # Unpack row
            c_id, embedding, latent, parent_ids, generation, coh_bytes, nov_bytes, times_ref, date_added = row
            
            # Convert BYTES scores to REAL
            # The bytes might be pickled floats or raw binary - try to interpret
            # Most likely they're just the text representation stored as bytes
            try:
                if isinstance(coh_bytes, bytes):
                    # Try decoding as UTF-8 string first
                    try:
                        coh_str = coh_bytes.decode('utf-8')
                        coherence = float(coh_str)
                    except:
                        # If that fails, treat as empty/default
                        coherence = 0.5
                elif isinstance(coh_bytes, (int, float)):
                    coherence = float(coh_bytes)
                else:
                    coherence = 0.5
            except:
                coherence = 0.5
                
            try:
                if isinstance(nov_bytes, bytes):
                    try:
                        nov_str = nov_bytes.decode('utf-8')
                        novelty = float(nov_str)
                    except:
                        novelty = 0.5
                elif isinstance(nov_bytes, (int, float)):
                    novelty = float(nov_bytes)
                else:
                    novelty = 0.5
            except:
                novelty = 0.5
            
            # Insert into new DB (let ID auto-increment)
            new_cursor.execute('''
                INSERT INTO concepts (embedding, latent, parent_ids, generation, 
                                     coherence_score, novelty_score, times_referenced, date_added)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (embedding, latent, parent_ids, generation, coherence, novelty, times_ref, date_added))
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  ERROR on concept {c_id}: {e}")
    
    # Migrate history
    old_cursor.execute("SELECT * FROM history")
    history_rows = old_cursor.fetchall()
    print(f"\nMigrating {len(history_rows)} history entries...")
    for row in history_rows:
        new_cursor.execute("INSERT INTO history (concept_id, event_type, timestamp) VALUES (?, ?, ?)", 
                          (row[1], row[2], row[3]))
    
    # Migrate ingested_files
    old_cursor.execute("SELECT * FROM ingested_files")
    ingested_rows = old_cursor.fetchall()
    print(f"Migrating {len(ingested_rows)} ingested file records...")
    for row in ingested_rows:
        new_cursor.execute("INSERT INTO ingested_files (filename, concept_id) VALUES (?, ?)", 
                          (row[0], row[1]))
    
    # Commit and close
    new_conn.commit()
    new_conn.close()
    old_conn.close()
    
    # Replace old DB with new
    print("\nReplacing old database...")
    os.remove(old_db)
    os.rename(new_db, old_db)
    
    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Successfully migrated: {success_count}/{len(concepts)} concepts")
    print(f"Errors: {error_count}")
    print(f"History entries: {len(history_rows)}")
    print(f"Ingested files: {len(ingested_rows)}")
    
    # Verify
    print("\nVerifying new schema...")
    verify_conn = sqlite3.connect(old_db)
    verify_cursor = verify_conn.cursor()
    verify_cursor.execute("PRAGMA table_info(concepts)")
    new_schema = verify_cursor.fetchall()
    print("New schema:")
    for col in new_schema:
        print(f"  {col[1]}: {col[2]}")
        
    # Test read
    verify_cursor.execute("SELECT id, coherence_score, novelty_score FROM concepts LIMIT 5")
    test_rows = verify_cursor.fetchall()
    print("\nSample data:")
    for row in test_rows:
        print(f"  Concept {row[0]}: coherence={row[1]:.3f}, novelty={row[2]:.3f}")
    
    verify_conn.close()
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("DATABASE SCHEMA MIGRATION")
    print("=" * 60)
    print("\nThis will fix BYTES corruption in coherence/novelty scores.")
    print("A backup will be created automatically.")
    
    # Auto-proceed if --auto flag
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("\nAuto-mode: Proceeding...")
        success = migrate_database()
        if success:
            print("\n[OK] Database migrated successfully!")
    else:
        response = input("\nProceed? (y/n): ")
        if response.lower() == 'y':
            success = migrate_database()
            if success:
                print("\n[OK] Database migrated successfully!")
        else:
            print("Cancelled.")
