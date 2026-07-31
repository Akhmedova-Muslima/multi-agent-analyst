import os
import sqlite3

def init_db():
    # Создаем папку data, если ее нет
    os.makedirs("./data", exist_ok=True)
    db_path = "./data/company.db"
    
    # Пересоздаем базу с нуля
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Таблица пользователей
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        status TEXT,
        joined_date TEXT
    )
    """)
    
    # Таблица метрик и финансовых показателей
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY,
        quarter TEXT,
        churned_customers INTEGER,
        revenue REAL
    )
    """)
    
    # Вставка тестовых данных
    cursor.executemany("""
    INSERT INTO users (id, name, status, joined_date) VALUES (?, ?, ?, ?)
    """, [
        (1, 'Alice', 'active', '2025-01-15'),
        (2, 'Bob', 'churned', '2025-02-10'),
        (3, 'Charlie', 'active', '2025-03-01'),
        (4, 'Diana', 'churned', '2025-04-12')
    ])
    
    cursor.executemany("""
    INSERT INTO metrics (id, quarter, churned_customers, revenue) VALUES (?, ?, ?, ?)
    """, [
        (1, 'Q1', 120, 450000.0),
        (2, 'Q2', 85, 520000.0),
        (3, 'Q3', 140, 480000.0),
        (4, 'Q4', 95, 610000.0)
    ])
    
    conn.commit()
    conn.close()
    print("✅ База данных ./data/company.db успешно создана!")

if __name__ == "__main__":
    init_db()
