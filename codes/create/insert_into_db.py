import pandas as pd
import psycopg2
from datetime import datetime
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_to_db():
    """Create a connection to the PostgreSQL database"""
    try:
        connection = psycopg2.connect(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def insert_factory_pools(csv_path):
    """Insert data from CSV into factory_pool_created table"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create database connection
    connection = connect_to_db()
    if not connection:
        return
    
    try:
        cursor = connection.cursor()
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            # Skip rows with NaN or NaT values in critical fields
            if pd.isna(row['BLOCK_TIMESTAMP']) or pd.isna(row['BLOCK_NUMBER']):
                print(f"Skipping row with invalid timestamp or block number: {row}")
                continue

            insert_query = """
            INSERT INTO factory_pool_created (
                chain_name, block_timestamp, block_number, 
                transaction_hash, log_index, token0, token1,
                fee, tickSpacing, pool
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            # Convert timestamp if it's not already in datetime format
            try:
                block_timestamp = pd.to_datetime(row['BLOCK_TIMESTAMP'])
                if pd.isna(block_timestamp):
                    print(f"Skipping row with invalid timestamp: {row}")
                    continue
            except:
                print(f"Error converting timestamp for row: {row}")
                continue
            
            values = (
                row['CHAIN_NAME'] if not pd.isna(row['CHAIN_NAME']) else 'ethereum',
                block_timestamp,
                int(row['BLOCK_NUMBER']),
                row['TRANSACTION_HASH'] if not pd.isna(row['TRANSACTION_HASH']) else '',
                int(row['LOG_INDEX']) if not pd.isna(row['LOG_INDEX']) else 0,
                row['TOKEN0'] if not pd.isna(row['TOKEN0']) else '',
                row['TOKEN1'] if not pd.isna(row['TOKEN1']) else '',
                row['FEE'] if not pd.isna(row['FEE']) else '0',
                row['TICKSPACING'] if not pd.isna(row['TICKSPACING']) else '0',
                row['POOL'] if not pd.isna(row['POOL']) else ''
            )
            
            cursor.execute(insert_query, values)
        
        # Commit the transaction
        connection.commit()
        print("Factory pool data inserted successfully")
        
    except Exception as e:
        print(f"Error inserting factory pool data: {e}")
        connection.rollback()
    
    finally:
        if connection:
            cursor.close()
            connection.close()

def insert_pool_initialize_events(csv_path):
    """Insert data from CSV into pool_initialize_events table"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create database connection
    connection = connect_to_db()
    if not connection:
        return
    
    try:
        cursor = connection.cursor()
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            # Skip rows with NaN or NaT values in critical fields
            if pd.isna(row['BLOCK_TIMESTAMP']) or pd.isna(row['BLOCK_NUMBER']):
                print(f"Skipping row with invalid timestamp or block number: {row}")
                continue

            insert_query = """
            INSERT INTO pool_initialize_events (
                chain_name, address, block_timestamp, block_number,
                transaction_hash, log_index, sqrtPriceX96, tick,
                to_address, from_address, transaction_index,
                gas_price, gas_used
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            # Convert timestamp if it's not already in datetime format
            try:
                block_timestamp = pd.to_datetime(row['BLOCK_TIMESTAMP'])
                if pd.isna(block_timestamp):
                    print(f"Skipping row with invalid timestamp: {row}")
                    continue
            except:
                print(f"Error converting timestamp for row: {row}")
                continue
            
            values = (
                'ethereum',
                row['ADDRESS'] if not pd.isna(row['ADDRESS']) else '',
                block_timestamp,
                int(row['BLOCK_NUMBER']),
                row['TRANSACTION_HASH'] if not pd.isna(row['TRANSACTION_HASH']) else '',
                int(row['LOG_INDEX']) if not pd.isna(row['LOG_INDEX']) else 0,
                str(row['SQRTPRICEX96']) if not pd.isna(row['SQRTPRICEX96']) else '0',
                int(row['TICK']) if not pd.isna(row['TICK']) else 0,
                row['TO_ADDRESS'] if not pd.isna(row['TO_ADDRESS']) else '',
                row['FROM_ADDRESS'] if not pd.isna(row['FROM_ADDRESS']) else '',
                int(row['TRANSACTION_INDEX']) if not pd.isna(row['TRANSACTION_INDEX']) else 0,
                str(row['GAS_PRICE']) if not pd.isna(row['GAS_PRICE']) else '0',
                str(row['GAS_USED']) if not pd.isna(row['GAS_USED']) else '0'
            )
            
            cursor.execute(insert_query, values)
        
        # Commit the transaction
        connection.commit()
        print("Pool initialize events data inserted successfully")
        
    except Exception as e:
        print(f"Error inserting pool initialize events data: {e}")
        connection.rollback()
    
    finally:
        if connection:
            cursor.close()
            connection.close()

def insert_pool_mint_burn_events(csv_path):
    """Insert data from CSV into pool_mint_burn_events table"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create database connection
    connection = connect_to_db()
    if not connection:
        return
    
    try:
        cursor = connection.cursor()
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            # Skip rows with NaN or NaT values in critical fields
            if pd.isna(row['block_timestamp']) or pd.isna(row['block_timestamp']):
                print(f"Skipping row with invalid timestamp or block number: {row}")
                continue

            insert_query = """
            INSERT INTO pool_mint_burn_events (
                chain_name, address, block_timestamp, block_number,
                transaction_hash, log_index, amount, amount0,
                amount1, owner, tick_lower, tick_upper,
                type_of_event, to_address, from_address,
                transaction_index, gas_price, gas_used, l1_fee
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            # Convert timestamp if it's not already in datetime format
            try:
                block_timestamp = pd.to_datetime(row['block_timestamp'])
                if pd.isna(block_timestamp):
                    print(f"Skipping row with invalid timestamp: {row}")
                    continue
            except:
                print(f"Error converting timestamp for row: {row}")
                continue
            
            values = (
                'ethereum',
                row['address'] if not pd.isna(row['address']) else '',
                block_timestamp,
                int(row['block_number']),
                row['transaction_hash'] if not pd.isna(row['transaction_hash']) else '',
                int(row['log_index']) if not pd.isna(row['log_index']) else 0,
                str(row['amount']) if not pd.isna(row['amount']) else '0',
                str(row['amount0']) if not pd.isna(row['amount0']) else '0',
                str(row['amount1']) if not pd.isna(row['amount1']) else '0',
                row['owner'] if not pd.isna(row['owner']) else '',
                int(row['tick_lower']) if not pd.isna(row['tick_lower']) else 0,
                int(row['tick_upper']) if not pd.isna(row['tick_upper']) else 0,
                int(row['type_of_event']) if not pd.isna(row['type_of_event']) else 0,
                row['to_address'] if not pd.isna(row['to_address']) else '',
                row['from_address'] if not pd.isna(row['from_address']) else '',
                int(row['transaction_index']) if not pd.isna(row['transaction_index']) else 0,
                str(row['gas_price']) if not pd.isna(row['gas_price']) else '0',
                str(row['gas_used']) if not pd.isna(row['gas_used']) else '0',
                str(row['l1_fee']) if not pd.isna(row['l1_fee']) else '0'
            )
            
            cursor.execute(insert_query, values)
        
        # Commit the transaction
        connection.commit()
        print("Pool mint/burn events data inserted successfully")
        
    except Exception as e:
        print(f"Error inserting pool mint/burn events data: {e}")
        connection.rollback()
    
    finally:
        if connection:
            cursor.close()
            connection.close()

def insert_pool_swap_events(pool_address):
    """Insert data from CSV into pool_swap_events table for a specific pool"""
    csv_path = f"../../data/raw/swap_data_{pool_address}.csv"
    if not os.path.exists(csv_path):
        print(f"Pool swap events CSV file not found: {csv_path}")
        return

    print(f"Processing swap events for pool: {pool_address}")
    # Read CSV file
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Create database connection
    connection = connect_to_db()
    if not connection:
        return
    
    try:
        cursor = connection.cursor()
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            # Skip rows with NaN or NaT values in critical fields
            if pd.isna(row['block_timestamp']) or pd.isna(row['block_number']):
                print(f"Skipping row with invalid timestamp or block number: {row}")
                continue

            insert_query = """
            INSERT INTO pool_swap_events (
                chain_name, address, block_timestamp, block_number,
                transaction_hash, log_index, sender, recipient,
                amount0, amount1, sqrtPriceX96, liquidity, tick,
                from_address, to_address, transaction_index,
                gas_price, gas_used, l1_fee
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            # Convert timestamp if it's not already in datetime format
            try:
                block_timestamp = pd.to_datetime(row['block_timestamp'])
                if pd.isna(block_timestamp):
                    print(f"Skipping row with invalid timestamp: {row}")
                    continue
            except:
                print(f"Error converting timestamp for row: {row}")
                continue
            
            values = (
                'ethereum',
                row['address'] if not pd.isna(row['address']) else '',
                block_timestamp,
                int(row['block_number']),
                row['transaction_hash'] if not pd.isna(row['transaction_hash']) else '',
                int(row['log_index']) if not pd.isna(row['log_index']) else 0,
                row['sender'] if not pd.isna(row['sender']) else '',
                row['recipient'] if not pd.isna(row['recipient']) else '',
                str(row['amount0']) if not pd.isna(row['amount0']) else '0',
                str(row['amount1']) if not pd.isna(row['amount1']) else '0',
                str(row['sqrtpricex96']) if not pd.isna(row['sqrtpricex96']) else '0',
                str(row['liquidity']) if not pd.isna(row['liquidity']) else '0',
                int(row['tick']) if not pd.isna(row['tick']) else 0,
                row['from_address'] if not pd.isna(row['from_address']) else '',
                row['to_address'] if not pd.isna(row['to_address']) else '',
                int(row['transaction_index']) if not pd.isna(row['transaction_index']) else 0,
                str(row['gas_price']) if not pd.isna(row['gas_price']) else '0',
                str(row['gas_used']) if not pd.isna(row['gas_used']) else '0',
                '0'
            )
            
            cursor.execute(insert_query, values)
        
        # Commit the transaction
        connection.commit()
        print(f"Pool swap events data inserted successfully for pool: {pool_address}")
        
    except Exception as e:
        print(f"Error inserting pool swap events data for pool {pool_address}: {e}")
        connection.rollback()
    
    finally:
        if connection:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    factory_pools_csv = "../../data/raw/factory_pool_created.csv"
    pool_initialize_csv = "../../data/raw/pool_initialize_events.csv"
    pool_mint_burn_csv = "../../data/raw/pool_mint_burn_events.csv"
    
    # List of pools to process swap events
    POOL_ADDRS = [
        '0x11b815efb8f581194ae79006d24e0d814b7697f6',
        '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
        '0x69d91b94f0aaf8e8a2586909fa77a5c2c89818d5',
        '0x84383fb05f610222430f69727aa638f8fdbf5cc1',
        '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
        '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
        '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8',
        '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
    ]
    
    # Insert data from CSV files
    # if os.path.exists(factory_pools_csv):
    #     insert_factory_pools(factory_pools_csv)
    # else:
    #     print(f"Factory pools CSV file not found: {factory_pools_csv}")
        
    # if os.path.exists(pool_initialize_csv):
    #     insert_pool_initialize_events(pool_initialize_csv)
    # else:
    #     print(f"Pool initialize events CSV file not found: {pool_initialize_csv}")
        
    # if os.path.exists(pool_mint_burn_csv):
    #     insert_pool_mint_burn_events(pool_mint_burn_csv)
    # else:
    #     print(f"Pool mint/burn events CSV file not found: {pool_mint_burn_csv}")

    # Process swap events for each pool
    for pool_address in POOL_ADDRS:
        insert_pool_swap_events(pool_address)
