import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset

logger = logging.getLogger(__name__)

class MedicalLoader:
    def __init__(self, input_dir: str, output_dir: str, 
                 max_events_per_session: int = 8, 
                 time_window_hours: int = 24):
        """
        Medical Data Loader
        
        Parameters:
        input_dir: Path to raw data directory
        output_dir: Path to processed output directory
        max_events_per_session: Maximum number of events per session
        time_window_hours: Time window size (hours)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_events_per_session = max_events_per_session
        self.time_window_hours = time_window_hours
        self.table_files = {
            'ChemistryEvents': 'ChemistryEvents.csv',
            'ABGEvents': 'ABGEvents.csv',
            'CultureEvents': 'CultureEvents.csv',
            'CBCEvents': 'CBCEvents.csv'
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_process(self):
        """Load and process all table data"""
        logger.info(f"Starting medical data processing from: {self.input_dir}")
        
        # 1. Read all table data
        all_data = {}
        for table_name, filename in self.table_files.items():
            file_path = os.path.join(self.input_dir, filename)
            logger.info(f"Loading table: {table_name}, path: {file_path}")
            
            if os.path.exists(file_path):
                try:
                    # Read CSV file, handle special values
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # Additional error handling
                    if 'time_event' not in df.columns:
                        logger.warning(f"Table {table_name} missing 'time_event' column, skipping")
                        continue
                        
                    # Rename columns for consistency
                    if table_name == 'CultureEvents':
                        df['value'] = df['result']
                    elif table_name == 'ABGEvents':
                        df['variable_name'] = df['abg_ventilator_mode'] + '-' + df['abg_name']
                        df.drop(columns=['abg_ventilator_mode', 'abg_name'], inplace=True)
                        df.rename(columns={'value': 'value'}, inplace=True)
                    elif table_name == 'CBCEvents':
                        df.rename(columns={'cbc_name': 'variable_name'}, inplace=True)
                    elif table_name == 'ChemistryEvents':
                        df.rename(columns={'chem_name': 'variable_name'}, inplace=True)
                    
                    # Add table type marker
                    df['table_type'] = table_name
                    all_data[table_name] = df
                except Exception as e:
                    logger.error(f"Error loading table {table_name}: {str(e)}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # If no data, return immediately
        if not all_data:
            logger.error("No valid data found")
            return None
        
        # 2. Merge all table data
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        logger.info(f"Merged data count: {len(combined_df)}")
        
        # 3. Parse time format and normalize
        combined_df['time_event'] = pd.to_datetime(
            combined_df['time_event'], 
            errors='coerce'
        )
        num_valid = combined_df['time_event'].notna().sum()
        num_invalid = combined_df['time_event'].isna().sum()
        logger.info(f"Valid time records: {num_valid}")
        logger.info(f"Invalid time records: {num_invalid}")

        # Print some examples of invalid time records
        if num_invalid > 0:
            invalid_samples = combined_df[combined_df['time_event'].isna()]['time_event'].head(5).tolist()
            logger.warning(f"Invalid time examples: {invalid_samples}")

        # Delete invalid time records
        combined_df = combined_df.dropna(subset=['time_event'])
        logger.info(f"Valid events count: {len(combined_df)}")
        
        # 4. Group by patient
        grouped = combined_df.groupby('PatientID')
        all_conversations = []
        
        # Create patient session index
        patient_index = {}
        
        for i, (patient_id, patient_data) in enumerate(grouped):
            if i >= 5:  # Process only first 5 patients for testing
                break
            logger.info(f"Processing patient: {patient_id}, events: {len(patient_data)}")
            
            # 5. Create time-clustered sessions for patient
            sessions = self._create_sessions_for_patient(patient_id, patient_data)
            
            # Create session objects
            session_objects = []
            for session_idx, session_data in enumerate(sessions):
                # Skip empty sessions
                if session_data.empty:
                    logger.warning(f"Session {session_idx+1} for patient {patient_id} is empty, skipping")
                    continue
                    
                table_objects = self._create_table_objects(session_data)
                
                # Create Session object
                session_id = f"{patient_id}_session_{session_idx+1}"
                
                # Safely calculate time range
                try:
                    min_time = session_data['time_event'].min()
                    max_time = session_data['time_event'].max()
                    
                    # Ensure time is valid
                    if pd.isna(min_time) or pd.isna(max_time):
                        time_range_str = "Invalid time range"
                    else:
                        time_range_str = f"{min_time.strftime('%Y-%m-%d %H:%M')} to {max_time.strftime('%Y-%m-%d %H:%M')}"
                except Exception as e:
                    logger.error(f"Error calculating time range: {str(e)}")
                    time_range_str = "Unknown time range"
                
                # Create dialogue turns
                turns = [
                    MultiModalTurn(
                        turn_id=f"{session_id}_intro",
                        speaker="System",
                        content=f"Session contains {len(session_data)} medical events"
                    )
                ]
                
                session_objects.append(Session(
                    session_id=session_id,
                    time=time_range_str,
                    participants=["User", "Assistant"],
                    turns=turns,
                    tables=table_objects
                ))
            
            # Create Conversation object
            conversation_id = f"patient_{patient_id}"
            conversation = Conversation(
                conversation_id=conversation_id,
                speakers=["User", "Assistant"],
                sessions=session_objects
            )
            
            all_conversations.append(conversation)
            patient_index[patient_id] = {
                'num_sessions': len(session_objects),
                'total_events': len(patient_data)
            }
        
        # 6. Save index file
        index_path = os.path.join(self.output_dir, "patient_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(patient_index, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved patient index file: {index_path}")
        
        # 7. Create and return dataset
        dataset = ConversationDataset(conversations=all_conversations)
        logger.info(f"Data processing complete, generated: {len(all_conversations)} conversations")
        return dataset
    
    def _create_sessions_for_patient(self, patient_id: str, patient_data: pd.DataFrame) -> List[pd.DataFrame]:
        """Create time-clustered sessions for a single patient"""
        # Sort events by time
        sorted_events = patient_data.sort_values('time_event').reset_index(drop=True)
        
        sessions = []
        current_session = []
        current_window_end = None
        
        for idx, event in sorted_events.iterrows():
            event_time = event['time_event']
            
            # If first event
            if not current_session:
                current_session.append(event)
                current_window_end = event_time + timedelta(hours=self.time_window_hours)
                continue
            
            # Check if event is within time window and hasn't reached max count
            if (event_time <= current_window_end) and (len(current_session) < self.max_events_per_session):
                current_session.append(event)
            else:
                # Save current session and start new one
                if current_session:  # Ensure session is not empty
                    sessions.append(pd.DataFrame(current_session))
                current_session = [event]
                current_window_end = event_time + timedelta(hours=self.time_window_hours)
        
        # Add last session
        if current_session:
            sessions.append(pd.DataFrame(current_session))
        
        logger.info(f"Patient {patient_id} divided into {len(sessions)} sessions")
        return sessions
    
    def _create_table_objects(self, session_data: pd.DataFrame) -> List[Table]:
        """Create Table objects from session data"""
        if session_data.empty:
            return []
        
        # Group by table type
        table_objects = []
        for table_type, group in session_data.groupby('table_type'):
            # Create appropriate column names
            if table_type == 'ChemistryEvents':
                headers = ['PatientID', 'time_event', 'variable_name', 'value']
                # Check if columns exist
                available_headers = [col for col in headers if col in group.columns]
                # Remove rows with NaN values
                cleaned_group = group[available_headers].dropna()
                rows = cleaned_group.to_dict('records')
            elif table_type == 'ABGEvents':
                headers = ['PatientID', 'time_event', 'abg_ventilator_mode', 'variable_name', 'value']
                available_headers = [col for col in headers if col in group.columns]
                cleaned_group = group[available_headers].dropna()
                rows = cleaned_group.to_dict('records')
            elif table_type == 'CultureEvents':
                headers = ['PatientID', 'time_event', 'culture_source', 'value']
                available_headers = [col for col in headers if col in group.columns]
                cleaned_group = group[available_headers].dropna()
                rows = cleaned_group.to_dict('records')
            elif table_type == 'CBCEvents':
                headers = ['PatientID', 'time_event', 'variable_name', 'value']
                available_headers = [col for col in headers if col in group.columns]
                cleaned_group = group[available_headers].dropna()
                rows = cleaned_group.to_dict('records')
            else:
                # Default handling
                headers = list(group.columns)
                cleaned_group = group.dropna()
                rows = cleaned_group.to_dict('records')
            
            table_objects.append(Table(headers=available_headers, rows=rows))
        
        return table_objects

    def save(self, dataset: ConversationDataset):
        """Save processed dataset"""
        output_path = os.path.join(self.output_dir, "processed_dataset.json")
        
        serialized = []
        for conversation in dataset.conversations:
            conv_data = {
                "conversation_id": conversation.id,
                "speakers": conversation.speakers,
                "sessions": []
            }
            for session in conversation.sessions:
                session_data = {
                    "session_id": session.id,
                    "time": session.time,
                    "participants": session.participants,
                    "turns": [
                        {
                            "turn_id": turn.id,
                            "speaker": turn.speaker,
                            "content": turn.content
                        } for turn in session.turns
                    ],
                    "tables": []
                }
                
                # Process table data
                for table in session.tables:
                    table_data = {
                        "headers": table.headers,
                        "rows": []
                    }
                    
                    # Convert each row of data
                    for row in table.rows:
                        converted_row = {}
                        for key, value in row.items():
                            if isinstance(value, pd.Timestamp):
                                # Convert Timestamp to string
                                converted_row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                converted_row[key] = value
                        table_data["rows"].append(converted_row)
                    
                    session_data["tables"].append(table_data)
                
                conv_data["sessions"].append(session_data)
            serialized.append(conv_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed dataset to: {output_path}")

def main():
    import argparse
    import time
    from utils.logger import setup_logging
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"medical_loader_{timestamp}.log"
    logger = setup_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Medical Data Processing Tool')
    parser.add_argument('--input_dir', type=str, default='artifacts/raw/medical',
                        help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default='artifacts/med_processed',
                        help='Path to processed output directory')
    parser.add_argument('--max_events', type=int, default=8,
                        help='Maximum events per session')
    parser.add_argument('--time_window', type=int, default=3,
                        help='Time window size (hours)')
    
    args = parser.parse_args()
    
    # Create loader and process data
    loader = MedicalLoader(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_events_per_session=args.max_events,
        time_window_hours=args.time_window
    )
    
    dataset = loader.load_and_process()
    if dataset:
        loader.save(dataset)
    
    logger.info("Medical data processing completed")

if __name__ == '__main__':
    main()