import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset
from utils.session_simulator import SessionSimulator
from utils.prompt_templates import PERSONA

logger = logging.getLogger(__name__)

class MedicalLoader:
    def __init__(self, input_dir: str, output_dir: str, 
                 max_events_per_session: int, 
                 time_window_hours: int,
                 generate_pseudo_dialogue: bool,
                 model: str,
                 cache_dir: str,
                 max_turns: int = 5,
                 is_step: bool = True):
        """
        Medical Data Loader
        
        Parameters:
        input_dir: Path to raw data directory
        output_dir: Path to processed output directory
        max_events_per_session: Maximum events per session
        time_window_hours: Time window size (hours)
        generate_pseudo_dialogue: Whether to generate pseudo dialogues
        model: Model to use for dialogue generation
        cache_dir: Directory for caching generated dialogues
        max_turns: Maximum number of dialogue turns (default: 5)
        is_step: Whether to enable step-by-step generation with pauses (default: True)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_events_per_session = max_events_per_session
        self.time_window_hours = time_window_hours
        self.generate_pseudo_dialogue = generate_pseudo_dialogue
        self.model = model
        self.cache_dir = cache_dir
        self.max_turns = max_turns
        self.is_step = is_step
        
        # Initialize session simulator if needed
        if self.generate_pseudo_dialogue:
            self.session_simulator = SessionSimulator(
                model=self.model, 
                max_turns=self.max_turns,
                is_step=self.is_step,
                cache_dir=self.cache_dir,
                domain="medical"
            )
            self.persona = PERSONA["medical"]
        
        self.table_files = {
            'ChemistryEvents': 'ChemistryEvents.csv',
            'ABGEvents': 'ABGEvents.csv',
            'CultureEvents': 'CultureEvents.csv',
            'CBCEvents': 'CBCEvents.csv'
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
                    
                    # 在加载时立即转换时间戳为字符串
                    if 'time_event' in df.columns:
                        df['time_event'] = pd.to_datetime(
                            df['time_event'], 
                            errors='coerce'
                        )
                        # 转换时间戳为字符串格式
                        df['time_event'] = df['time_event'].apply(
                            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(x) else ""
                        )
                    
                    # 处理特定表格的列
                    if table_name == 'ABGEvents':
                        df['variable_name'] = df['abg_ventilator_mode'] + '-' + df['abg_name']
                        df.drop(columns=['abg_ventilator_mode', 'abg_name'], inplace=True)
                    elif table_name == 'CultureEvents':
                        df['value'] = df['result']
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
        
        # 2. Merge all table data
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        logger.info(f"Merged data count: {len(combined_df)}")
        
        # 3. Delete invalid time records
        combined_df = combined_df[combined_df['time_event'] != ""]
        logger.info(f"Valid events count: {len(combined_df)}")
        
        # 4. Group by patient
        grouped = combined_df.groupby('PatientID')
        all_conversations = []
        
        # Create patient session index
        patient_index = {}
        
        for i, (patient_id, patient_data) in enumerate(grouped):
            if i >= 1:  # Process only first 5 patients for testing
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
                session_id = f"patient_{patient_id}_session_{session_idx+1}"
                
                # Safely calculate time range
                try:
                    # 由于时间戳已经是字符串，我们可以直接使用字符串操作
                    min_time = min(session_data['time_event'])
                    max_time = max(session_data['time_event'])
                    time_range_str = f"{min_time} to {max_time}"
                except Exception as e:
                    logger.error(f"Error calculating time range: {str(e)}")
                    time_range_str = "Unknown time range"
                
                # Generate pseudo dialogue or simple intro
                turns = self._generate_turns_for_session(
                    session_id, table_objects
                )
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
    
    def _generate_turns_for_session(self, session_id: str, tables: List[Dict]) -> List[MultiModalTurn]:
        """
        为会话生成对话回合
        """
        turns = []
        
        if self.generate_pseudo_dialogue:
            # 将表格转换为证据
            evidences = self._tables_to_evidences(tables)
            
            if not evidences:
                logger.warning(f"No valid evidences generated for session {session_id}")
                return [MultiModalTurn(
                    turn_id=f"{session_id}_intro",
                    speaker="System",
                    content=f"No valid evidences could be generated"
                )]
            
            # Generate dialogue
            logger.info(f"为会话 {session_id} 生成伪对话，共有 {len(evidences)} 条证据")
            
            # 设置医疗领域的persona
            persona = PERSONA["medical"]

            # 生成对话
            dialog = self.session_simulator.generate_dialog(
                evidences=evidences,
                persona=persona
            )
            
            # 转换为回合格式
            for i, turn in enumerate(dialog):
                # 确保返回MultiModalTurn对象
                turns.append(MultiModalTurn(
                    turn_id=turn["id"],
                    speaker=turn["speaker"],
                    content=turn["content"],
                    evidence=turn.get("mentioned_evidence", [])
                ))
        else:
            # 确保返回MultiModalTurn对象
            turns.append(MultiModalTurn(
                turn_id=f"{session_id}_intro",
                speaker="System",
                content=f"Session contains medical events",
                evidence=[]
            ))

        return turns
    
    def _tables_to_evidences(self, tables: List[Table]) -> List[Tuple]:
        """
        将表格转换为证据元组列表
        Evidence = Tuple[patient_id, timestamp, table_type, ...其他值]
        """
        evidences = []
        for table in tables:
            table_type = getattr(table, "table_type", "Unknown")
            for row in table.rows:
                try:
                    if isinstance(row, dict):
                        row_values = tuple(row.values())
                    else:
                        row_values = tuple(row)
                    
                    evidence = (row_values[:2] + (str(table_type),) + row_values[2:])
                    
                    evidences.append(evidence)
                    logger.debug(f"evidence:{evidence}")    
                except Exception as e:
                    logger.warning(f"Error creating evidence: {str(e)}")
                    logger.debug(f"Problematic row: {row}")
        return evidences

    def _create_sessions_for_patient(self, patient_id: str, patient_data: pd.DataFrame) -> List[pd.DataFrame]:
        """Create time-clustered sessions for a single patient"""
        # 由于时间戳已经是字符串，我们需要转换为datetime进行时间计算
        patient_data = patient_data.copy()
        patient_data['time_event_dt'] = pd.to_datetime(patient_data['time_event'], errors='coerce')
        patient_data = patient_data.dropna(subset=['time_event_dt'])
        
        # Sort events by time
        sorted_events = patient_data.sort_values('time_event_dt').reset_index(drop=True)
        
        sessions = []
        current_session = []
        current_window_end = None
        
        for idx, event in sorted_events.iterrows():
            event_time = event['time_event_dt']
            
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
                    # 删除临时datetime列
                    session_df = pd.DataFrame(current_session).drop(columns=['time_event_dt'])
                    sessions.append(session_df)
                current_session = [event]
                current_window_end = event_time + timedelta(hours=self.time_window_hours)
        
        # Add last session
        if current_session:
            session_df = pd.DataFrame(current_session).drop(columns=['time_event_dt'])
            sessions.append(session_df)
        
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
                headers = ['PatientID', 'time_event', 'variable_name', 'value']
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
            
            # Create table with additional metadata
            table = Table(headers=available_headers, rows=rows)
            table.table_type = table_type  # Add table type as attribute
            table_objects.append(table)
        
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
                            "content": turn.content,
                            "mentioned_evidence": turn.mentioned_evidence
                        } for turn in session.turns
                    ],
                    "tables": []
                }
                
                # Process table data
                for table in session.tables:
                    table_data = {
                        "headers": table.headers,
                        "rows": table.rows,
                        "table_type": getattr(table, 'table_type', 'Unknown')
                    }
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
    parser.add_argument('--generate_pseudo_dialogue', action='store_true',
                        help='Generate pseudo dialogues')
    parser.add_argument('--model', type=str, default='qwen-turbo-latest',
                        help='Model to use for dialogue generation')
    parser.add_argument('--cache_dir', type=str, default='artifacts/med_processed/cache',
                        help='Cache directory for generated dialogues')
    # 新增参数
    parser.add_argument('--max_turns', type=int, default=5,
                        help='Maximum number of dialogue turns')
    parser.add_argument('--is_step', action='store_true',
                        help='Enable step-by-step generation with pauses')
    
    args = parser.parse_args()
    
    # Create loader and process data
    loader = MedicalLoader(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_events_per_session=args.max_events,
        time_window_hours=args.time_window,
        generate_pseudo_dialogue=args.generate_pseudo_dialogue,
        model=args.model,
        cache_dir=args.cache_dir,
        max_turns=args.max_turns,
        is_step=args.is_step
    )
    
    dataset = loader.load_and_process()
    if dataset:
        loader.save(dataset)
    
    logger.info("Medical data processing completed")

if __name__ == '__main__':
    main()